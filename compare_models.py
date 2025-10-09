#!/usr/bin/env python3
import os
import re
import csv
import argparse
import time
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F

import aux_funcs as af
import network_architectures as arcs
from data import accuracy, AverageMeter

# 可选导入matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

print("python compare_models.py --models-path networks/1221")
print("python3 compare_models.py --models-path networks/1221 --device cuda:0 --thresholds 0.8,0.9,0.95")
def list_candidate_model_names(models_path: str):
    """
    尽可能发现可加载的模型名（不依赖 *_params.json）
    规则：
      - 子目录名
      - *_params.(json|pkl|pickle|pt|pth) 的前缀
      - *_epoch_*.pt/pth 的前缀
    成功用 arcs.load_model 验证后才返回。
    """
    if not os.path.isdir(models_path):
        return []
    entries = os.listdir(models_path)
    names = set()

    # 子目录
    for d in entries:
        if os.path.isdir(os.path.join(models_path, d)):
            names.add(d)

    # 文件名前缀
    param_suffixes = ['_params.json', '_params.pkl', '_params.pickle', '_params.pt', '_params.pth']
    for f in entries:
        lower = f.lower()
        for suf in param_suffixes:
            if lower.endswith(suf):
                names.add(f[: -len(suf)])
                break
        if lower.endswith('.pt') or lower.endswith('.pth'):
            m = re.match(r'(.+)_epoch_\d+.*\.(pt|pth)$', f)
            if m:
                names.add(m.group(1))

    # 验证
    valid = []
    for n in sorted(names):
        try:
            _m, _p = arcs.load_model(models_path, n, -1)
            valid.append(n)
        except Exception:
            pass
    return valid


def parse_arch_from_name(model_name: str):
    return model_name.split('_', 1)[0] if '_' in model_name else model_name


def param_count_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def eval_full_outputs(model, dataloader, device):
    """
    评测每个出口Top1/Top5，并测量完整前向（计算所有出口）的平均batch延迟(ms)。
    """
    model.eval()
    num_exits = None
    meters_top1, meters_top5 = [], []
    total_batches, total_time = 0, 0.0

    is_cuda = str(device).startswith('cuda')

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(images)
        if is_cuda:
            torch.cuda.synchronize()
        total_time += (time.perf_counter() - t0) * 1000.0  # ms
        total_batches += 1

        out_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

        if num_exits is None:
            num_exits = len(out_list)
            meters_top1 = [AverageMeter() for _ in range(num_exits)]
            meters_top5 = [AverageMeter() for _ in range(num_exits)]

        for i, logits in enumerate(out_list):
            t1 = accuracy(logits, targets, topk=(1,))[0]
            t5 = accuracy(logits, targets, topk=(5,))[0]
            meters_top1[i].update(t1.item(), n=images.size(0))
            meters_top5[i].update(t5.item(), n=images.size(0))

    exits_top1 = [m.avg for m in meters_top1]
    exits_top5 = [m.avg for m in meters_top5]
    avg_ms_per_batch = (total_time / max(total_batches, 1))
    return exits_top1, exits_top5, avg_ms_per_batch


def _infer_costs(model, num_exits):
    """
    推断每个出口的相对计算开销（0-1）。优先读取模型属性；否则用线性近似。
    """
    for attr in ['exit_flops', 'exits_flops', 'ic_costs', 'exit_costs', 'branch_costs']:
        if hasattr(model, attr):
            costs = getattr(model, attr)
            if isinstance(costs, torch.Tensor):
                costs = costs.detach().cpu().tolist()
            if isinstance(costs, (list, tuple)) and len(costs) == num_exits:
                # 归一化到 [0,1]
                mx = max(float(x) for x in costs) or 1.0
                return [float(x) / mx for x in costs]
    # 线性近似：第k个出口成本约为 k/E
    return [(i + 1) / num_exits for i in range(num_exits)]


@torch.no_grad()
def eval_early_exit(model, dataloader, device, thresholds):
    """
    评估早退策略：对每个阈值，统计
      - top1
      - 平均退出出口(avg_exit_idx, 1-based)
      - 出口分布(exit_counts)
      - 平均相对成本(avg_rel_cost)，基于每出口的相对成本向量（推断或线性近似）
    实现方式为“模拟策略”，不会减少一次前向的真实计算量。
    """
    model.eval()
    num_exits = None
    total_samples = 0
    results = {thr: {'correct': 0, 'sum_exit': 0, 'exit_counts': None, 'sum_rel_cost': 0.0} for thr in thresholds}
    costs_tensor = None  # [E]

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        out_list = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

        if num_exits is None:
            num_exits = len(out_list)
            for thr in thresholds:
                results[thr]['exit_counts'] = [0 for _ in range(num_exits)]
            # 准备相对成本
            costs = _infer_costs(model, num_exits)
            costs_tensor = torch.tensor(costs, device=device, dtype=torch.float32)

        B = targets.size(0)
        total_samples += B

        # 堆叠各出口logits与置信度
        logits_stack = torch.stack(out_list, dim=0)  # [E,B,C]
        confs = []
        for logits in out_list:
            pmax = F.softmax(logits, dim=1).max(dim=1).values  # [B]
            confs.append(pmax)
        confs = torch.stack(confs, dim=0)  # [E,B]

        # 每个阈值独立决策
        for thr in thresholds:
            meets = confs >= thr  # [E,B] bool
            has_meet = meets.any(dim=0)  # [B]
            first_idx = meets.float().argmax(dim=0)  # [B], 若全False则为0
            chosen_idx = torch.where(has_meet, first_idx, torch.full_like(first_idx, num_exits - 1))  # [B], 0..E-1

            # 选取对应出口的logits：gather按dim=0
            gathered = torch.gather(
                logits_stack,
                0,
                chosen_idx.view(1, B, 1).expand(1, B, logits_stack.size(2))
            ).squeeze(0)  # [B,C]
            preds = gathered.argmax(dim=1)  # [B]

            correct = (preds == targets).sum().item()
            results[thr]['correct'] += correct

            # 出口分布与平均出口
            exit_idx_1based = (chosen_idx + 1).to(torch.int64)  # [B]
            results[thr]['sum_exit'] += int(exit_idx_1based.sum().item())
            # 统计直方图
            for k in range(num_exits):
                results[thr]['exit_counts'][k] += int((chosen_idx == k).sum().item())

            # 平均相对成本
            rel_cost = costs_tensor[chosen_idx]  # [B]
            results[thr]['sum_rel_cost'] += float(rel_cost.sum().item())

    # 汇总
    summary = {}
    for thr in thresholds:
        r = results[thr]
        top1 = 100.0 * r['correct'] / max(total_samples, 1)
        avg_exit = r['sum_exit'] / max(total_samples, 1)
        avg_rel_cost = r['sum_rel_cost'] / max(total_samples, 1)
        summary[thr] = {
            'top1': top1,
            'avg_exit': avg_exit,
            'exit_counts': r['exit_counts'],
            'avg_rel_cost': avg_rel_cost,
            'total': total_samples,
        }
    return summary, num_exits


def test_and_compare(models_path, device, thresholds, epoch=-1):
    """
    对目录下所有可加载模型进行评测与早退对比。
    返回记录列表用于保存与汇总。
    """
    candidates = list_candidate_model_names(models_path)
    if not candidates:
        print('No loadable models found.')
        return []

    print(f'\nDiscovered {len(candidates)} model(s):')
    for n in candidates:
        print('  -', n)

    records = []
    for name in candidates:
        print('\n' + '=' * 100)
        print(f'Model: {name}')
        model, params = arcs.load_model(models_path, name, epoch)
        dataset = af.get_dataset(params['task'])
        arch = parse_arch_from_name(name)

        model.to(device)

        # 基础评测
        exits_top1, exits_top5, ms_per_batch = eval_full_outputs(model, dataset.test_loader, device)
        num_exits = len(exits_top1)
        p_m = param_count_m(model)

        # 早退策略
        ee_summary, ee_num_exits = eval_early_exit(model, dataset.test_loader, device, thresholds)
        assert ee_num_exits == num_exits

        # 打印摘要
        if num_exits == 1:
            print(f'  Final: Top-1 {exits_top1[0]:.2f}% | Top-5 {exits_top5[0]:.2f}% '
                  f'| Params {p_m:.2f}M | Latency {ms_per_batch:.2f} ms/batch')
        else:
            for i, (t1, t5) in enumerate(zip(exits_top1, exits_top5), 1):
                print(f'  Exit {i}: Top-1 {t1:.2f}% | Top-5 {t5:.2f}%')
            print(f'  Final: Top-1 {exits_top1[-1]:.2f}% | Top-5 {exits_top5[-1]:.2f}% '
                  f'| Params {p_m:.2f}M | Latency {ms_per_batch:.2f} ms/batch')

            for thr in thresholds:
                s = ee_summary[thr]
                dist = ', '.join(str(x) for x in s["exit_counts"])
                print(f'    EE thr={thr:.2f}: Top-1 {s["top1"]:.2f}% | avg_exit {s["avg_exit"]:.2f} / {num_exits} '
                      f'| rel_cost {s["avg_rel_cost"]:.3f} | dist [{dist}]')

        # 组织记录（用于导出）
        rec = {
            'model_name': name,
            'task': params.get('task', 'unknown'),
            'arch': arch,
            'num_exits': num_exits,
            'params_m': p_m,
            'lat_ms_per_batch': ms_per_batch,
            'final_top1': exits_top1[-1],
            'final_top5': exits_top5[-1],
            'exits_top1': exits_top1,
            'exits_top5': exits_top5,
            'ee': ee_summary,
        }
        records.append(rec)
    return records


def save_compare_csv_and_md(records, thresholds, out_dir='outputs/test_results'):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'compare_summary_{ts}.csv')
    md_path = os.path.join(out_dir, f'compare_report_{ts}.md')

    # CSV：将每个模型一行展开
    # exit列 & 每个阈值的早退统计
    max_exits = max(r['num_exits'] for r in records) if records else 1
    fieldnames = [
        'model_name', 'task', 'arch', 'num_exits', 'params_m', 'lat_ms_per_batch',
        'final_top1', 'final_top5'
    ]
    for i in range(1, max_exits + 1):
        fieldnames += [f'exit{i}_top1', f'exit{i}_top5']
    for thr in thresholds:
        tag = f'{thr:.2f}'
        fieldnames += [f'ee_top1@{tag}', f'ee_avg_exit@{tag}', f'ee_rel_cost@{tag}']

    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in records:
            row = {
                'model_name': r['model_name'],
                'task': r['task'],
                'arch': r['arch'],
                'num_exits': r['num_exits'],
                'params_m': f'{r["params_m"]:.2f}',
                'lat_ms_per_batch': f'{r["lat_ms_per_batch"]:.2f}',
                'final_top1': f'{r["final_top1"]:.2f}',
                'final_top5': f'{r["final_top5"]:.2f}',
            }
            for i in range(r['num_exits']):
                row[f'exit{i+1}_top1'] = f'{r["exits_top1"][i]:.2f}'
                row[f'exit{i+1}_top5'] = f'{r["exits_top5"][i]:.2f}'
            for thr in thresholds:
                tag = f'{thr:.2f}'
                s = r['ee'][thr]
                row[f'ee_top1@{tag}'] = f'{s["top1"]:.2f}'
                row[f'ee_avg_exit@{tag}'] = f'{s["avg_exit"]:.2f}'
                row[f'ee_rel_cost@{tag}'] = f'{s["avg_rel_cost"]:.3f}'
            wr.writerow(row)

    # Markdown：按任务分组展示
    def fmt_pct(x, n=2): return f'{x:.{n}f}%'
    def fmt_ms(x): return f'{x:.2f} ms/batch'

    by_task = defaultdict(list)
    for r in records:
        by_task[r['task']].append(r)
    for t in by_task:
        by_task[t].sort(key=lambda x: x['final_top1'], reverse=True)

    with open(md_path, 'w') as f:
        f.write('# Model Comparison Report\n\n')
        f.write(f'Generated: {ts}\n\n')
        for task, rows in by_task.items():
            f.write(f'## Task: {task}\n\n')
            for r in rows:
                f.write(f'- {r["model_name"]} (arch={r["arch"]}, exits={r["num_exits"]}, '
                        f'params={r["params_m"]:.2f}M, latency={fmt_ms(r["lat_ms_per_batch"])})\n')
                if r['num_exits'] == 1:
                    f.write(f'  - Final: Top-1 {fmt_pct(r["final_top1"])}, Top-5 {fmt_pct(r["final_top5"])}\n')
                else:
                    # 出口精度
                    exit_str = '; '.join([f'E{i+1} {fmt_pct(r["exits_top1"][i])}/{fmt_pct(r["exits_top5"][i])}'
                                          for i in range(r['num_exits'])])
                    f.write(f'  - Exits (Top-1/Top-5): {exit_str}\n')
                    # 早退
                    for thr in thresholds:
                        s = r['ee'][thr]
                        dist = ', '.join(str(x) for x in s['exit_counts'])
                        f.write(f'  - EE thr={thr:.2f}: Top-1 {fmt_pct(s["top1"])}, '
                                f'avg_exit {s["avg_exit"]:.2f}/{r["num_exits"]}, '
                                f'rel_cost {s["avg_rel_cost"]:.3f}, '
                                f'dist [{dist}]\n')
                f.write('\n')

    print(f'\nSaved:')
    print(f'- {csv_path}')
    print(f'- {md_path}')
    return csv_path, md_path


def plot_reports(records, thresholds, out_dir='outputs/test_results', dpi=160):
    """
    生成可视化图表：
      - 每任务 Final Top-1 排行条形图
      - 每任务 Latency vs Final Top-1 散点图
      - 每任务 SDN 出口精度阶梯图
      - 每任务 早退 Top-1 vs 平均相对成本曲线（多阈值）
      - 每任务 早退出口分布堆叠图（使用最高阈值）
    """
    if not HAS_MPL:
        print('Matplotlib 未安装，跳过绘图。pip install matplotlib')
        return

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    from collections import defaultdict
    by_task = defaultdict(list)
    for r in records:
        by_task[r['task']].append(r)

    for task, rows in by_task.items():
        # 1) Final Top-1 排行
        rows_sorted = sorted(rows, key=lambda x: x['final_top1'], reverse=True)
        labels = [r['model_name'] for r in rows_sorted]
        vals = [r['final_top1'] for r in rows_sorted]
        plt.figure(figsize=(max(6, 0.5*len(labels)), 4))
        plt.bar(range(len(vals)), vals, color='#4C78A8')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
        plt.ylabel('Final Top-1 (%)')
        plt.title(f'{task}: Final Top-1')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{task}_final_top1_{ts}.png'), dpi=dpi)
        plt.close()

        # 2) Latency vs Accuracy
        plt.figure(figsize=(6, 4))
        xs = [r['lat_ms_per_batch'] for r in rows]
        ys = [r['final_top1'] for r in rows]
        plt.scatter(xs, ys, c='#F58518')
        for r in rows:
            plt.annotate(r['arch'], (r['lat_ms_per_batch'], r['final_top1']), fontsize=7, alpha=0.8)
        plt.xlabel('Latency (ms/batch)')
        plt.ylabel('Final Top-1 (%)')
        plt.title(f'{task}: Latency vs Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{task}_lat_vs_acc_{ts}.png'), dpi=dpi)
        plt.close()

        # 3) SDN 出口精度阶梯图
        sdn_rows = [r for r in rows if r['num_exits'] > 1]
        if sdn_rows:
            plt.figure(figsize=(6, 4))
            for r in sdn_rows:
                xs = list(range(1, r['num_exits'] + 1))
                plt.plot(xs, r['exits_top1'], marker='o', label=r['arch'])
            plt.xlabel('Exit index')
            plt.ylabel('Top-1 (%)')
            plt.title(f'{task}: Exit accuracy ladder')
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{task}_exit_ladder_{ts}.png'), dpi=dpi)
            plt.close()

        # 4) 早退 Top-1 vs 相对成本（多阈值曲线）
        if thresholds:
            plt.figure(figsize=(6, 4))
            for r in rows:
                xs, ys = [], []
                for thr in thresholds:
                    s = r['ee'][thr]
                    xs.append(s['avg_rel_cost'])
                    ys.append(s['top1'])
                plt.plot(xs, ys, marker='o', label=r['arch'])
            plt.xlabel('Avg relative cost')
            plt.ylabel('Top-1 (%)')
            plt.title(f'{task}: Early-exit tradeoff')
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{task}_ee_tradeoff_{ts}.png'), dpi=dpi)
            plt.close()

        # 5) 早退出口分布堆叠图（最高阈值）
        if sdn_rows and thresholds:
            thr = max(thresholds)
            import numpy as np
            labels = [r['arch'] for r in sdn_rows]
            K = max(r['num_exits'] for r in sdn_rows)
            # 计算百分比
            perc = []
            for r in sdn_rows:
                cnts = r['ee'][thr]['exit_counts']
                if len(cnts) < K:
                    cnts = cnts + [0] * (K - len(cnts))
                tot = sum(cnts) or 1
                perc.append([c * 100.0 / tot for c in cnts])

            ind = np.arange(len(labels))
            bottom = np.zeros(len(labels))
            plt.figure(figsize=(max(6, 0.5*len(labels)), 4))
            cmap = plt.get_cmap('tab10')
            for k in range(K):
                vals = [p[k] for p in perc]
                plt.bar(ind, vals, bottom=bottom, color=cmap(k), label=f'E{k+1}')
                bottom += np.array(vals)
            plt.xticks(ind, labels, rotation=45, ha='right', fontsize=8)
            plt.ylabel(f'Exit distribution @ thr={thr:.2f} (%)')
            plt.title(f'{task}: Early-exit distribution')
            plt.legend(fontsize=8, ncol=min(5, K))
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{task}_ee_dist_thr{thr:.2f}_{ts}.png'), dpi=dpi)
            plt.close()


def main():
    ap = argparse.ArgumentParser(description='Compare trained CNN/SDN models with early-exit analysis.')
    ap.add_argument('--models-path', required=True, help='e.g., networks/1221')
    ap.add_argument('--device', default=None, help='cuda, cuda:0, or cpu (default: auto)')
    ap.add_argument('--epoch', type=int, default=-1, help='epoch to load (-1 latest)')
    ap.add_argument('--thresholds', default='0.70,0.80,0.90,0.95', help='comma-separated confidence thresholds')
    ap.add_argument('--only', nargs='*', help='only test these model names (subdir names)')
    ap.add_argument('--skip', nargs='*', default=[], help='skip these model names')
    ap.add_argument('--plot', action='store_true', default=False, help='generate visualization PNGs')
    args = ap.parse_args()

    device = args.device or af.get_pytorch_device()
    thr_list = [float(x) for x in args.thresholds.split(',') if x.strip()]
    print(f'Using PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
    print(f'Using device: {device}')
    print(f'Models path: {args.models_path}')
    print(f'Thresholds: {thr_list}')

    # 可选子集过滤
    all_names = list_candidate_model_names(args.models_path)
    if args.only:
        candidates = [n for n in args.only if n in all_names]
    else:
        candidates = all_names
    candidates = [n for n in candidates if n not in set(args.skip)]

    if not candidates:
        print('No loadable models found under this path.')
        if os.path.isdir(args.models_path):
            print('\nDirectory listing (first 50):')
            for f in sorted(os.listdir(args.models_path))[:50]:
                print('  -', f)
        return

    # 执行评测
    records = []
    for name in candidates:
        try:
            recs = test_and_compare(args.models_path, device, thr_list, epoch=args.epoch)
            records.extend(recs)
            break  # 已在内部对全部候选做过发现与循环（保持与 test_networks 行为一致）
        except Exception as e:
            print(f'Failed to test models: {e}')
            return

    # 保存
    if records:
        save_compare_csv_and_md(records, thr_list, out_dir='outputs/test_results')
        if args.plot:
            plot_reports(records, thr_list, out_dir='outputs/test_results')


if __name__ == '__main__':
    main()