"""
通用 PGD 攻击脚本（支持 SDN 多出口模型）

原始版本针对 BranchyNet(ResNet18) 只有两个早退分支 (e1,e2) + 最终输出。
本修改版本支持本仓库中的 SDN 模型：forward(x) -> [ic_0, ic_1, ..., ic_{N-2}, final]

攻击目标：
  在保持最终分类正确 (通过交叉熵或 CW 形式) 的前提下，提高早期分支的归一化熵(=降低置信度)，
  使样本更可能“逃过”早退，从而强制走到更深层(或改变早退分布)。

可选：使用 PCGrad 以减少“保持正确” 与 “提升早期熵” 两组梯度冲突。

指标：
  1.  Clean 与 Adv 的最终准确率 (只在原本正确的样本上尝试扰动)
  2.  早退出口分布变化（依据提供的 entropy 阈值列表决定早退）
  3.  扰动大小 (Linf / L2)

使用方式示例：
  python attack.py \
    --models_path networks/1221 \
    --model_name cifar10_resnet56_sdn \
    --dataset cifar10 \
    --entropy_thresholds 0.2,0.2,0.2,0.2,0.2 \
    --lambda_exits 1.5,1.2,1.0,1.0,1.0 \
    --eps 8/255 --alpha 2/255 --pgd_steps 20 --cw --pcgrad

若不提供 --lambda_exits，会对每个内部出口使用相同权重 (默认 1.0)。
若阈值个数少于内部出口，则自动补齐；多于则截断。
"""

import math
import argparse
import os
import re
import csv
from datetime import datetime
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
# torchvision 的直接加载已被项目自带的标准化数据加载替代
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import tqdm

import network_architectures as arcs
import aux_funcs as af

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


def parse_float_expr(v: str) -> float:
    """允许 8/255 这类表达式。"""
    v = v.strip()
    if '/' in v:
        num, den = v.split('/')
        return float(num) / float(den)
    return float(v)


def build_dataset(name: str, root: str, resize_to: Optional[int] = None):
    name = name.lower()
    if name == 'cifar10':
        transform_list = []
        if resize_to:
            transform_list.append(transforms.Resize(resize_to))
        transform_list.extend([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.Compose(transform_list))
    elif name == 'cifar100':
        transform_list = []
        if resize_to:
            transform_list.append(transforms.Resize(resize_to))
        transform_list.extend([transforms.ToTensor()])
        testset = datasets.CIFAR100(root=root, train=False, download=True, transform=transforms.Compose(transform_list))
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return testset


def normalized_entropy_from_logits(logits):
    """返回归一化熵 (batch,)（除以 log(C)）用于 hinge 判定"""
    probs = F.softmax(logits, dim=1).clamp(min=1e-12)
    ent = -(probs * probs.log()).sum(dim=1)  # natural log
    denom = math.log(max(2, logits.size(1)))
    return ent / denom


def select_exits(outputs: List[torch.Tensor], thresholds: List[float], force_exit_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """依据 entropy 阈值列表决定早退出口。
    outputs: [ic_0, ..., ic_{N-2}, final]
    thresholds: len == N-1 (内部 IC 个数)
    返回: (pred_labels, exit_ids)  exit_id 从 1 开始，final 为 N
    """
    num_internal = len(outputs) - 1
    final_idx = len(outputs)  # 1-based id
    batch = outputs[0].size(0)
    # 若强制出口，直接返回该出口的预测
    if force_exit_id is not None and 1 <= force_exit_id <= final_idx:
        forced_logits = outputs[force_exit_id - 1]
        preds = forced_logits.argmax(dim=1)
        exit_ids = torch.full((batch,), force_exit_id, dtype=torch.long, device=forced_logits.device)
        return preds.cpu(), exit_ids.cpu()
    # 计算所有内部 IC 的 normalized entropy
    entropies = [normalized_entropy_from_logits(logits) for logits in outputs[:-1]]
    # 默认全部走到 final（放到与输出相同的设备上）
    dev = outputs[-1].device
    exit_ids = torch.full((batch,), final_idx, dtype=torch.long, device=dev)
    chosen_logits = outputs[-1]
    for ic_id in range(num_internal):
        thr = thresholds[ic_id] if ic_id < len(thresholds) else thresholds[-1]
        mask = (exit_ids == final_idx) & (entropies[ic_id] <= thr)
        if mask.any():
            exit_ids[mask] = ic_id + 1  # 1-based
            chosen_logits = torch.where(mask.view(-1, 1), outputs[ic_id], chosen_logits)
    preds = chosen_logits.argmax(dim=1)
    return preds.cpu(), exit_ids.cpu()


def attack_batch(model, x, y, thresholds, eps, alpha, steps, lambda_exits, lambda_ce, lambda_l2,
                 cw=True, c=0.15, kappa=5, pcgrad=False, same_acc_early_loss=False,
                 force_exit_id: Optional[int] = None):
    """对 batch 做 constrained PGD，返回 (x_adv, linf, l2, preds_adv, exits_adv, pcgrad_conflict_count)

    仅对原本被正确分类的样本调用。兼容 SDN（多出口）与 CNN（单出口）。
    """
    x_orig = x.detach()
    batch_size = x.size(0)
    delta = torch.zeros_like(x, device=x.device, requires_grad=True)
    num_internal = model.num_output - 1 if hasattr(model, 'num_output') else 0

    # 若 lambda_exits 长度不足，补齐；超出截断
    if len(lambda_exits) < num_internal:
        lambda_exits = lambda_exits + [lambda_exits[-1]] * (num_internal - len(lambda_exits))
    elif len(lambda_exits) > num_internal:
        lambda_exits = lambda_exits[:num_internal]

    conflict_steps = 0
    for _ in range(steps):
        x_adv = (x_orig + delta).clamp(0.0, 1.0)
        outputs = model.forward(x_adv)
        if isinstance(outputs, list):
            internal_outputs = outputs[:-1]
            final_out = outputs[-1]
        else:
            internal_outputs = []
            final_out = outputs

        # 计算早期熵损失
        early_losses = []
        if same_acc_early_loss:
            for i, logits in enumerate(internal_outputs):
                coef = 0.1 if i == 0 else 0.05
                early_losses.append(-F.cross_entropy(logits, y) * coef)
        else:
            # 若指定强制出口，则将早期损失设置为：
            # - 对于 i < target: 提高熵，避免更早出口 -> hinge(t - nent, 0)
            # - 对于 i == target: 降低熵，触发该出口 -> hinge(nent - t, 0)
            # - 对于 i > target: 不约束（已不会到达）
            target = force_exit_id if (force_exit_id is not None and force_exit_id >= 1) else None
            for i, logits in enumerate(internal_outputs):
                nent = normalized_entropy_from_logits(logits)
                t = thresholds[i] if i < len(thresholds) else (thresholds[-1] if len(thresholds) > 0 else 0.0)
                exit_idx = i + 1  # 1-based for internal
                if target is None:
                    # 默认策略：整体提高早期熵，推动更晚退出
                    early_losses.append(torch.clamp(t - nent, min=0.0).mean())
                else:
                    if target == num_internal + 1:
                        # 强制最终出口：对所有内部出口都避免早退
                        early_losses.append(torch.clamp(t - nent, min=0.0).mean())
                    elif exit_idx < target:
                        early_losses.append(torch.clamp(t - nent, min=0.0).mean())
                    elif exit_idx == target:
                        early_losses.append(torch.clamp(nent - t, min=0.0).mean())
                    else:
                        # exit_idx > target: 无约束
                        early_losses.append(torch.zeros(1, device=final_out.device, dtype=final_out.dtype))
        if early_losses:
            loss_early_total = sum(w * l for w, l in zip(lambda_exits, early_losses))
            if not isinstance(loss_early_total, torch.Tensor):
                loss_early_total = torch.tensor(loss_early_total, device=final_out.device, dtype=final_out.dtype)
        else:
            loss_early_total = torch.zeros(1, device=final_out.device, dtype=final_out.dtype)

        # 最终输出保持正确 (CW 或 CE)
        if cw:
            batch_indices = torch.arange(batch_size, device=final_out.device)
            target_logit = final_out[batch_indices, y]
            mask = torch.nn.functional.one_hot(y, num_classes=final_out.size(1)).bool()
            other_logit = final_out.masked_fill(mask, -1e9).max(dim=1)[0]
            f = torch.clamp(other_logit - target_logit + kappa, min=0.0)
            loss_ce_term = f.mean() * c
        else:
            loss_ce_term = F.cross_entropy(final_out, y)

        loss_l2_term = (delta.view(batch_size, -1).norm(p=2, dim=1).mean())

        if pcgrad and early_losses:
            g_acc = torch.autograd.grad(loss_ce_term * lambda_ce, delta, retain_graph=True)[0]
            g_early = torch.autograd.grad(loss_early_total, delta, retain_graph=True)[0]
            g_reg = torch.autograd.grad(lambda_l2 * loss_l2_term, delta)[0]
            if torch.dot(g_acc.flatten(), g_early.flatten()) < 0:
                g_acc = g_acc - (torch.dot(g_acc.flatten(), g_early.flatten()) / (g_early.flatten().norm() ** 2 + 1e-12)) * g_early
                conflict_steps += 1
            grads = g_acc + g_early + g_reg
        else:
            total_loss = loss_early_total + lambda_ce * loss_ce_term + lambda_l2 * loss_l2_term
            grads = torch.autograd.grad(total_loss, delta, retain_graph=False)[0]

        # PGD 更新（最小化 loss）
        delta.data = delta.data - alpha * grads.sign()
        delta.data = torch.clamp(delta.data, -eps, eps)
        delta.data = torch.clamp(x_orig + delta.data, 0.0, 1.0) - x_orig
        delta.grad = None

    x_adv = (x_orig + delta).clamp(0.0, 1.0)
    with torch.no_grad():
        adv_outputs = model.forward(x_adv)
        if isinstance(adv_outputs, list):
            preds_adv, exits_adv = select_exits(adv_outputs, thresholds, force_exit_id=force_exit_id)
        else:
            preds_adv = adv_outputs.argmax(dim=1).cpu()
            exits_adv = torch.full((batch_size,), 1, dtype=torch.long)
        linf = delta.abs().view(batch_size, -1).max(dim=1)[0].cpu()
        l2 = delta.view(batch_size, -1).norm(p=2, dim=1).cpu()
    return x_adv.detach(), linf, l2, preds_adv, exits_adv, int(conflict_steps / max(1, steps))

def list_candidate_model_names(models_path: str) -> List[str]:
    """列出 models_path 下可能的模型名称，并用 load_model 校验。"""
    if not os.path.isdir(models_path):
        return []
    entries = os.listdir(models_path)
    names = set()
    # 子目录名
    for d in entries:
        if os.path.isdir(os.path.join(models_path, d)):
            names.add(d)
    # 文件名前缀匹配
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
    # 校验可加载
    valid = []
    for n in sorted(names):
        try:
            _m, _p = arcs.load_model(models_path, n, -1)
            valid.append(n)
        except Exception:
            pass
    return valid


def evaluate_one_model(models_path: str, model_name: str, args) -> dict:
    """评测单个模型并返回记录。"""
    # 加载模型
    model, params = arcs.load_model(models_path, model_name, epoch=-1)
    model.to(device)
    model.eval()
    num_internal = model.num_output - 1 if hasattr(model, 'num_output') else 0

    # 数据集（与训练一致）
    task = params.get('task', args.dataset)
    dataset = af.get_dataset(task, batch_size=args.batch_size)
    if dataset is None or not hasattr(dataset, 'test_loader'):
        # 兜底：直接调用具体加载函数
        if task == 'cifar10':
            dataset = af.load_cifar10(args.batch_size)
        elif task == 'cifar100':
            dataset = af.load_cifar100(args.batch_size)
        elif task == 'tinyimagenet':
            dataset = af.load_tinyimagenet(args.batch_size)
        else:
            raise ValueError(f"Unknown/unsupported dataset task: {task}")
    testloader = dataset.test_loader

    # 阈值 & 权重
    entropy_thresholds = [float(x) for x in args.entropy_thresholds.split(',') if x.strip() != '']
    if num_internal > 0:
        if len(entropy_thresholds) < num_internal:
            entropy_thresholds += [entropy_thresholds[-1]] * (num_internal - len(entropy_thresholds))
        elif len(entropy_thresholds) > num_internal:
            entropy_thresholds = entropy_thresholds[:num_internal]
    else:
        entropy_thresholds = []

    if args.lambda_exits and num_internal > 0:
        lambda_exits = [float(x) for x in args.lambda_exits.split(',') if x.strip() != '']
        if len(lambda_exits) < num_internal:
            lambda_exits += [lambda_exits[-1]] * (num_internal - len(lambda_exits))
        elif len(lambda_exits) > num_internal:
            lambda_exits = lambda_exits[:num_internal]
    else:
        lambda_exits = [1.0] * max(num_internal, 1)

    eps = parse_float_expr(args.eps)
    alpha = parse_float_expr(args.alpha)

    # 主循环
    total = 0
    correct_clean = 0
    correct_adv = 0
    exit_counts_clean = [0] * (num_internal + 2)  # 使用 1..num_internal+1
    exit_counts_adv = [0] * (num_internal + 2)
    avg_linf: List[float] = []
    avg_l2: List[float] = []
    pc_conflict_fraction = 0

    pbar = tqdm.tqdm(testloader, desc=f"{model_name}")
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.size(0)
        total += batch_size

        with torch.no_grad():
            outputs_clean = model.forward(imgs)
            if num_internal > 0:
                preds_clean, exits_clean = select_exits(outputs_clean, entropy_thresholds, force_exit_id=args.force_exit)
            else:
                preds_clean = outputs_clean[-1].argmax(dim=1).cpu() if isinstance(outputs_clean, list) else outputs_clean.argmax(dim=1).cpu()
                exits_clean = torch.full((batch_size,), 1, dtype=torch.long)
            final_out_clean = outputs_clean[-1] if isinstance(outputs_clean, list) else outputs_clean
            final_pred_clean = final_out_clean.argmax(dim=1)

        correct_mask = (final_pred_clean == labels).cpu()
        correct_clean += correct_mask.sum().item()
        for e in exits_clean:
            idx = int(e.item()) if num_internal > 0 else 1
            exit_counts_clean[idx] += 1

        if correct_mask.sum().item() == 0:
            for e in exits_clean:
                idx = int(e.item()) if num_internal > 0 else 1
                exit_counts_adv[idx] += 1
            correct_adv += (final_pred_clean == labels).sum().item()
            pbar.set_postfix(clean_acc=f"{correct_clean/max(1,total):.4f}", adv_acc=f"{correct_adv/max(1,total):.4f}")
            continue

        mask_idx = torch.nonzero(correct_mask, as_tuple=False).squeeze(1).to(device)
        imgs_correct = imgs[mask_idx]
        labels_correct = labels[mask_idx]

        x_adv_subset, linf_vals, l2_vals, preds_adv_subset, exits_adv_subset, conflict_flag = attack_batch(
            model, imgs_correct, labels_correct,
            entropy_thresholds, eps, alpha, args.pgd_steps,
            lambda_exits, args.lambda_ce, args.lambda_l2,
            cw=args.cw, c=args.c, kappa=args.kappa, pcgrad=args.pcgrad,
            same_acc_early_loss=args.same_acc_early_loss_value,
            force_exit_id=args.force_exit
        )
        if conflict_flag:
            pc_conflict_fraction += batch_size

        imgs_adv = imgs.clone()
        imgs_adv[mask_idx] = x_adv_subset
        with torch.no_grad():
            outputs_adv = model.forward(imgs_adv)
            if num_internal > 0:
                preds_adv_batch, exits_adv_batch = select_exits(outputs_adv, entropy_thresholds, force_exit_id=args.force_exit)
            else:
                preds_adv_batch = outputs_adv[-1].argmax(dim=1).cpu() if isinstance(outputs_adv, list) else outputs_adv.argmax(dim=1).cpu()
                exits_adv_batch = torch.full((batch_size,), 1, dtype=torch.long)
            final_out_adv = outputs_adv[-1] if isinstance(outputs_adv, list) else outputs_adv
            final_pred_adv = final_out_adv.argmax(dim=1)

        correct_adv += (final_pred_adv == labels).sum().item()
        for e in exits_adv_batch:
            idx = int(e.item()) if num_internal > 0 else 1
            exit_counts_adv[idx] += 1
        avg_linf.extend(linf_vals.cpu().tolist())
        avg_l2.extend(l2_vals.cpu().tolist())
        pbar.set_postfix(clean_acc=f"{correct_clean/total:.4f}", adv_acc=f"{correct_adv/total:.4f}")

    clean_acc = correct_clean / max(1, total)
    adv_acc = correct_adv / max(1, total)
    clean_value = sum(cnt * i for i, cnt in enumerate(exit_counts_clean) if i > 0)
    adv_value = sum(cnt * i for i, cnt in enumerate(exit_counts_adv) if i > 0)
    dist_clean = exit_counts_clean[1 : num_internal + 2]
    dist_adv = exit_counts_adv[1 : num_internal + 2]
    rec = {
        'model_name': model_name,
        'task': params.get('task', 'unknown'),
        'num_exits': (num_internal + 1) if num_internal >= 0 else 1,
        'eps': parse_float_expr(args.eps),
        'alpha': parse_float_expr(args.alpha),
        'pgd_steps': args.pgd_steps,
        'cw': bool(args.cw),
        'pcgrad': bool(args.pcgrad),
        'clean_final_acc': clean_acc * 100.0,
        'adv_final_acc': adv_acc * 100.0,
        'avg_exit_clean': clean_value / max(1, total),
        'avg_exit_adv': adv_value / max(1, total),
        'exit_dist_clean': dist_clean,
        'exit_dist_adv': dist_adv,
        'avg_linf': (sum(avg_linf)/len(avg_linf)) if avg_linf else 0.0,
        'avg_l2': (sum(avg_l2)/len(avg_l2)) if avg_l2 else 0.0,
        'thresholds': entropy_thresholds,
        'lambda_exits': lambda_exits,
        'pcgrad_conflict_frac': pc_conflict_fraction / max(1, total),
        'forced_exit': int(args.force_exit) if args.force_exit is not None else None,
    }
    if str(device).startswith('cuda'):
        del model
        torch.cuda.empty_cache()
    return rec


def save_attack_summary(records: List[dict], out_dir='outputs/test_results'):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'attack_summary_{ts}.csv')
    md_path = os.path.join(out_dir, f'attack_summary_{ts}.md')

    fields = [
        'model_name','task','num_exits','eps','alpha','pgd_steps','cw','pcgrad',
        'clean_final_acc','adv_final_acc','avg_exit_clean','avg_exit_adv','avg_linf','avg_l2',
        'thresholds','lambda_exits','exit_dist_clean','exit_dist_adv','pcgrad_conflict_frac','forced_exit'
    ]
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in records:
            row = r.copy()
            row['thresholds'] = ','.join(f"{x:.2f}" for x in r.get('thresholds', []))
            le = r.get('lambda_exits')
            if isinstance(le, (list, tuple)):
                row['lambda_exits'] = ','.join(f"{float(x):.2f}" for x in le)
            else:
                row['lambda_exits'] = le
            # 序列化出口分布
            edc = r.get('exit_dist_clean', [])
            eda = r.get('exit_dist_adv', [])
            row['exit_dist_clean'] = ','.join(str(int(x)) for x in edc)
            row['exit_dist_adv'] = ','.join(str(int(x)) for x in eda)
            wr.writerow(row)

    with open(md_path, 'w') as f:
        f.write('# Attack Summary\n\n')
        f.write(f'Generated: {ts}\n\n')
        for r in records:
            f.write(f"- {r['model_name']} (task={r['task']}, exits={r['num_exits']})\n")
            f.write(f"  - Clean/Adv: {r['clean_final_acc']:.2f}% / {r['adv_final_acc']:.2f}%\n")
            f.write(f"  - Avg exit (clean/adv): {r['avg_exit_clean']:.2f} / {r['avg_exit_adv']:.2f}\n")
            f.write(f"  - Norms (Linf/L2): {r['avg_linf']:.6f} / {r['avg_l2']:.6f}\n")
            ths = r.get('thresholds', [])
            f.write(f"  - Thresholds: {','.join(f'{x:.2f}' for x in ths) if ths else '-'}\n\n")
            fe = r.get('forced_exit', None)
            if fe is not None:
                f.write(f"  - Forced exit: {fe}\n\n")

    print(f"Saved summary:\n- {csv_path}\n- {md_path}")


def main():
    parser = argparse.ArgumentParser(description='PGD Attack for SDN Models')
    parser.add_argument('--models_path', type=str, default='networks/3221', help='路径: 存放训练模型的根目录')
    parser.add_argument('--model_name', type=str, default=None, help='单模型名；为空则批量评测 models_path 下所有可加载模型')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'], help='数据集名称')
    parser.add_argument('--data_root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--entropy_thresholds', type=str, default='0.70,0.80,0.90,0.95', help='逗号分隔，对应每个内部出口')
    parser.add_argument('--lambda_exits', type=str, default=None, help='逗号分隔，对应各内部出口权重；缺省=全部1.0')
    parser.add_argument('--lambda_ce', type=float, default=0.5)
    parser.add_argument('--lambda_l2', type=float, default=0.01)
    parser.add_argument('--eps', type=str, default='8/255')
    parser.add_argument('--alpha', type=str, default='2/255')
    parser.add_argument('--pgd_steps', type=int, default=20)
    parser.add_argument('--cw', action='store_true', default=True, help='使用 CW 风格保持正确 (默认 False=CE)')
    parser.add_argument('--c', type=float, default=0.55, help='CW 系数 c')
    parser.add_argument('--kappa', type=float, default=5.0, help='CW 置信度 margin')
    parser.add_argument('--pcgrad', action='store_true', help='是否使用 PCGrad', default=True)
    parser.add_argument('--same_acc_early_loss_value', action='store_true', help='使用和原始示例相似的早期分支负交叉熵法')
    parser.add_argument('--device', type=str, default='cuda:0', help='手动指定设备，如 cuda:0')
    parser.add_argument('--only', nargs='*', help='仅评测这些模型名（可多个）')
    parser.add_argument('--skip', nargs='*', default=[], help='跳过这些模型名')
    parser.add_argument('--skip_contains', nargs='*', default=["cnn", "training"], help='排除名字中包含这些子串的模型（多值 OR）')
    parser.add_argument('--force_exit', type=int, default=None, help='强制模型在指定出口做决策（1..N；N为最终出口）')
    parser.add_argument('--out_dir', type=str, default='outputs/test_results', help='导出结果目录')
    args = parser.parse_args()

    global device
    if args.device:
        device = torch.device(args.device)

    # 单模型或批量评测
    records: List[dict] = []
    if args.model_name:
        rec = evaluate_one_model(args.models_path, args.model_name, args)
        records.append(rec)
    else:
        all_names = list_candidate_model_names(args.models_path)
        if args.only:
            names = [n for n in args.only if n in all_names]
        else:
            names = all_names
        names = [n for n in names if n not in set(args.skip)]
        if args.skip_contains:
            substrs = list(args.skip_contains)
            names = [n for n in names if not any(s in n for s in substrs)]
        if not names:
            print('No loadable models found under this path.')
            if os.path.isdir(args.models_path):
                print('\nDirectory listing (first 50):')
                for f in sorted(os.listdir(args.models_path))[:50]:
                    print('  -', f)
            return
        print(f"Discovered {len(names)} model(s):")
        for n in names:
            print(' -', n)
        for n in names:
            rec = evaluate_one_model(args.models_path, n, args)
            records.append(rec)

    if records:
        save_attack_summary(records, out_dir=args.out_dir)


if __name__ == '__main__':
    main()