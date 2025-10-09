#!/usr/bin/env python3
import os
import re
import csv
import argparse
from datetime import datetime

import torch

import aux_funcs as af
import network_architectures as arcs
from data import accuracy, AverageMeter


def list_candidate_model_names(models_path: str):
    """
    在 models_path 下尽可能发现模型名：
    - 优先把子目录名当作模型名（通常每个模型一个子目录）
    - 同时尝试从 *_params.* 或 *_epoch_*.pt/pth 文件名中提取前缀
    """
    if not os.path.isdir(models_path):
        return []

    entries = os.listdir(models_path)
    names = set()

    # 1) 子目录视为模型名
    for d in entries:
        if os.path.isdir(os.path.join(models_path, d)):
            names.add(d)

    # 2) 文件名规则提取
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

    # 3) 验证可加载性（保留能 load 的）
    valid = []
    for n in sorted(names):
        try:
            _m, _p = arcs.load_model(models_path, n, -1)
            valid.append(n)
        except Exception:
            # 可能是未完成训练或占位目录，忽略
            pass

    return valid


@torch.no_grad()
def eval_model_by_forward(model, dataloader, device):
    """
    通用前向评测：对 CNN 或 SDN（多出口）都适用。
    返回：
      - exits_top1: 每个出口/最终输出的Top1
      - exits_top5: 每个出口/最终输出的Top5
    """
    model.eval()
    num_exits = None
    meters_top1 = []
    meters_top5 = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            out_list = list(outputs)
        else:
            out_list = [outputs]

        if num_exits is None:
            num_exits = len(out_list)
            meters_top1 = [AverageMeter() for _ in range(num_exits)]
            meters_top5 = [AverageMeter() for _ in range(num_exits)]

        for i, logits in enumerate(out_list):
            top1 = accuracy(logits, targets, topk=(1,))[0]  # tensor
            top5 = accuracy(logits, targets, topk=(5,))[0]
            meters_top1[i].update(top1.item(), n=images.size(0))
            meters_top5[i].update(top5.item(), n=images.size(0))

    exits_top1 = [m.avg for m in meters_top1]
    exits_top5 = [m.avg for m in meters_top5]
    return exits_top1, exits_top5


def test_single_model(models_path, model_name, epoch=-1, device=None):
    """
    加载并评测单个模型。优先用通用前向评测，保证可用性。
    失败会抛异常，由上层捕获。
    """
    if device is None:
        device = af.get_pytorch_device()

    print('=' * 80)
    print(f'Model: {model_name} (epoch={epoch})')

    model, params = arcs.load_model(models_path, model_name, epoch)
    dataset = af.get_dataset(params['task'])

    model.to(device)

    # 通用前向评测（不依赖 test_func）
    exits_top1, exits_top5 = eval_model_by_forward(model, dataset.test_loader, device)

    # 结果打印
    if len(exits_top1) == 1:
        print(f'  Top-1: {exits_top1[0]:.2f}% | Top-5: {exits_top5[0]:.2f}%')
    else:
        for i, (t1, t5) in enumerate(zip(exits_top1, exits_top5), 1):
            print(f'  Exit {i}: Top-1 {t1:.2f}% | Top-5 {t5:.2f}%')
        print(f'  Final: Top-1 {exits_top1[-1]:.2f}% | Top-5 {exits_top5[-1]:.2f}%')

    return {
        'model_name': model_name,
        'task': params.get('task', 'unknown'),
        'network_type': params.get('network_type', 'unknown'),
        'num_exits': len(exits_top1),
        'exits_top1': exits_top1,
        'exits_top5': exits_top5,
        'final_top1': exits_top1[-1],
        'final_top5': exits_top5[-1],
    }


def save_summary_csv(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'test_summary_{ts}.csv')

    # 汇总列：基础 + 展开每个出口
    max_exits = max((r['num_exits'] for r in results), default=1)
    fieldnames = ['model_name', 'task', 'network_type', 'num_exits', 'final_top1', 'final_top5']
    for i in range(1, max_exits + 1):
        fieldnames.append(f'exit{i}_top1')
        fieldnames.append(f'exit{i}_top5')

    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in results:
            row = {
                'model_name': r['model_name'],
                'task': r['task'],
                'network_type': r['network_type'],
                'num_exits': r['num_exits'],
                'final_top1': f"{r['final_top1']:.2f}",
                'final_top5': f"{r['final_top5']:.2f}",
            }
            for i in range(r['num_exits']):
                row[f'exit{i+1}_top1'] = f"{r['exits_top1'][i]:.2f}"
                row[f'exit{i+1}_top5'] = f"{r['exits_top5'][i]:.2f}"
            wr.writerow(row)

    print(f'\nSaved summary: {csv_path}')
    return csv_path


def main():
    parser = argparse.ArgumentParser(description='Test all trained CNN/SDN models in a seed directory.')
    parser.add_argument('--models-path', required=True, help='e.g., networks/1221')
    parser.add_argument('--device', default=None, help='cuda, cuda:0, or cpu. Default: auto')
    parser.add_argument('--epoch', type=int, default=-1, help='epoch to load, -1 = latest')
    parser.add_argument('--only', nargs='*', help='only test these model names (subdir names)')
    parser.add_argument('--skip', nargs='*', default=[], help='skip these model names')
    args = parser.parse_args()

    device = args.device or af.get_pytorch_device()
    print(f'Using PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
    print(f'Using device: {device}')
    print(f'Models path: {args.models_path}')

    if not os.path.isdir(args.models_path):
        print('Models path not found.')
        return

    # 发现可用模型名
    if args.only:
        candidates = args.only
    else:
        candidates = list_candidate_model_names(args.models_path)

    if not candidates:
        print('No loadable models found under this path.')
        print('Hint: check subdirectories or ensure training has saved weights.')
        # 打印目录帮助排查
        try:
            print('\nDirectory listing (first 50):')
            for f in sorted(os.listdir(args.models_path))[:50]:
                print('  -', f)
        except Exception:
            pass
        return

    # 过滤 skip
    candidates = [n for n in candidates if n not in set(args.skip)]

    print(f'\nDiscovered {len(candidates)} model(s):')
    for n in candidates:
        print('  -', n)

    # 逐个评测
    results = []
    failures = []
    for name in candidates:
        try:
            res = test_single_model(args.models_path, name, epoch=args.epoch, device=device)
            results.append(res)
        except Exception as e:
            print(f'Failed: {name} -> {e}')
            failures.append((name, str(e)))

    if results:
        save_summary_csv(results, out_dir='outputs/test_results')

    if failures:
        print('\nSome models failed to test:')
        for n, msg in failures:
            print(f'  - {n}: {msg}')


if __name__ == '__main__':
    print("python3 test_networks.py --models-path networks/你的seed")
    main()