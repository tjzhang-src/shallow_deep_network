#!/usr/bin/env python3
"""
Compute per-exit accuracies for SDN/CNN models.

Usage example:
  python tools/compute_exit_accuracy.py \
    --models_path networks/1221 \
    --models cifar10_resnet56_sdn \
    --dataset cifar10 \
    --batch_size 256 \
    --device cuda:0 \
    --out_csv outputs/exit_accuracy.csv

This script loads each model via `network_architectures.load_model`, runs the test set,
computes accuracy for each exit (internal ICs + final), and writes a CSV with per-exit
correct/total/accuracy and an optional layer mapping when available.
"""

import os
import sys
import csv
import argparse
from typing import List

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import tqdm

import network_architectures as arcs
import aux_funcs as af


def _build_exit_to_blockcount(model) -> dict:
    """Map exit_id (1..N) to a coarse 'computational blocks' count based on SDN layers.

    For non-SDN models this will map final exit to 1.
    """
    mapping = {}
    if not hasattr(model, 'layers'):
        # Single-head CNN: single exit mapped to layer 1
        mapping[1] = 1
        return mapping
    exit_id = 1
    for i, blk in enumerate(model.layers):
        has_exit = getattr(blk, 'output', None) is not None and not getattr(blk, 'no_output', True)
        if has_exit:
            mapping[exit_id] = i + 1
            exit_id += 1
    # final output id
    mapping[exit_id] = len(model.layers)
    return mapping


def load_testloader(task: str, batch_size: int):
    # Try aux_funcs helper first
    ds = af.get_dataset(task, batch_size=batch_size)
    if ds is not None and hasattr(ds, 'test_loader'):
        return ds.test_loader
    # Fallbacks similar to attack.py
    task = task.lower()
    if task == 'cifar10':
        ds = af.load_cifar10(batch_size)
        return ds.test_loader
    if task == 'cifar100':
        ds = af.load_cifar100(batch_size)
        return ds.test_loader
    if task == 'tinyimagenet':
        ds = af.load_tinyimagenet(batch_size)
        return ds.test_loader
    raise ValueError(f"Unsupported dataset task: {task}")


def evaluate_model_exits(models_path: str, model_name: str, args) -> List[dict]:
    """Evaluate per-exit accuracy for a single model. Returns list of records for CSV."""
    model, params = arcs.load_model(models_path, model_name, epoch=-1)
    device = torch.device(args.device if args.device else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    model.eval()

    task = params.get('task', args.dataset)
    testloader = load_testloader(task, args.batch_size)

    # Prepare counters
    # We'll count exits as 1..N where N is number of outputs (final included)
    # For single-head CNN (outputs is tensor), treat as N=1
    total = 0
    # We'll inspect one batch to get number of exits
    sample_iter = iter(testloader)
    try:
        sample_imgs, sample_labels = next(sample_iter)
    except StopIteration:
        return []
    sample_imgs = sample_imgs.to(device)
    with torch.no_grad():
        outs = model.forward(sample_imgs)
        if isinstance(outs, list):
            num_exits = len(outs)
        else:
            num_exits = 1
    # initialize counters
    correct_counts = [0] * num_exits
    total_counts = [0] * num_exits

    # mapping exit -> block/layer index if available
    exit_to_blocks = _build_exit_to_blockcount(model)

    pbar = tqdm.tqdm(testloader, desc=f"Eval {model_name}")
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.size(0)
        total += batch_size
        with torch.no_grad():
            outs = model.forward(imgs)
            if isinstance(outs, list):
                # For each exit, compute preds from that exit's logits
                for eid, logits in enumerate(outs, start=1):
                    preds = logits.argmax(dim=1)
                    match = (preds == labels).cpu().sum().item()
                    correct_counts[eid-1] += int(match)
                    total_counts[eid-1] += batch_size
            else:
                preds = outs.argmax(dim=1)
                match = (preds == labels).cpu().sum().item()
                correct_counts[0] += int(match)
                total_counts[0] += batch_size
        pbar.set_postfix(total=total)

    # Build records
    records = []
    for eid in range(1, num_exits+1):
        corr = correct_counts[eid-1]
        tot = total_counts[eid-1]
        acc = (float(corr) / float(tot) * 100.0) if tot > 0 else 0.0
        rec = {
            'model_name': model_name,
            'task': task,
            'exit_id': eid,
            'exit_layer_block': int(exit_to_blocks.get(eid, -1)),
            'correct': int(corr),
            'total': int(tot),
            'accuracy_pct': float(acc)
        }
        records.append(rec)

    # cleanup
    if str(device).startswith('cuda'):
        del model
        torch.cuda.empty_cache()
    return records


def discover_models(models_path: str) -> List[str]:
    """Discover all valid model names in models_path directory."""
    if not os.path.isdir(models_path):
        return []
    
    model_names = []
    for entry in os.listdir(models_path):
        entry_path = os.path.join(models_path, entry)
        if os.path.isdir(entry_path):
            # Check if it has parameters_last or last file (trained model)
            if os.path.exists(os.path.join(entry_path, 'parameters_last')) or \
               os.path.exists(os.path.join(entry_path, 'last')):
                model_names.append(entry)
    
    return sorted(model_names)


def main():
    parser = argparse.ArgumentParser(description='Compute per-exit accuracy for models')
    parser.add_argument('--models_path', type=str, required=True, help='Root directory containing model folders/files')
    parser.add_argument('--models', type=str, nargs='*', default=None, 
                        help='One or more model names to evaluate. If omitted, discovers all models in models_path.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100','tinyimagenet'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default=None, help='e.g. cuda:0')
    parser.add_argument('--out_csv', type=str, default='outputs/exit_accuracy.csv')
    parser.add_argument('--filter', type=str, default=None, 
                        help='Only evaluate models containing this substring (e.g., "sdn_training" or "cifar100")')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)

    # Discover or use specified models
    if args.models is None or len(args.models) == 0:
        print(f"Auto-discovering models in {args.models_path} ...")
        models_to_eval = discover_models(args.models_path)
        if args.filter:
            models_to_eval = [m for m in models_to_eval if args.filter in m]
            print(f"Filtered to {len(models_to_eval)} models containing '{args.filter}'")
        print(f"Found {len(models_to_eval)} models")
    else:
        models_to_eval = args.models
    
    if not models_to_eval:
        print("No models to evaluate!")
        return

    all_records = []
    for m in models_to_eval:
        print(f"Evaluating {m} ...")
        try:
            recs = evaluate_model_exits(args.models_path, m, args)
            all_records.extend(recs)
        except Exception as e:
            print(f"[WARN] Failed to evaluate {m}: {e}")

    # write CSV
    fields = ['model_name','task','exit_id','exit_layer_block','correct','total','accuracy_pct']
    with open(args.out_csv, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in all_records:
            wr.writerow(r)

    print(f"Wrote results to {args.out_csv}")

if __name__ == '__main__':
    main()
