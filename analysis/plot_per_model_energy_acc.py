import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_rows(csv_path: str) -> List[dict]:
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def find_latest_csv(out_dir: str) -> str:
    pattern = os.path.join(out_dir, 'attack_summary_*.csv')
    files = [f for f in glob.glob(pattern)
             if ('acc_vs_energy' not in f and not f.endswith('.points.csv') and 'per_model' not in f)]
    if not files:
        raise FileNotFoundError(f'No base attack_summary CSV found by pattern {pattern}')
    # sort by modification time to get the newest genuine summary
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def group_by_model(rows: List[dict]) -> Dict[str, List[dict]]:
    g: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        name = r.get('model_name', 'unknown')
        g[name].append(r)
    return g


def compute_auc(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    if len(pairs) < 2:
        return 0.0, 0.0
    auc = 0.0
    x_min, x_max = pairs[0][0], pairs[-1][0]
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        w = (x1 - x0)
        h = 0.5 * (y0 + y1)
        auc += w * h
    x_range = max(1e-12, (x_max - x_min))
    norm_auc = auc / (x_range * 100.0)
    return auc, norm_auc


def plot_one_model(model: str, rows: List[dict], out_dir: str, base_tag: str):
    # Collect points: anchor clean (x=1, y=clean_acc) + adv points per row
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []

    # Use first row's clean acc as anchor; if multiple rows exist, they should have same clean acc
    clean_acc = to_float(rows[0].get('clean_final_acc', 0.0))
    xs.append(1.0)
    ys.append(clean_acc)
    labels.append('clean (x=1.0)')

    # Add adv points
    for idx, r in enumerate(rows):
        xr = to_float(r.get('energy_ratio_adv_over_clean', 0.0))
        ya = to_float(r.get('adv_final_acc', 0.0))
        xs.append(xr)
        ys.append(ya)
        labels.append(f'adv#{idx+1} ({xr:.3f})')

    # Sort for plotting and AUC
    pairs = sorted(zip(xs, ys, labels), key=lambda t: t[0])
    xs_s = [p[0] for p in pairs]
    ys_s = [p[1] for p in pairs]
    labs = [p[2] for p in pairs]

    auc, n_auc = compute_auc(xs_s, ys_s)

    # Plot
    plt.figure(figsize=(6.8, 4.6), dpi=140)
    plt.plot(xs_s, ys_s, marker='o', linewidth=2, color='#1f77b4')
    for x, y, lb in zip(xs_s, ys_s, labs):
        plt.scatter([x], [y], color='#1f77b4')
        plt.annotate(lb, (x, y), textcoords='offset points', xytext=(5,5), fontsize=8)
    plt.xlabel('Energy ratio (adv / clean)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model} — Acc vs Energy ratio')
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()

    # Ensure per-model folder
    model_sanitized = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in model)
    subdir = os.path.join(out_dir, 'per_model')
    os.makedirs(subdir, exist_ok=True)

    png_path = os.path.join(subdir, f'{base_tag}_{model_sanitized}.png')
    txt_path = os.path.join(subdir, f'{base_tag}_{model_sanitized}.metrics.txt')
    pts_path = os.path.join(subdir, f'{base_tag}_{model_sanitized}.points.csv')

    plt.savefig(png_path)
    plt.close()

    # Save metrics and points
    with open(txt_path, 'w') as f:
        f.write(f'model: {model}\n')
        f.write(f'points: {len(xs_s)}\n')
        f.write(f'auc: {auc:.6f}\n')
        f.write(f'normalized_auc: {n_auc:.6f}\n')
        f.write(f'x_min: {min(xs_s):.6f}, x_max: {max(xs_s):.6f}\n')

    with open(pts_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['label','x_energy_ratio','y_accuracy_percent'])
        for lb, x, y in zip(labs, xs_s, ys_s):
            wr.writerow([lb, f'{x:.6f}', f'{y:.6f}'])

    return png_path, txt_path, pts_path


def main():
    parser = argparse.ArgumentParser(description='Per-model Acc vs Energy ratio plots')
    parser.add_argument('--out_dir', type=str, default='outputs/test_results')
    parser.add_argument('--csv', type=str, default=None)
    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_latest_csv(args.out_dir)
    base_tag = os.path.splitext(os.path.basename(csv_path))[0]

    rows = load_rows(csv_path)
    groups = group_by_model(rows)

    print(f'Found {len(groups)} model group(s) in {csv_path}')

    for model, rs in groups.items():
        # sort rows by energy ratio just for stable labeling/order
        rs_sorted = sorted(rs, key=lambda r: to_float(r.get('energy_ratio_adv_over_clean', 0.0)))
        png, txt, pts = plot_one_model(model, rs_sorted, args.out_dir, base_tag)
        print('Saved:')
        print(' -', png)
        print(' -', txt)
        print(' -', pts)


if __name__ == '__main__':
    main()
