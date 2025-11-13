import argparse
import csv
import glob
import os
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_records(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def find_latest_csv(out_dir: str) -> str:
    pattern = os.path.join(out_dir, 'attack_summary_*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'No CSV found by pattern {pattern}')
    files.sort()
    return files[-1]


def build_points(rows: List[dict], use_adv_acc: bool) -> Tuple[List[float], List[float], List[str]]:
    xs, ys, labels = [], [], []
    for r in rows:
        x = to_float(r.get('energy_ratio_adv_over_clean', 0.0))
        # y as accuracy (percentage)
        y = to_float(r.get('adv_final_acc' if use_adv_acc else 'clean_final_acc', 0.0))
        xs.append(x)
        ys.append(y)
        labels.append(r.get('model_name', 'unknown'))
    return xs, ys, labels


def compute_auc(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    # Sort by x
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    if len(pairs) < 2:
        return 0.0, 0.0
    auc = 0.0
    x_min, x_max = pairs[0][0], pairs[-1][0]
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        w = (x1 - x0)
        h = 0.5 * (y0 + y1)
        auc += w * h
    # Normalized AUC: divide by range * 100 (since y in %)
    x_range = max(1e-12, (x_max - x_min))
    norm_auc = auc / (x_range * 100.0)
    return auc, norm_auc


def plot_curve(xs: List[float], ys: List[float], labels: List[str], title: str, save_path: str):
    pairs = sorted(zip(xs, ys, labels), key=lambda t: t[0])
    xs_sorted = [p[0] for p in pairs]
    ys_sorted = [p[1] for p in pairs]
    labels_sorted = [p[2] for p in pairs]

    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.plot(xs_sorted, ys_sorted, marker='o', linewidth=2, label='models')
    for x, y, lb in zip(xs_sorted, ys_sorted, labels_sorted):
        plt.annotate(lb, (x, y), textcoords="offset points", xytext=(5,5), fontsize=8)
    plt.xlabel('Energy ratio (adv / clean)')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_points(xs: List[float], ys: List[float], labels: List[str], save_csv: str):
    with open(save_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['label', 'energy_ratio', 'accuracy_percent'])
        for lb, x, y in zip(labels, xs, ys):
            wr.writerow([lb, f'{x:.6f}', f'{y:.6f}'])


def main():
    parser = argparse.ArgumentParser(description='Plot Accuracy vs Energy Ratio from attack summary CSV')
    parser.add_argument('--out_dir', type=str, default='outputs/test_results', help='Directory containing attack_summary_*.csv')
    parser.add_argument('--csv', type=str, default=None, help='Path to a specific CSV; if omitted, latest is used')
    parser.add_argument('--use_adv_acc', action='store_true', default=True, help='Use adv_final_acc instead of clean_final_acc')
    parser.add_argument('--tag', type=str, default='', help='Optional tag appended to output filenames')
    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_latest_csv(args.out_dir)
    rows = load_records(csv_path)
    xs, ys, labels = build_points(rows, use_adv_acc=args.use_adv_acc)

    auc, norm_auc = compute_auc(xs, ys)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    tag = (('_' + args.tag) if args.tag else '')
    out_png = os.path.join(args.out_dir, f'{base}_acc_vs_energy{tag}.png')
    out_csv = os.path.join(args.out_dir, f'{base}_acc_vs_energy{tag}.points.csv')
    out_txt = os.path.join(args.out_dir, f'{base}_acc_vs_energy{tag}.metrics.txt')

    plot_curve(xs, ys, labels, title=f'Accuracy vs Energy Ratio ({"Adv" if args.use_adv_acc else "Clean"})', save_path=out_png)
    save_points(xs, ys, labels, save_csv=out_csv)

    with open(out_txt, 'w') as f:
        f.write(f'CSV: {csv_path}\n')
        f.write(f'Points: {len(xs)}\n')
        f.write(f'AUC (trapezoidal, unnormalized): {auc:.6f}\n')
        f.write(f'Normalized AUC (0..1): {norm_auc:.6f}\n')

    print('Saved:')
    print(' -', out_png)
    print(' -', out_csv)
    print(' -', out_txt)


if __name__ == '__main__':
    main()
