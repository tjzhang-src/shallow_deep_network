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
import torch.optim as optim
from torch.optim.sgd import SGD
# torchvision 的直接加载已被项目自带的标准化数据加载替代
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import network_architectures as arcs
import aux_funcs as af
import profiler as prof

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _compute_eec_score(exit_dist: List[int]) -> float:
    """Compute EECScore: normalized area under cumulative early-exit curve.

    Given per-exit counts or proportions for exits 1..N (including final as N),
    build cumulative fractions c_k and return mean(c_k), in [0,1]. Higher means earlier exits on average.
    """
    s = float(sum(exit_dist))
    if s <= 0:
        return 0.0
    probs = [float(x) / s for x in exit_dist]
    cum = 0.0
    acc = 0.0
    for p in probs:
        cum += p
        acc += cum
    return acc / len(exit_dist)


def _compute_curve_distances(c1: List[float], c2: List[float]) -> Tuple[float, float]:
    """Compute simple distances between two aligned cumulative curves c1, c2.

    - Hausdorff (aligned grid): max_k |c1[k]-c2[k]|
    - SSPD (aligned grid surrogate): mean_k |c1[k]-c2[k]|
    """
    if not c1 or not c2 or len(c1) != len(c2):
        return 0.0, 0.0
    diffs = [abs(float(a) - float(b)) for a, b in zip(c1, c2)]
    haus = max(diffs) if diffs else 0.0
    sspd = sum(diffs) / len(diffs) if diffs else 0.0
    return haus, sspd


def _build_exit_to_blockcount(model) -> dict:
    """Map exit_id (1..N) to a coarse 'computational blocks' count based on SDN layers.

    We define blocks as SDN 'layers' modules traversed. For an internal exit attached to
    model.layers[i], blocks = i+1. For the final classifier, blocks = len(model.layers).
    """
    mapping = {}
    exit_id = 1
    for i, blk in enumerate(model.layers):
        has_exit = getattr(blk, 'output', None) is not None and not getattr(blk, 'no_output', True)
        if has_exit:
            mapping[exit_id] = i + 1
            exit_id += 1
    # final output id
    mapping[exit_id] = len(model.layers)
    return mapping


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


def get_norm_stats_for_task(task: str) -> Tuple[List[float], List[float]]:
    """Return per-channel mean and std used during training for a given task.
    These must match the Normalize() used in data.py.
    """
    t = task.lower()
    if t == 'cifar10':
        # Matches data.CIFAR10 normalization
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if t == 'cifar100':
        return [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    if t == 'tinyimagenet':
        return [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
    # Fallback to identity (no normalization)
    return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]


def select_exits(outputs: List[torch.Tensor], thresholds: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """依据 entropy 阈值列表决定早退出口。
    outputs: [ic_0, ..., ic_{N-2}, final]
    thresholds: len == N-1 (内部 IC 个数)
    返回: (pred_labels, exit_ids)  exit_id 从 1 开始，final 为 N
    """
    num_internal = len(outputs) - 1
    final_idx = len(outputs)  # 1-based id
    batch = outputs[0].size(0)
    # 不进行强制选择，由阈值决定
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


def attack_batch(model, x, y, thresholds, eps, alpha, steps, lambda_exits, lambda_earlyexits, lambda_ce, lambda_l2,
                 cw=False, c=0.15, kappa=5, pcgrad=False, same_acc_early_loss=False,
                 prefer_exit_id: Optional[int] = None,
                 norm_mean: Optional[List[float]] = None,
                 norm_std: Optional[List[float]] = None,
                 gradnorm: bool = False,
                 gradnorm_alpha: float = 1.0,
                 gradnorm_lr: float = 1e-3,
                 pcgrad_mode: str = 'symmetric',
                 pcgrad_partial: float = 1.0,
                 pre_target_margin: float = 0.0,
                 pre_target_weight: float = 1.0,
                 pgd_update_mode: str = 'linf_sign',
                 pgd_momentum: float = 0.0,
                 auto_balance_early: bool = False,
                 ab_target_ratio: float = 1.0,
                 grad_select: str = 'none',
                 snapshot_every: int = 0,
                 snapshot_steps: Optional[List[int]] = None):
    """对 batch 做 constrained PGD，返回 (x_adv, linf, l2, preds_adv, exits_adv, pcgrad_conflict_count)

    仅对原本被正确分类的样本调用。兼容 SDN（多出口）与 CNN（单出口）。
    """
    x_orig = x.detach()
    batch_size = x.size(0)
    delta = torch.zeros_like(x, device=x.device, requires_grad=True)
    num_internal = model.num_output - 1 if hasattr(model, 'num_output') else 0

    # Prepare normalization-aware bounds and step sizes (operate in normalized space)
    if norm_mean is None or norm_std is None:
        # Assume inputs are already in [0,1] without normalization
        lower = torch.zeros((1, x.size(1), 1, 1), device=x.device)
        upper = torch.ones((1, x.size(1), 1, 1), device=x.device)
        # eps/alpha already in pixel unit; when no normalization, keep as scalars
        eps_tensor = torch.full_like(x, fill_value=eps)
        alpha_tensor = torch.full_like(x, fill_value=alpha)
    else:
        mean = torch.tensor(norm_mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(norm_std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        # Valid range in normalized space for pixel range [0,1]
        lower = (0.0 - mean) / std
        upper = (1.0 - mean) / std
        # Map pixel-space eps/alpha to normalized space per channel
        eps_tensor = torch.ones_like(x) * (eps / std)
        alpha_tensor = torch.ones_like(x) * (alpha / std)

    # 若 lambda_exits 长度不足，补齐；超出截断
    if len(lambda_exits) < num_internal:
        lambda_exits = lambda_exits + [lambda_exits[-1]] * (num_internal - len(lambda_exits))
    elif len(lambda_exits) > num_internal:
        lambda_exits = lambda_exits[:num_internal]

    conflict_steps = 0

    # GradNorm state (for two tasks: accuracy and early-exit). Initialize on demand
    use_two_tasks = False
    if (lambda_earlyexits > 0.0) and (len(lambda_exits) > 0) and (len(lambda_exits) == (model.num_output - 1 if hasattr(model, 'num_output') else 0)):
        use_two_tasks = True
    if gradnorm and not use_two_tasks:
        # No early-exit loss -> GradNorm degenerates to single task; disable
        gradnorm = False
    # holders
    w = None
    opt_w = None
    initial_losses = None
    # momentum buffer for PGD (if enabled)
    v = torch.zeros_like(delta)
    # configure snapshot set (1-based step indices)
    snapshot_set: set = set()
    if snapshot_steps is not None and len(snapshot_steps) > 0:
        snapshot_set = set(int(s) for s in snapshot_steps if int(s) >= 1)
    elif snapshot_every and int(snapshot_every) > 0:
        snapshot_set = set(i for i in range(int(snapshot_every), steps + 1, int(snapshot_every)))
    snapshots_summary: dict = {}
    for step_idx in range(1, steps + 1):
        # Compose adversarial example in normalized space and clamp to valid normalized bounds
        x_adv = (x_orig + delta).clamp(lower, upper)
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
            # 若指定偏好出口，则将早期损失设置为：
            # - 对于 i < target: 提高熵，避免更早出口 -> hinge(t - nent, 0)
            # - 对于 i == target: 降低熵，触发该出口 -> hinge(nent - t, 0)
            # - 对于 i > target: 不约束（已不会到达）
            target = prefer_exit_id if (prefer_exit_id is not None and prefer_exit_id >= 1) else None
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
                        # 在目标之前的层，稍微“抬高”阈值为 (t + pre_target_margin)，并可调节权重
                        early_losses.append(pre_target_weight * torch.clamp((t + float(pre_target_margin)) - nent, min=0.0).mean())
                    elif exit_idx == target:
                        early_losses.append(torch.clamp(nent - t, min=0.0).mean())
                    else:
                        # exit_idx > target: 无约束
                        early_losses.append(torch.zeros(1, device=final_out.device, dtype=final_out.dtype))
        if early_losses:
            # Avoid colliding with GradNorm weight variable name
            loss_early_total = sum(w_exit * l for w_exit, l in zip(lambda_exits, early_losses))
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
            g_early = torch.autograd.grad(loss_early_total * lambda_earlyexits, delta, retain_graph=True)[0]
            g_reg = torch.autograd.grad(lambda_l2 * loss_l2_term, delta)[0]
            dot = torch.dot(g_acc.flatten(), g_early.flatten())
            if dot < 0:
                conflict_steps += 1
                # Partial or full projection controlled by pcgrad_partial
                denom_e = (g_early.flatten().norm() ** 2 + 1e-12)
                denom_a = (g_acc.flatten().norm() ** 2 + 1e-12)
                proj_acc = g_acc - pcgrad_partial * (dot / denom_e) * g_early
                if pcgrad_mode == 'symmetric':
                    proj_early = g_early - pcgrad_partial * (dot / denom_a) * g_acc
                else:
                    proj_early = g_early
                g_acc, g_early = proj_acc, proj_early
            grads = g_acc + g_early + g_reg
        elif gradnorm and early_losses:
            # lazy init for GradNorm weights/optimizer
            if w is None:
                w = torch.nn.Parameter(torch.ones(2, device=x.device))
                opt_w = SGD([w], lr=float(gradnorm_lr))
            # Two-task GradNorm between accuracy and early-exit losses
            loss_acc_scaled = lambda_ce * loss_ce_term
            loss_early_scaled = lambda_earlyexits * loss_early_total

            # Initialize initial losses at first step (detach to avoid graph retention)
            if initial_losses is None:
                with torch.no_grad():
                    l0_acc = float(loss_acc_scaled.item()) if loss_acc_scaled.requires_grad else float(loss_acc_scaled)
                    l0_early = float(loss_early_scaled.item()) if loss_early_scaled.requires_grad else float(loss_early_scaled)
                    # avoid zeros
                    if l0_acc <= 0.0:
                        l0_acc = 1e-4
                    if l0_early <= 0.0:
                        l0_early = 1e-4
                    initial_losses = torch.tensor([l0_acc, l0_early], device=x.device)

            # 1) Weighted total loss for delta update (do not update w here)
            L_total = w[0] * loss_acc_scaled + w[1] * loss_early_scaled + lambda_l2 * loss_l2_term
            grads = torch.autograd.grad(L_total, delta, retain_graph=True)[0]

            # 2) Grad norms per task on shared param (delta)
            G = torch.zeros(2, device=x.device)
            g_acc = torch.autograd.grad(w[0] * loss_acc_scaled, delta, retain_graph=True, create_graph=True)[0]
            g_early = torch.autograd.grad(w[1] * loss_early_scaled, delta, retain_graph=True, create_graph=True)[0]
            G[0] = g_acc.view(-1).norm(p=2)
            G[1] = g_early.view(-1).norm(p=2)
            G_bar = G.mean()

            # 3) Relative inverse training rates r_i
            with torch.no_grad():
                L_t = torch.tensor([
                    float(loss_acc_scaled.item()),
                    float(loss_early_scaled.item())
                ], device=x.device)
                L0 = initial_losses
                Ltilde = L_t / L0
                r = Ltilde / Ltilde.mean()

            # 4) Target gradient norms
            G_star = G_bar * (r ** float(gradnorm_alpha))

            # 5) GradNorm loss and update w
            grad_loss = (G - G_star).abs().sum()
            assert opt_w is not None
            opt_w.zero_grad()
            grad_loss.backward()
            opt_w.step()

            # 6) Normalize w to keep sum(w)=T and positive
            with torch.no_grad():
                w.data = torch.clamp(w.data, min=1e-4)
                w.data = (2.0 * w.data) / w.data.sum()
        else:
            # Optionally auto-balance early loss magnitude by gradient norm ratio to CE
            scale_early = 1.0
            if auto_balance_early and early_losses:

                g_acc_tmp = torch.autograd.grad(lambda_ce * loss_ce_term, delta, retain_graph=True)[0]
                g_early_tmp = torch.autograd.grad(lambda_earlyexits * loss_early_total, delta, retain_graph=True)[0]
                norm_acc = g_acc_tmp.view(g_acc_tmp.size(0), -1).norm(p=2, dim=1).mean()
                norm_early = g_early_tmp.view(g_early_tmp.size(0), -1).norm(p=2, dim=1).mean()
                if float(norm_early) > 0.0:
                    scale_early = float(ab_target_ratio) * float(norm_acc) / float(norm_early)
                    # keep within a sane range
                    scale_early = float(max(0.1, min(10.0, scale_early)))
            # Compute individual grads (scaled) to optionally select/merge
            loss_acc_scaled2 = (lambda_ce * loss_ce_term)
            loss_early_scaled2 = (scale_early * (lambda_earlyexits * loss_early_total))
            g_acc2 = torch.autograd.grad(loss_acc_scaled2, delta, retain_graph=True)[0]
            if early_losses:
                g_early2 = torch.autograd.grad(loss_early_scaled2, delta, retain_graph=True)[0]
            else:
                g_early2 = torch.zeros_like(delta)
            g_reg2 = torch.autograd.grad(lambda_l2 * loss_l2_term, delta)[0]

            if isinstance(grad_select, str):
                mode = grad_select.lower()
            else:
                mode = 'none'

            if mode == 'max_norm' and early_losses:
                # Choose the task gradient with larger mean L2 norm (batch-averaged)
                n_acc = g_acc2.view(g_acc2.size(0), -1).norm(p=2, dim=1).mean()
                n_early = g_early2.view(g_early2.size(0), -1).norm(p=2, dim=1).mean()
                grads = (g_early2 if float(n_early) >= float(n_acc) else g_acc2) + g_reg2
            elif mode == 'max_element' and early_losses:
                # Element-wise select the larger magnitude component from the two task grads
                grads = torch.where(g_acc2.abs() >= g_early2.abs(), g_acc2, g_early2) + g_reg2
            else:
                # Default: sum the two objectives (with auto-balancing if enabled)
                grads = g_acc2 + g_early2 + g_reg2

        # PGD 更新（最小化 loss），在归一化空间按通道尺度更新与投影
        # Apply momentum if enabled
        if pgd_momentum and pgd_momentum > 0.0:
            v = pgd_momentum * v + grads
            grad_dir = v
        else:
            grad_dir = grads

        if pgd_update_mode == 'linf_sign':
            step_dir = grad_dir.sign()
        else:
            # l2-like direction
            flat = grad_dir.view(grad_dir.size(0), -1)
            norms = flat.norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12
            step_dir = grad_dir / norms

        delta.data = delta.data - alpha_tensor * step_dir
        delta.data = torch.max(torch.min(delta.data, eps_tensor), -eps_tensor)
        delta.data = (x_orig + delta.data).clamp(lower, upper) - x_orig
        delta.grad = None
        # snapshot after update
        if step_idx in snapshot_set:
            with torch.no_grad():
                x_snap = (x_orig + delta).clamp(lower, upper)
                snap_outs = model.forward(x_snap)
                if isinstance(snap_outs, list):
                    preds_s, exits_s = select_exits(snap_outs, thresholds)
                    final_out_s = snap_outs[-1]
                else:
                    preds_s = snap_outs.argmax(dim=1).cpu()
                    exits_s = torch.full((batch_size,), 1, dtype=torch.long)
                    final_out_s = snap_outs
                final_pred_s = final_out_s.argmax(dim=1)
                correct_s = int((final_pred_s == y).sum().item())
                # build exit counts vector length = num_internal+1
                num_internal_local = model.num_output - 1 if hasattr(model, 'num_output') else 0
                counts = [0] * (num_internal_local + 1)
                for e in exits_s:
                    eid = int(e.item()) if num_internal_local > 0 else 1
                    # map final to last index (1..N)
                    if eid < 1:
                        eid = 1
                    if eid > (num_internal_local + 1):
                        eid = num_internal_local + 1
                    counts[eid - 1] += 1
                snapshots_summary[int(step_idx)] = {
                    'exit_counts': counts,
                    'correct': correct_s,
                    'num': batch_size,
                }

    x_adv = (x_orig + delta).clamp(lower, upper)
    with torch.no_grad():
        adv_outputs = model.forward(x_adv)
        if isinstance(adv_outputs, list):
            preds_adv, exits_adv = select_exits(adv_outputs, thresholds)
        else:
            preds_adv = adv_outputs.argmax(dim=1).cpu()
            exits_adv = torch.full((batch_size,), 1, dtype=torch.long)
        # Report norms in pixel space for interpretability
        if norm_std is not None:
            std = torch.tensor(norm_std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            delta_pix = delta * std
        else:
            delta_pix = delta
        linf = delta_pix.abs().view(batch_size, -1).max(dim=1)[0].cpu()
        l2 = delta_pix.view(batch_size, -1).norm(p=2, dim=1).cpu()
    # Return conflict ratio (fraction of steps with conflicting gradients)
    return x_adv.detach(), linf, l2, preds_adv, exits_adv, (conflict_steps / max(1, steps)), snapshots_summary

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
    # Normalization stats for correct clamping in normalized space
    mean, std = get_norm_stats_for_task(task)

    # 主循环
    total = 0
    correct_clean = 0
    correct_adv = 0
    exit_counts_clean = [0] * (num_internal + 2)  # 使用 1..num_internal+1
    exit_counts_adv = [0] * (num_internal + 2)
    avg_linf: List[float] = []
    avg_l2: List[float] = []
    # Accumulate PCGrad conflict fraction weighted by number of attacked samples
    pc_conflict_fraction = 0.0
    pc_conflict_weight = 0
    # FRP (flops recovery) lists: per-candidate sample ratio
    frr_flops_vals = []
    # Energy (pJ) trackers
    energy_clean_pj: List[float] = []
    energy_adv_pj: List[float] = []
    # Maximum confidence trackers (for chosen exits)
    maxconf_clean: List[float] = []
    maxconf_adv: List[float] = []

    pbar = tqdm.tqdm(testloader, desc=f"{model_name}")
    # Pre-compute per-exit FLOPs (cumulative) using profiler if available
    try:
        input_size = 64 if task == 'tinyimagenet' else 32
        out_ops, out_params = prof.profile_sdn(model, input_size, device)
        # out_ops are in GFLOPs per output id (0-based). Build mapping exit_id(1-based)->flops
        ops_per_exit = {}
        max_key = max(out_ops.keys()) if out_ops else 0
        for exit_id in range(1, (num_internal + 2)):
            gflops = out_ops.get(exit_id - 1, out_ops.get(max_key, 0.0))
            ops_per_exit[exit_id] = float(gflops) * 1e9
    except Exception:
        ops_per_exit = None

    # Snapshot setup
    snapshot_steps_set: set = set()
    try:
        if hasattr(args, 'snapshot_steps') and isinstance(args.snapshot_steps, str) and args.snapshot_steps.strip() != '':
            snapshot_steps_set = set(int(s) for s in args.snapshot_steps.split(',') if s.strip() != '')
        elif hasattr(args, 'snapshot_every') and int(getattr(args, 'snapshot_every', 0)) > 0:
            ev = int(args.snapshot_every)
            snapshot_steps_set = set(range(ev, int(args.pgd_steps) + 1, ev))
    except Exception:
        snapshot_steps_set = set()
    # Accumulators for dataset-level snapshot metrics (attacked subset only)
    snapshot_exit_counts = {int(k): [0] * (num_internal + 1) for k in sorted(list(snapshot_steps_set))}
    snapshot_correct = {int(k): 0 for k in snapshot_exit_counts.keys()}
    snapshot_total = {int(k): 0 for k in snapshot_exit_counts.keys()}
    energy_clean_attacked_pj: List[float] = []
    # Per-batch snapshot pairs (for plotting more granular curve): step -> list of dicts
    step_pairs = {int(k): [] for k in sorted(list(snapshot_steps_set))}
    combined_pairs = []  # across all steps and batches: list of (ratio, acc)

    # Pre-compute exit-id -> block-count mapping
    exit_to_blocks = _build_exit_to_blockcount(model)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.size(0)
        total += batch_size

        with torch.no_grad():
            outputs_clean = model.forward(imgs)
            if num_internal > 0:
                preds_clean, exits_clean = select_exits(outputs_clean, entropy_thresholds)
            else:
                preds_clean = outputs_clean[-1].argmax(dim=1).cpu() if isinstance(outputs_clean, list) else outputs_clean.argmax(dim=1).cpu()
                exits_clean = torch.full((batch_size,), 1, dtype=torch.long)
            final_out_clean = outputs_clean[-1] if isinstance(outputs_clean, list) else outputs_clean
            final_pred_clean = final_out_clean.argmax(dim=1)

        # Max confidence distribution (clean) based on chosen exit logits
        try:
            if isinstance(outputs_clean, list):
                for i in range(batch_size):
                    eid = int(exits_clean[i].item())
                    logits = outputs_clean[eid-1] if eid <= num_internal else outputs_clean[-1]
                    probs_i = F.softmax(logits[i], dim=0)
                    maxconf_clean.append(float(probs_i.max().item()))
            else:
                # single-head CNN fallback
                probs = F.softmax(outputs_clean, dim=1)
                maxconf_clean.extend(probs.max(dim=1)[0].detach().cpu().tolist())
        except Exception:
            pass

        # --- Clean energy (if profiler available) ---
        if ops_per_exit is not None:
            # energy per flop = 3.2 pJ
            for e in exits_clean:
                eid = int(e.item())
                fl = ops_per_exit.get(eid, None)
                if fl is None:
                    continue
                energy_clean_pj.append(float(fl) * 3.2)

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
            # If no attacked samples, adversarial exits equal clean; mirror energy as well
            if ops_per_exit is not None:
                # reuse the same per-sample energies computed above for this batch
                # Append the last batch_size clean energies to adv energies
                # safer: recompute for clarity
                for e in exits_clean:
                    eid = int(e.item())
                    fl = ops_per_exit.get(eid, None)
                    if fl is None:
                        continue
                    energy_adv_pj.append(float(fl) * 3.2)
            pbar.set_postfix(clean_acc=f"{correct_clean/max(1,total):.4f}", adv_acc=f"{correct_adv/max(1,total):.4f}")
            continue

        mask_idx = torch.nonzero(correct_mask, as_tuple=False).squeeze(1).to(device)
        imgs_correct = imgs[mask_idx]
        labels_correct = labels[mask_idx]

        # Gather clean energy for attacked subset (for global ratio denominator)
        if ops_per_exit is not None:
            for idx in mask_idx:
                eid_clean = int(exits_clean[int(idx.item())].item()) if num_internal > 0 else 1
                fl = ops_per_exit.get(eid_clean, None)
                if fl is not None:
                    energy_clean_attacked_pj.append(float(fl) * 3.2)
        # Also gather per-batch denominator (mean clean energy for attacked subset in this batch)
        batch_clean_den_pj = 0.0
        batch_attacked_num = int(mask_idx.numel())
        if ops_per_exit is not None and batch_attacked_num > 0:
            e_sum_b = 0.0
            for idx in mask_idx:
                eid_clean = int(exits_clean[int(idx.item())].item()) if num_internal > 0 else 1
                fl = ops_per_exit.get(eid_clean, None)
                if fl is None:
                    continue
                e_sum_b += float(fl) * 3.2
            if batch_attacked_num > 0:
                batch_clean_den_pj = e_sum_b / float(batch_attacked_num)

        x_adv_subset, linf_vals, l2_vals, preds_adv_subset, exits_adv_subset, conflict_ratio, snapshots_summary = attack_batch(
            model, imgs_correct, labels_correct,
            entropy_thresholds, eps, alpha, args.pgd_steps,
            lambda_exits,args.lambda_earlyexits, args.lambda_ce, args.lambda_l2,
            cw=args.cw, c=args.c, kappa=args.kappa, pcgrad=args.pcgrad,
            same_acc_early_loss=args.same_acc_early_loss_value,
            prefer_exit_id=args.prefer_exit,
            norm_mean=mean, norm_std=std,
            gradnorm=args.gradnorm, gradnorm_alpha=args.gradnorm_alpha, gradnorm_lr=args.gradnorm_lr,
            pcgrad_mode=args.pcgrad_mode, pcgrad_partial=args.pcgrad_partial,
            pre_target_margin=args.pre_target_margin, pre_target_weight=args.pre_target_weight,
            pgd_update_mode=args.pgd_update_mode, pgd_momentum=args.pgd_momentum,
            auto_balance_early=args.auto_balance_early, ab_target_ratio=args.ab_target_ratio, grad_select=args.grad_select,
            snapshot_every=int(getattr(args, 'snapshot_every', 0)) if hasattr(args, 'snapshot_every') else 0,
            snapshot_steps=[int(s) for s in sorted(list(snapshot_steps_set))] if snapshot_steps_set else None
        )
        # Weight by number of attacked samples in this batch
        attacked_count = imgs_correct.size(0)
        pc_conflict_fraction += float(conflict_ratio) * attacked_count
        pc_conflict_weight += attacked_count

        imgs_adv = imgs.clone()
        imgs_adv[mask_idx] = x_adv_subset
        with torch.no_grad():
            outputs_adv = model.forward(imgs_adv)
            if num_internal > 0:
                preds_adv_batch, exits_adv_batch = select_exits(outputs_adv, entropy_thresholds)
            else:
                preds_adv_batch = outputs_adv[-1].argmax(dim=1).cpu() if isinstance(outputs_adv, list) else outputs_adv.argmax(dim=1).cpu()
                exits_adv_batch = torch.full((batch_size,), 1, dtype=torch.long)
            final_out_adv = outputs_adv[-1] if isinstance(outputs_adv, list) else outputs_adv
            final_pred_adv = final_out_adv.argmax(dim=1)

        # Max confidence distribution (adv) based on chosen exit logits
        try:
            if isinstance(outputs_adv, list):
                for i in range(batch_size):
                    eid = int(exits_adv_batch[i].item())
                    logits = outputs_adv[eid-1] if eid <= num_internal else outputs_adv[-1]
                    probs_i = F.softmax(logits[i], dim=0)
                    maxconf_adv.append(float(probs_i.max().item()))
            else:
                probs = F.softmax(outputs_adv, dim=1)
                maxconf_adv.extend(probs.max(dim=1)[0].detach().cpu().tolist())
        except Exception:
            pass

        # --- Adv energy (if profiler available) ---
        if ops_per_exit is not None:
            for e in exits_adv_batch:
                eid = int(e.item())
                fl = ops_per_exit.get(eid, None)
                if fl is None:
                    continue
                energy_adv_pj.append(float(fl) * 3.2)

        # Accumulate snapshots for attacked subset
        if snapshots_summary:
            for k, sdict in snapshots_summary.items():
                if int(k) not in snapshot_exit_counts:
                    continue
                counts_vec = sdict.get('exit_counts', [])
                corr = int(sdict.get('correct', 0))
                num_b = int(sdict.get('num', 0))
                for i_c, cnt in enumerate(counts_vec):
                    snapshot_exit_counts[int(k)][i_c] += int(cnt)
                snapshot_correct[int(k)] += corr
                snapshot_total[int(k)] += num_b
                # Build per-batch pair for plotting: ratio vs acc
                if ops_per_exit is not None and batch_attacked_num > 0 and batch_clean_den_pj > 0:
                    energy_sum_step = 0.0
                    for i_c, cnt in enumerate(counts_vec, start=1):
                        fl = ops_per_exit.get(i_c, None)
                        if fl is None:
                            continue
                        energy_sum_step += float(cnt) * float(fl) * 3.2
                    mean_e_step = energy_sum_step / float(max(1, num_b))
                    ratio_b = mean_e_step / batch_clean_den_pj if batch_clean_den_pj > 0 else 0.0
                    acc_b = (float(corr) / float(max(1, num_b))) * 100.0
                    step_pairs[int(k)].append({'ratio': float(ratio_b), 'acc': float(acc_b), 'n': int(num_b)})
                    combined_pairs.append((float(ratio_b), float(acc_b)))

        # --- FLOPs-based recovery ratio per attacked sample (if profiler available) ---
        if ops_per_exit is not None and num_internal > 0:
            final_id = num_internal + 1
            # mask_idx (device) points to attacked positions; use CPU indices
            try:
                attacked_pos_cpu = mask_idx.cpu()
            except Exception:
                attacked_pos_cpu = None
            if attacked_pos_cpu is not None and attacked_pos_cpu.numel() > 0:
                clean_ex = exits_clean[attacked_pos_cpu]  # CPU tensor of clean exit ids
                adv_ex = exits_adv_batch[attacked_pos_cpu]  # CPU tensor of adv exit ids
                cnn_flops = ops_per_exit.get(final_id, None)
                if cnn_flops is not None:
                    for i_idx in range(attacked_pos_cpu.numel()):
                        c_ex = int(clean_ex[i_idx].item())
                        a_ex = int(adv_ex[i_idx].item())
                        fl_before = ops_per_exit.get(c_ex, None)
                        fl_after = ops_per_exit.get(a_ex, None)
                        if fl_before is None or fl_after is None:
                            continue
                        denom = (cnn_flops - fl_before)
                        if denom <= 0:
                            continue
                        ratio = (fl_after - fl_before) / denom
                        ratio = max(0.0, min(1.0, float(ratio)))
                        frr_flops_vals.append(ratio)

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
    # EECScore from exit distributions (higher => earlier exits)
    eec_clean = _compute_eec_score(dist_clean)
    eec_adv = _compute_eec_score(dist_adv)
    # Curve distances between cumulative early-exit curves
    total_clean = float(sum(dist_clean)) if sum(dist_clean) > 0 else 1.0
    total_adv = float(sum(dist_adv)) if sum(dist_adv) > 0 else 1.0
    cum_clean = []
    cum_adv = []
    cc = 0.0
    ca = 0.0
    for k in range(len(dist_clean)):
        cc += float(dist_clean[k]) / total_clean
        ca += float(dist_adv[k]) / total_adv
        cum_clean.append(cc)
        cum_adv.append(ca)
    haus_exitcdf, sspd_exitcdf = _compute_curve_distances(cum_clean, cum_adv)

    # Avg computational blocks (by SDN layers traversed)
    def _avg_blocks(dist: List[int]) -> float:
        tot = float(sum(dist))
        if tot <= 0:
            return 0.0
        val = 0.0
        for i, cnt in enumerate(dist, start=1):
            blocks = float(exit_to_blocks.get(i, len(model.layers)))
            val += blocks * float(cnt)
        return val / tot
    avg_blocks_clean = _avg_blocks(dist_clean)
    avg_blocks_adv = _avg_blocks(dist_adv)
    # Sanity checks for exit distribution counts
    clean_sum = int(sum(dist_clean))
    adv_sum = int(sum(dist_adv))
    try:
        dataset_len = len(testloader.dataset)  # type: ignore[attr-defined]
    except Exception:
        dataset_len = total
    if clean_sum != total or adv_sum != total or total != dataset_len:
        print(f"[WARN] Exit counts mismatch: clean_sum={clean_sum}, adv_sum={adv_sum}, total_iter={total}, dataset_len={dataset_len}")
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
    # Average fraction of conflicting steps across attacked samples
    'pcgrad_conflict_frac': pc_conflict_fraction / max(1, pc_conflict_weight),
        'preferred_exit': int(args.prefer_exit) if args.prefer_exit is not None else None,
        'gradnorm': bool(args.gradnorm),
        # EEC and curve distances
        'eec_clean': eec_clean,
        'eec_adv': eec_adv,
        # Directional summary: larger late-area => more late exits (attack goal)
        'late_exit_area_clean': (1.0 - eec_clean),
        'late_exit_area_adv': (1.0 - eec_adv),
        'late_exit_area_gain': (1.0 - eec_adv) - (1.0 - eec_clean),
        # Signed delta (adv - clean) on EEC where lower implies later exits
        'eec_delta': (eec_adv - eec_clean),
        'haus_exitcdf': haus_exitcdf,
        'sspd_exitcdf': sspd_exitcdf,
        # Blocks-based complexity
        'avg_blocks_clean': avg_blocks_clean,
        'avg_blocks_adv': avg_blocks_adv,
    }
    # Snapshot points: energy ratio (adv step / clean attacked) vs adv accuracy at step
    if ops_per_exit is not None and snapshot_exit_counts and energy_clean_attacked_pj:
        denom_e = sum(energy_clean_attacked_pj) / max(1, len(energy_clean_attacked_pj))
        snap_points = []
        for k in sorted(snapshot_exit_counts.keys()):
            total_k = snapshot_total.get(int(k), 0)
            if total_k <= 0:
                continue
            counts_vec = snapshot_exit_counts[int(k)]
            energy_sum_k = 0.0
            for i_c, cnt in enumerate(counts_vec, start=1):
                fl = ops_per_exit.get(i_c, None)
                if fl is None:
                    continue
                energy_sum_k += float(cnt) * float(fl) * 3.2
            mean_e_k = energy_sum_k / float(total_k)
            ratio = (mean_e_k / denom_e) if denom_e > 0 else 0.0
            adv_acc_k = (float(snapshot_correct.get(int(k), 0)) / float(total_k)) * 100.0
            snap_points.append({'step': int(k), 'energy_ratio': float(ratio), 'adv_acc': float(adv_acc_k)})
        rec['snapshot_energy_acc_points'] = snap_points
    # Aggregate energy stats (pJ)
    if energy_clean_pj:
        rec['avg_energy_clean_pj'] = sum(energy_clean_pj) / len(energy_clean_pj) / 1e9
    else:
        rec['avg_energy_clean_pj'] = 0.0
    if energy_adv_pj:
        rec['avg_energy_adv_pj'] = sum(energy_adv_pj) / len(energy_adv_pj) / 1e9
    else:
        rec['avg_energy_adv_pj'] = 0.0
    # Energy ratio: adv over clean (unitless)
    try:
        denom = float(rec['avg_energy_clean_pj'])
        rec['energy_ratio_adv_over_clean'] = (float(rec['avg_energy_adv_pj']) / denom) if denom > 0 else 0.0
    except Exception:
        rec['energy_ratio_adv_over_clean'] = 0.0
    # Aggregate FLOPs-based FRR stats
    if frr_flops_vals:
        svals = sorted(frr_flops_vals)
        nfv = len(svals)
        frr_mean = sum(svals) / nfv
        if nfv % 2 == 1:
            frr_median = svals[nfv // 2]
        else:
            frr_median = 0.5 * (svals[nfv // 2 - 1] + svals[nfv // 2])
    else:
        frr_mean = 0.0
        frr_median = 0.0
    rec['frr_flops_count'] = len(frr_flops_vals)
    rec['frr_flops_mean'] = frr_mean
    rec['frr_flops_median'] = frr_median
    # Max confidence summary
    if maxconf_clean:
        rec['maxconf_clean_mean'] = sum(maxconf_clean) / len(maxconf_clean)
    else:
        rec['maxconf_clean_mean'] = 0.0
    if maxconf_adv:
        rec['maxconf_adv_mean'] = sum(maxconf_adv) / len(maxconf_adv)
    else:
        rec['maxconf_adv_mean'] = 0.0
    if str(device).startswith('cuda'):
        del model
        torch.cuda.empty_cache()

    # Plot combined per-batch snapshot curve if available
    try:
        if ops_per_exit is not None and combined_pairs:
            # Step-mean pairs (weighted by batch size)
            step_mean_pairs = []
            for k, items in step_pairs.items():
                if not items:
                    continue
                total_n = sum(it['n'] for it in items)
                if total_n <= 0:
                    continue
                mean_ratio = sum(it['ratio'] * it['n'] for it in items) / float(total_n)
                mean_acc = sum(it['acc'] * it['n'] for it in items) / float(total_n)
                step_mean_pairs.append((mean_ratio, mean_acc, k))
            # Prepare plot
            pairs_sorted = sorted(combined_pairs, key=lambda t: t[0])
            xs = [p[0] for p in pairs_sorted]
            ys = [p[1] for p in pairs_sorted]
            sm_sorted = sorted(step_mean_pairs, key=lambda t: t[0])
            xs_m = [p[0] for p in sm_sorted]
            ys_m = [p[1] for p in sm_sorted]
            plt.figure(figsize=(7.2, 5.0), dpi=140)
            # scatter of all batch points
            plt.scatter(xs, ys, s=14, alpha=0.35, label='batches')
            # line of step means
            if xs_m:
                plt.plot(xs_m, ys_m, '-o', color='#d62728', linewidth=2, markersize=4, label='step mean')
            # baseline clean point
            plt.scatter([1.0], [rec['clean_final_acc']], color='#2ca02c', s=30, label='clean (x=1)')
            plt.xlabel('Relative energy (adv / clean)')
            plt.ylabel('Accuracy (%)')
            plt.title(f"{model_name}: Acc vs Relative Energy")
            plt.grid(True, linestyle='--', alpha=0.35)
            plt.legend(loc='best', fontsize=8)
            os.makedirs(os.path.join(args.out_dir, 'plots'), exist_ok=True)
            ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = ''.join(c if c.isalnum() or c in ('-','_') else '_' for c in model_name)
            out_png = os.path.join(args.out_dir, 'plots', f'{safe_name}_acc_vs_rel_energy_{ts_tag}.png')
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()
            print(f"Saved per-batch energy-accuracy curve: {out_png}")
            rec['acc_vs_energy_png'] = out_png
    except Exception as e:
        print(f"[WARN] Failed to plot per-batch energy curve: {e}")
    return rec


def save_attack_summary(records: List[dict], out_dir='outputs/test_results'):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'attack_summary_{ts}.csv')
    md_path = os.path.join(out_dir, f'attack_summary_{ts}.md')

    fields = [
        'model_name','task','num_exits','eps','alpha','pgd_steps','cw','pcgrad',
        'clean_final_acc','adv_final_acc','avg_exit_clean','avg_exit_adv','avg_linf','avg_l2',
        'thresholds','lambda_exits','exit_dist_clean','exit_dist_adv','pcgrad_conflict_frac',
        'avg_energy_clean_pj','avg_energy_adv_pj','energy_ratio_adv_over_clean',
        'frr_flops_count','frr_flops_mean','frr_flops_median',
        # Early-exit curve metrics
        'eec_clean','eec_adv','eec_delta','late_exit_area_clean','late_exit_area_adv','late_exit_area_gain','haus_exitcdf','sspd_exitcdf',
        # Blocks-based complexity
        'avg_blocks_clean','avg_blocks_adv',
        # Confidence stats
        'maxconf_clean_mean','maxconf_adv_mean',
        'snapshot_energy_acc_points','acc_vs_energy_png',
        'preferred_exit','gradnorm'
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
            snaps = r.get('snapshot_energy_acc_points', []) or []
            row['snapshot_energy_acc_points'] = ';'.join(f"{int(p.get('step',0))}:{float(p.get('energy_ratio',0.0)):.6f}:{float(p.get('adv_acc',0.0)):.3f}" for p in snaps)
            wr.writerow(row)

    with open(md_path, 'w') as f:
        f.write('# Attack Summary\n\n')
        f.write(f'Generated: {ts}\n\n')
        for r in records:
            f.write(f"- {r['model_name']} (task={r['task']}, exits={r['num_exits']})\n")
            f.write(f"  - Clean/Adv: {r['clean_final_acc']:.2f}% / {r['adv_final_acc']:.2f}%\n")
            f.write(f"  - Avg exit (clean/adv): {r['avg_exit_clean']:.2f} / {r['avg_exit_adv']:.2f}\n")
            f.write(f"  - Norms (Linf/L2): {r['avg_linf']:.6f} / {r['avg_l2']:.6f}\n")
            f.write(f"  - Energy (pJ) avg (clean/adv): {r.get('avg_energy_clean_pj',0.0):.3f} / {r.get('avg_energy_adv_pj',0.0):.3f}\n")
            f.write(f"  - Energy ratio (adv/clean): {r.get('energy_ratio_adv_over_clean',0.0):.4f}\n")
            f.write(f"  - EECScore (clean/adv): {r.get('eec_clean',0.0):.4f} / {r.get('eec_adv',0.0):.4f}  (delta={r.get('eec_delta',0.0):.4f})\n")
            f.write(f"  - Late-Exit Area (clean/adv): {r.get('late_exit_area_clean',0.0):.4f} / {r.get('late_exit_area_adv',0.0):.4f}  (gain={r.get('late_exit_area_gain',0.0):.4f})\n")
            f.write(f"  - Early-exit curve distances (Haus/SSPD): {r.get('haus_exitcdf',0.0):.4f} / {r.get('sspd_exitcdf',0.0):.4f}\n")
            f.write(f"  - Avg blocks (SDN layers) clean/adv: {r.get('avg_blocks_clean',0.0):.2f} / {r.get('avg_blocks_adv',0.0):.2f}\n")
            f.write(f"  - Max confidence mean (clean/adv): {r.get('maxconf_clean_mean',0.0):.4f} / {r.get('maxconf_adv_mean',0.0):.4f}\n")
            snaps = r.get('snapshot_energy_acc_points', []) or []
            if snaps:
                f.write("  - Snapshot energy-ratio vs adv-acc points (step:ratio->acc%):\n")
                for p in snaps:
                    f.write(f"    - s{int(p.get('step',0))}: {float(p.get('energy_ratio',0.0)):.4f} -> {float(p.get('adv_acc',0.0)):.2f}%\n")
            ths = r.get('thresholds', [])
            f.write(f"  - Thresholds: {','.join(f'{x:.2f}' for x in ths) if ths else '-'}\n\n")
            if r.get('num_exits', 1) > 1:
                f.write(f"  - FRP (flops) count/mean/median: {r.get('frr_flops_count',0)} / {r.get('frr_flops_mean',0.0):.4f} / {r.get('frr_flops_median',0.0):.4f}\n\n")
            pe = r.get('preferred_exit', None)
            if pe is not None:
                f.write(f"  - Preferred exit: {pe}\n\n")

    print(f"Saved summary:\n- {csv_path}\n- {md_path}")


def main():
    parser = argparse.ArgumentParser(description='PGD Attack for SDN Models')
    parser.add_argument('--models_path', type=str, default='networks/4221', help='路径: 存放训练模型的根目录')
    parser.add_argument('--model_name', type=str, default=None, help='单模型名；为空则批量评测 models_path 下所有可加载模型')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tinyimagenet'], help='数据集名称')
    parser.add_argument('--data_root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--entropy_thresholds', type=str, default='0.20,0.20,0.30,0.3,0.35,0.35', help='逗号分隔，对应每个内部出口')
    parser.add_argument('--lambda_exits', type=str, default="1,0.9,0.8,0.7,0.6,0.5,0.4", help='逗号分隔，对应各内部出口权重；缺省=全部1.0')
    parser.add_argument('--lambda_earlyexits', type=float, default=160)
    parser.add_argument('--lambda_ce', type=float, default=1)
    parser.add_argument('--lambda_l2', type=float, default=0.01)
    parser.add_argument('--eps', type=str, default='8/255')
    parser.add_argument('--alpha', type=str, default='2/255')
    parser.add_argument('--pgd_steps', type=int, default=20)
    parser.add_argument('--cw', action='store_true', default=False, help='使用 CW 风格保持正确 (默认 False=CE)')
    parser.add_argument('--c', type=float, default=0.1, help='CW 系数 c')
    parser.add_argument('--kappa', type=float, default=3, help='CW 置信度 margin')
    parser.add_argument('--pcgrad', action='store_true', help='是否使用 PCGrad', default=False)
    parser.add_argument('--pcgrad_mode', type=str, choices=['one-sided','symmetric'], default='symmetric', help='PCGrad 投影方式：单边/对称')
    parser.add_argument('--pcgrad_partial', type=float, default=1.0, help='PCGrad 投影强度比例 [0,1]，1为完全投影')
    parser.add_argument('--gradnorm', action='store_true', help='是否使用 GradNorm 解决多目标梯度冲突', default=False)
    parser.add_argument('--gradnorm_alpha', type=float, default=1.0, help='GradNorm 的 alpha 超参数')
    parser.add_argument('--gradnorm_lr', type=float, default=1e-3, help='GradNorm 中对权重 w 的学习率')
    parser.add_argument('--same_acc_early_loss_value', default=False, action='store_true', help='使用和原始示例相似的早期分支负交叉熵法')
    parser.add_argument('--device', type=str, default='cuda:2', help='手动指定设备，如 cuda:0')
    parser.add_argument('--only', nargs='*', help='仅评测这些模型名（可多个）')
    parser.add_argument('--skip', nargs='*', default=[], help='跳过这些模型名')
    parser.add_argument('--skip_contains', nargs='*', default=["cnn","cifar10", "training"], help='排除名字中包含这些子串的模型（多值 OR）')
    parser.add_argument('--pre_target_margin', type=float, default=0.1, help='在目标出口之前的层，额外增加的熵阈值裕量 (默认0不启用)')
    parser.add_argument('--pre_target_weight', type=float, default=1.0, help='目标之前层的约束权重(乘子)')
    parser.add_argument('--prefer_exit', type=int, default=7, help='攻击时倾向让样本在该出口退出（1..N；N为最终出口）')
    parser.add_argument('--out_dir', type=str, default='outputs/test_results', help='导出结果目录')
    # PGD update options
    parser.add_argument('--pgd_update_mode', type=str, choices=['linf_sign','l2_dir','linf_projected_dir'], default='l2_dir', help='PGD 更新方向：Linf-sign 或 L2 方向（保留Linf投影）')
    parser.add_argument('--pgd_momentum', type=float, default=0.9, help='PGD 动量项系数，0禁用')
    parser.add_argument('--auto_balance_early', action='store_true', default=False, help='按梯度范数比自适应缩放早期损失')
    parser.add_argument('--ab_target_ratio', type=float, default=1.0, help='自适应缩放目标：||g_early|| ≈ ab_target_ratio * ||g_acc||')
    parser.add_argument('--grad_select', type=str, choices=['none','max_norm','max_element'], default='none',
                        help='梯度合成方式：none=加和(默认，可配合auto_balance_early)；max_norm=在两者中选择范数更大的；max_element=逐元素选择绝对值更大的。')
    parser.add_argument('--snapshot_every', type=int, default=4, help='PGD 步长间隔进行一次快照 (能耗/精度)，0=禁用')
    parser.add_argument('--snapshot_steps', type=str, default=None, help='逗号分隔的具体 PGD 步编号(1-based)进行快照，优先级高于 snapshot_every')
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