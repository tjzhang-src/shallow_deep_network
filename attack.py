# eval_branchy_resnet18_pgd_attack.py
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from branchy_resnet18 import BranchyResNet18
import math
import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


testset = datasets.CIFAR10(root='../alexnet/data', train=False, download=True,
                           transform=transforms.Compose([transforms.Resize(224),
                                                         transforms.ToTensor()]))
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

# --- model load ---
model = BranchyResNet18(num_classes=10).to(device)
model.load_state_dict(torch.load('branchy_resnet18.v2.pth', map_location=device))
model.eval()

# --- attack / PGD hyperparams ---
eps = 8/255.0               # Linf bound
alpha = 2/255.0             # step size
pgd_steps = 20            # PGD steps
entropy_thresholds = (0.2, 0.2)  # same thresholds used by infer_with_threshold
lambda_e1 = 1.5             # 权重：对 exit1 的 hinge(ent<=t1) 惩罚
lambda_e2 = 1.2             # 权重：对 exit2 的 hinge(ent<=t2) 惩罚
lambda_ce = 3.5          # 保持预测正确的交叉熵权重 （较大以优先保持精度）
lambda_l2 = 0.01            # 扰动 L2 作为正则（可选）

cwStatus = True
c = 0.15 # cw 攻击使用的
kappa = 5
pcgrad = True
same_acc_early_loss_value = False

def normalized_entropy_from_logits(logits):
    """返回归一化熵 (batch,)（除以 log(C)）用于 hinge 判定"""
    probs = F.softmax(logits, dim=1).clamp(min=1e-12)
    ent = -(probs * probs.log()).sum(dim=1)  # natural log
    denom = math.log(max(2, logits.size(1)))
    return ent / denom

def attack_batch(model, x, y, thresholds, eps, alpha, steps,
                 lambda_e1, lambda_e2, lambda_ce, lambda_l2, cw=True, pcgrad=False,
                 same_acc_early_loss=False):
    """
    对 batch 做 constrained PGD，返回 (x_adv, delta_norms, exits_adv, preds_adv)
    仅对原本被正确分类的样本尝试，所以调用前请筛选 correct mask（见主逻辑）
    """
    x_orig = x.detach()
    batch_size = x.size(0)

    # 初始化 delta 为 0
    delta = torch.zeros_like(x, device=x.device, requires_grad=True)
    gradsisdone = 0
    # PGD（我们要最小化 loss，所以用梯度下降）
    for _ in range(steps):
        # construct perturbed input
        x_adv = (x_orig + delta).clamp(0.0, 1.0)

        # get logits (e1,e2,out)
        e1, e2, out = model.forward(x_adv)

        # compute normalized entropies

        nent1 = normalized_entropy_from_logits(e1)
        nent2 = normalized_entropy_from_logits(e2)

        # hinge losses: 若 nent <= t 则有正损失，需要被最小化 -> push nent > t
        t1, t2 = thresholds
        # 要求交叉熵打印thresholds
        if same_acc_early_loss_value:
            loss_e1 = -F.cross_entropy(e1, y) * 0.1
            loss_e2 = -F.cross_entropy(e2, y) * 0.05
        else:
            loss_e1 = torch.clamp(t1 - nent1, min=-10.0).mean()
            loss_e2 = torch.clamp(t2 - nent2, min=-10.0).mean()

        # cross-entropy to keep prediction on true label
        if cw:
            # compute f as in CW (targeted)
            target_logit = out[:, y]          # if batch, index properly
            #这里可以使用反向求导，梯度只会流向 “产生最大值的那个输入元素”，其他元素的梯度会被置为0。
            other_logit, _ = (out + -1e9 * torch.eye(out.size(1)).to(out.device)[y].to(out.device)).max(dim=1)
            # more robust batch-safe implementation uses masks; above is conceptual

            f = torch.clamp(other_logit - target_logit, min=-kappa)  # want this <= 0
            loss_ce = f.mean() * c
        else:
            loss_ce = F.cross_entropy(out, y) #+  F.cross_entropy(e2, y) F.cross_entropy(e1, y) +

        # l2 norm reg on delta (mean over batch)
        loss_l2 = (delta.view(batch_size, -1).norm(p=2, dim=1).mean())
        grads = 0
        
        if pcgrad:

            g_acc = torch.autograd.grad(loss_ce * lambda_ce, delta, retain_graph=True)[0]
            g_early = torch.autograd.grad(lambda_e1*loss_e1 + lambda_e2*loss_e2, delta)[0]
            g_reg = torch.autograd.grad(lambda_l2 * loss_l2, delta)[0]
            if torch.dot(g_acc.flatten(), g_early.flatten()) < 0:
                # 投影：移除 g_acc 在 g_early 方向上的分量
                g_acc = g_acc - (torch.dot(g_acc.flatten(), g_early.flatten()) / 
                                (g_early.flatten().norm()**2 + 1e-12)) * g_early
                gradsisdone = 1 + gradsisdone
            grads = g_acc + g_early + g_reg
        else:
            loss = lambda_e1 * loss_e1 + lambda_e2 * loss_e2 + lambda_ce * loss_ce + lambda_l2 * loss_l2
            # gradient step: minimize loss
            grads = torch.autograd.grad(loss, delta, retain_graph=False)[0]
            
        # for Linf constraint usually use sign step; since we minimize, step in negative grad direction
        delta.data = delta.data - alpha * grads.sign()
        # projection to Linf ball
        delta.data = torch.clamp(delta.data, -eps, eps)
        # ensure valid image range (optional but safe)
        delta.data = torch.clamp(x_orig + delta.data, 0.0, 1.0) - x_orig

        # zero grad for next iter
        delta.grad = None

    # final perturbed inputs
    x_adv = (x_orig + delta).clamp(0.0, 1.0)
    # evaluate
    with torch.no_grad():
        e1_adv, e2_adv, out_adv = model.forward(x_adv)
        preds_adv = out_adv.argmax(dim=1)
        # infer_with_threshold returns CPU tensors
        final_preds_adv, exits_adv = model.infer_with_threshold(x_adv, thresholds)
        # compute per-sample Linf/L2 norm
        linf = delta.abs().view(batch_size, -1).max(dim=1)[0].cpu()
        l2 = delta.view(batch_size, -1).norm(p=2, dim=1).cpu()

    return x_adv.detach(), linf, l2, final_preds_adv, exits_adv, int(gradsisdone / steps)

# --- 主循环：按 batch 攻击并统计 ---
total = 0
correct_clean = 0
correct_adv = 0

exit_counts_clean = [0, 0, 0, 0]  # index 1..3 used
exit_counts_adv = [0, 0, 0, 0]

avg_linf = []
avg_l2 = []
do_pc_grad_nums = 0
pbar = tqdm.tqdm(testloader)
for imgs, labels in pbar:
    imgs = imgs.to(device)
    labels = labels.to(device)
    batch_size = imgs.size(0)

    # 1) 先计算 clean predictions & exits
    with torch.no_grad():
        preds_clean, exits_clean = model.infer_with_threshold(imgs, entropy_thresholds)
    preds_clean = preds_clean.to(device)
    # compute which samples are correctly classified (w.r.t ground truth)
    correct_mask = (preds_clean == labels)
    correct_mask_t = correct_mask.to(device)

    # accumulate clean stats
    total += batch_size
    correct_clean += correct_mask.sum().item()
    for e in exits_clean:
        exit_counts_clean[int(e.item())] += 1

    # 2) 对那些被正确分类的样本尝试攻击（保持精度）
    if correct_mask.sum().item() == 0:
        # 没有正确样本，直接记录原 clean stats and continue
        # For fairness, we also compute adv exits as same as clean for these
        for e in exits_clean:
            exit_counts_adv[int(e.item())] += 1
        continue

    # prepare masked tensors of only correctly classified samples
    mask_idx = torch.nonzero(correct_mask_t, as_tuple=False).squeeze(1)
    imgs_correct = imgs[mask_idx]
    labels_correct = labels[mask_idx]

    # run PGD attack on this subset
    x_adv_subset, linf_vals, l2_vals, final_preds_adv_subset, exits_adv_subset, gradisdone = attack_batch(
        model, imgs_correct, labels_correct,
        entropy_thresholds, eps, alpha, pgd_steps,
        lambda_e1, lambda_e2, lambda_ce, lambda_l2,cwStatus,pcgrad=pcgrad,
        same_acc_early_loss = same_acc_early_loss_value
    )
    if gradisdone:
        do_pc_grad_nums += batch_size
    # integrate back: create adv batch = copy of original, replace masked indices with x_adv_subset
    imgs_adv = imgs.clone()
    imgs_adv[mask_idx] = x_adv_subset

    # evaluate whole batch on adv inputs
    with torch.no_grad():
        preds_adv_batch, exits_adv_batch = model.infer_with_threshold(imgs_adv, entropy_thresholds)

    # convert preds_adv_batch to device-aligned labels for accuracy check
    preds_adv_batch = preds_adv_batch.to(device)
    correct_adv += (preds_adv_batch == labels).sum().item()

    # accumulate exit counts for adv
    for e in exits_adv_batch:
        exit_counts_adv[int(e.item())] += 1

    # record norms only for attacked subset
    avg_linf.extend(linf_vals.cpu().tolist())
    avg_l2.extend(l2_vals.cpu().tolist())

    # progress display
    pbar.set_description(f"Processed {total}, clean_acc={correct_clean/total:.4f}, adv_acc={correct_adv/total:.4f}")

# overall stats
print("==== Final results ====")
print(f"Total samples: {total}")
print(f"Clean accuracy on tested set: {correct_clean/total:.4f}")
print(f"Accuracy after attack (attempted on correctly classified samples): {correct_adv/total:.4f}")
clean_value = 0
adv_value = 0
for i in [1,2,3]:
    clean_value += exit_counts_clean[i] * i
    adv_value += exit_counts_adv[i] * i
print(f"Exit distribution (clean) 1..3:{exit_counts_clean[1:]}, value : {clean_value/total}")
print(f"Exit distribution (adv)   1..3:{exit_counts_adv[1:]}, value:{adv_value/total}")
print(f"Do pc grad nums {do_pc_grad_nums/total}")
if len(avg_linf) > 0:
    print(f"Average Linf on attacked samples: {sum(avg_linf)/len(avg_linf):.6f}")
    print(f"Average L2 on attacked samples: {sum(avg_l2)/len(avg_l2):.6f}")
else:
    print("No attacked samples (no correctly-classified samples found).")