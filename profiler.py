# profiler.py
# to compute GFLOPs (inference cost) and num params of a CNN or SDN

import torch
import torch.nn as nn

import aux_funcs as af

"""
Additional SDN profiling utilities using user-specified FLOPs formulas:
 - Conv2d: H_out * W_out * K * K * C_in * C_out
 - Linear: in_features * out_features (per sample)

We compute cumulative operation counts from input to each SDN exit (internal
classifiers and final output). Batch size is set to 1 to get per-sample counts.
"""

def _count_conv2d_convfc(m: nn.Conv2d, x, y):
    """Count Conv2d ops using user's formula: H*W*K*K*C_in*C_out."""
    y0 = y
    if isinstance(y, (tuple, list)):
        y0 = y[0]
    H = int(y0.shape[-2])
    W = int(y0.shape[-1])
    kh, kw = m.kernel_size
    cin = int(m.in_channels)
    cout = int(m.out_channels)
    ops = H * W * kh * kw * cin * cout
    # store as tensor to be consistent with existing collectors
    m.total_ops += torch.tensor([int(ops)], device=m.total_ops.device)

def _count_linear_convfc(m: nn.Linear, x, y):
    """Count Linear ops using user's formula: in_features * out_features per sample."""
    y0 = y
    if isinstance(y, (tuple, list)):
        y0 = y[0]
    batch = int(y0.shape[0]) if hasattr(y0, 'shape') and y0.dim() > 0 else 1
    ops = int(m.in_features) * int(m.out_features) * batch
    m.total_ops += torch.tensor([int(ops)], device=m.total_ops.device)

def profile_sdn_convfc(model: nn.Module, input_size: int, device: torch.device):
    """Profile SDN per-exit cumulative ops using only Conv2d and Linear with
    user-specified formulas. Returns: (exit_ops), where exit_ops is a dict
    mapping exit_id (0-based) -> ops_count (int).

    For SDN, exit_id 0..N-2 are internal classifiers, exit_id N-1 is final.
    We follow the same boundary detection strategy as profile_sdn().
    """
    model.eval()

    inp = (1, 3, input_size, input_size)

    def add_hooks_convfc(m: nn.Module):
        # Only leaf modules collect stats
        if len(list(m.children())) > 0:
            return
        m.register_buffer('total_ops', torch.zeros(1))

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(_count_conv2d_convfc)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(_count_linear_convfc)
        else:
            # ignore other ops per user's requirement
            pass

    # Attach hooks
    model.apply(add_hooks_convfc)

    # Run one dummy forward to trigger hooks
    x = torch.zeros(inp, device=device)
    model.to(device)
    with torch.no_grad():
        model(x)

    # Accumulate totals and snapshot at exits similar to profile_sdn()
    exit_ops = {}
    total_ops = torch.zeros(1, device=device)

    cur_output_id = 0
    cur_output_layer_id = -10
    wait_for = -10
    for layer_id, m in enumerate(model.modules()):
        if isinstance(m, af.InternalClassifier):
            cur_output_layer_id = layer_id

        # Heuristic: SDN heads follow InternalClassifier; align with profile_sdn
        if layer_id == cur_output_layer_id + 1:
            # linear heads vs conv heads require different offset to include the head
            if isinstance(m, nn.Linear):
                wait_for = 1
            else:
                wait_for = 3

        if len(list(m.children())) > 0:
            continue

        # Sum leaf totals (only conv/linear populated)
        if hasattr(m, 'total_ops'):
            total_ops = total_ops + m.total_ops.to(device)

        if layer_id == cur_output_layer_id + wait_for:
            # Record cumulative ops up to this exit (0-based id)
            exit_ops[cur_output_id] = int(total_ops.item())
            cur_output_id += 1

    # Final output cumulative ops
    exit_ops[cur_output_id] = int(total_ops.item())

    return exit_ops

def print_sdn_model_and_convfc_flops(model: nn.Module, input_size: int, device: torch.device):
    """Pretty-print the model structure and per-exit conv+fc flops counts."""
    print('===== Model Structure =====')
    print(model)
    print('\n===== Per-exit conv+fc flops counts =====')
    exit_ops = profile_sdn_convfc(model, input_size, device)
    num_exits = len(exit_ops)
    for eid in range(num_exits):
        ops = exit_ops.get(eid, 0)
        print(f'Exit {eid} : {ops} flops')
    return exit_ops

# -----------------------
# Block-style SDN printer
# -----------------------

def _summarize_module(m: nn.Module) -> str:
    """Return a concise, single-line summary of a module."""
    if isinstance(m, nn.Conv2d):
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        sh, sw = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
        ph, pw = m.padding if isinstance(m.padding, tuple) else (m.padding, m.padding)
        return f"Conv2d({m.in_channels}->{m.out_channels}, k={kh}x{kw}, s={sh}, p={ph})"
    if isinstance(m, nn.BatchNorm2d):
        return f"BN({m.num_features})"
    if isinstance(m, nn.ReLU):
        return "ReLU"
    if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
        k = m.kernel_size
        s = m.stride
        t = m.__class__.__name__
        return f"{t}(k={k}, s={s})"
    if isinstance(m, nn.AdaptiveAvgPool2d):
        return f"AdaptiveAvgPool2d({m.output_size})"
    if isinstance(m, nn.Flatten):
        return "Flatten"
    if isinstance(m, nn.Dropout):
        return f"Dropout(p={m.p})"
    if isinstance(m, nn.Linear):
        return f"Linear({m.in_features}->{m.out_features})"
    if isinstance(m, nn.Sequential):
        parts = [ _summarize_module(c) for c in m.children() ]
        return " -> ".join(parts) if parts else "Sequential([])"
    if isinstance(m, nn.ModuleList):
        parts = [ _summarize_module(c) for c in m ]
        joined = " | ".join(parts)
        return f"ModuleList[{len(parts)}]: {joined}" if parts else "ModuleList([])"
    # Fallback
    return m.__class__.__name__

def format_sdn_block_structure(model: nn.Module) -> str:
    """Create a human-friendly block representation of an SDN model.

    Assumes common SDN interface: model.init_conv, model.layers (each has `.layers` and `.output`/`.no_output`), model.end_layers.
    Works with VGG_SDN, ResNet_SDN, WideResNet_SDN, MobileNet_SDN variants.
    """
    lines = []
    def sep(title: str):
        bar = "+" + "-" * (len(title) + 2) + "+"
        lines.append(bar)
        lines.append(f"| {title} |")
        lines.append(bar)

    # Init block
    sep("Init")
    try:
        init_parts = [ _summarize_module(c) for c in model.init_conv.children() ]
        if not init_parts:
            lines.append("(empty)")
        else:
            for p in init_parts:
                lines.append(f"  - {p}")
    except Exception:
        lines.append("(unknown init_conv)")

    # Main blocks
    lines.append("")
    lines.append("Blocks")
    lines.append("======")
    exit_id = 0
    for i, blk in enumerate(model.layers):
        title = f"Block {i}: {blk.__class__.__name__}"
        sep(title)
        # Summarize block internals
        try:
            sub = blk.layers
            if isinstance(sub, nn.Sequential):
                parts = [ _summarize_module(c) for c in sub.children() ]
                for p in parts:
                    lines.append(f"  - {p}")
            elif isinstance(sub, nn.ModuleList):
                for idx, c in enumerate(sub):
                    lines.append(f"  [sub-{idx}] {_summarize_module(c)}")
            else:
                lines.append(f"  {_summarize_module(sub)}")
        except Exception:
            lines.append("  (unable to summarize block internals)")

        # Exit head (if any)
        try:
            has_exit = getattr(blk, 'output', None) is not None and not getattr(blk, 'no_output', True)
            if has_exit:
                head = blk.output
                head_desc = _summarize_module(head)
                lines.append(f"  Exit {exit_id}: {head.__class__.__name__} -> {head_desc}")
                exit_id += 1
            else:
                lines.append("  Exit: None")
        except Exception:
            lines.append("  Exit: (unknown)")

    # End block
    lines.append("")
    sep("End")
    try:
        end_parts = [ _summarize_module(c) for c in model.end_layers.children() ]
        for p in end_parts:
            lines.append(f"  - {p}")
    except Exception:
        lines.append("  (unknown end_layers)")

    return "\n".join(lines)

def print_sdn_block_structure(model: nn.Module):
    """Print the SDN model in a block-style, human-friendly layout."""
    print(format_sdn_block_structure(model))

def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops * m.groups

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def profile_sdn(model, input_size, device):
    inp = (1, 3, input_size, input_size)
    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            #print("Not implemented for ", m)
            pass
        
    model.apply(add_hooks)

    x = torch.zeros(inp)
    x = x.to(device)
    model(x)

    output_total_ops = {}
    output_total_params = {}

    # initialize as tensors to satisfy type checkers and ensure .cpu().item() exists
    total_ops = torch.zeros(1)
    total_params = torch.zeros(1)

    cur_output_id = 0
    cur_output_layer_id = -10
    wait_for = -10
    for layer_id, m in enumerate(model.modules()):
        if isinstance(m, af.InternalClassifier):
            cur_output_layer_id = layer_id
        
        if layer_id == cur_output_layer_id + 1:
            if isinstance(m, nn.Linear):
                wait_for = 1
            else:
                wait_for = 3

        if len(list(m.children())) > 0: 
            continue

        total_ops = total_ops + m.total_ops
        total_params = total_params + m.total_params

        if layer_id == cur_output_layer_id + wait_for:
            output_total_ops[cur_output_id] = float(total_ops.cpu().item())/1e9
            output_total_params[cur_output_id] = float(total_params.cpu().item())/1e6
            cur_output_id += 1

    output_total_ops[cur_output_id] = float(total_ops.cpu().item())/1e9
    output_total_params[cur_output_id] = float(total_params.cpu().item())/1e6

    return output_total_ops, output_total_params

def profile(model, input_size, device):

    inp = (1, 3, input_size, input_size)
    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            #print("Not implemented for ", m)
            pass
        
    model.apply(add_hooks)

    x = torch.zeros(inp)
    x = x.to(device)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params
    total_ops = total_ops
    total_params = total_params

    return total_ops, total_params

