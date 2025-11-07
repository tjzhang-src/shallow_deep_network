import os
from typing import List, Tuple
import torch
import torch.nn as nn

# Lightweight module summarizer (mirrors the style used in profiler)
def _summarize_module(m: nn.Module) -> str:
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
    return m.__class__.__name__


def _collect_init_lines(model: nn.Module) -> List[str]:
    lines: List[str] = []
    try:
        children = list(model.init_conv.children())
        if not children:
            lines.append("(empty)")
        else:
            for c in children:
                lines.append(_summarize_module(c))
    except Exception:
        lines.append("(unknown init_conv)")
    return lines


def _collect_block_lines(model: nn.Module) -> Tuple[List[Tuple[str, List[str]]], int]:
    blocks: List[Tuple[str, List[str]]] = []
    exit_id = 0
    for i, blk in enumerate(model.layers):
        title = f"Block {i}: {blk.__class__.__name__}"
        body: List[str] = []
        # internals
        try:
            sub = blk.layers
            if isinstance(sub, nn.Sequential):
                parts = [ _summarize_module(c) for c in sub.children() ]
                body.extend(parts)
            elif isinstance(sub, nn.ModuleList):
                for idx, c in enumerate(sub):
                    body.append(f"[sub-{idx}] {_summarize_module(c)}")
            else:
                body.append(_summarize_module(sub))
        except Exception:
            body.append("(unable to summarize internals)")
        # exit
        try:
            has_exit = getattr(blk, 'output', None) is not None and not getattr(blk, 'no_output', True)
            if has_exit:
                head = blk.output
                body.append(f"Exit {exit_id}: {head.__class__.__name__}")
                exit_id += 1
            else:
                body.append("Exit: None")
        except Exception:
            body.append("Exit: (unknown)")
        blocks.append((title, body))
    return blocks, exit_id


def _collect_end_lines(model: nn.Module) -> List[str]:
    out: List[str] = []
    try:
        for c in model.end_layers.children():
            out.append(_summarize_module(c))
    except Exception:
        out.append("(unknown end_layers)")
    return out


# Simple SVG primitives

def _esc(text: str) -> str:
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))


def _text_line(x: int, y: int, text: str, font_size: int = 14) -> str:
    return f'<text x="{x}" y="{y}" font-family="monospace" font-size="{font_size}">{_esc(text)}</text>'


def _rect(x: int, y: int, w: int, h: int, rx: int = 8, ry: int = 8, stroke: str = "#333", fill: str = "#fafafa") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" />'


def _arrow(x1: int, y1: int, x2: int, y2: int, stroke: str = "#888") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="2" marker-end="url(#arrow)" />'


def render_sdn_structure_svg(model: nn.Module, out_path: str, max_width: int = 960) -> str:
    """Render SDN model as a block-style SVG image and save to out_path.

    Sections: Init, Blocks (per-layer block with optional Exit), End.
    Layout: vertical stack of rounded rectangles with a title and content lines.
    """
    padding_x = 24
    padding_y = 20
    line_h = 18
    title_h = 22
    gap = 16

    # Collect content
    init_lines = _collect_init_lines(model)
    blocks, _ = _collect_block_lines(model)
    end_lines = _collect_end_lines(model)

    # Compute per-section heights
    def box_height(num_lines: int) -> int:
        # title + lines + inner paddings
        return padding_y * 2 + title_h + (num_lines * line_h)

    sections = []  # list of (title, lines, height)
    sections.append(("Init", init_lines, box_height(len(init_lines))))
    for title, body in blocks:
        sections.append((title, body, box_height(len(body))))
    sections.append(("End", end_lines, box_height(len(end_lines))))

    total_h = gap * (len(sections) + 1) + sum(h for _, _, h in sections)
    width = max_width
    x = padding_x
    cur_y = gap
    content_x = x + 12

    svg_parts: List[str] = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}">')
    # marker for arrows
    svg_parts.append('<defs>\n  <marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">\n    <polygon points="0 0, 10 3.5, 0 7" fill="#888" />\n  </marker>\n</defs>')

    # Draw sections
    prev_center = None
    for title, lines, h in sections:
        svg_parts.append(_rect(x, cur_y, width - 2 * padding_x, h))
        # Title
        svg_parts.append(_text_line(content_x, cur_y + padding_y + title_h, f"{title}", font_size=16))
        # Body
        body_y = cur_y + padding_y + title_h + 8
        for i, s in enumerate(lines):
            svg_parts.append(_text_line(content_x, body_y + i * line_h, s, font_size=13))
        # Arrow from previous box
        center = (x + (width - 2 * padding_x) // 2, cur_y + h)
        if prev_center is not None:
            svg_parts.append(_arrow(prev_center[0], prev_center[1] + 4, center[0], cur_y - 4))
        prev_center = center
        cur_y += h + gap

    svg_parts.append('</svg>')

    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg_parts))
    return out_path
