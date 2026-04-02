"""Model summary utilities."""

from __future__ import annotations


def model_summary(model) -> str:
    """Return a lightweight textual model summary."""
    lines = []
    total_params = 0
    for name, p in model.named_parameters():
        n_params = int(p.data.size)
        total_params += n_params
        lines.append(f"{name:30s} shape={tuple(p.data.shape)!s:18s} params={n_params}")
    lines.append("-" * 70)
    lines.append(f"Total trainable parameters: {total_params}")
    return "\n".join(lines)
