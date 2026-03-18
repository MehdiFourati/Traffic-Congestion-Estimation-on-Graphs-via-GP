from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


def load_graph_spectrum(graph_spec_dir: str | Path):
    graph_spec_dir = Path(graph_spec_dir)

    evals = np.load(graph_spec_dir / "eigenvalues.npy")
    evecs = np.load(graph_spec_dir / "eigenvectors.npy")
    node_order = np.load(graph_spec_dir / "node_order.npy")

    evals_t = torch.tensor(evals, dtype=torch.float32)
    evecs_t = torch.tensor(evecs, dtype=torch.float32)
    node_order_t = torch.tensor(node_order, dtype=torch.long)

    return {
        "eigenvalues": evals_t,
        "eigenvectors": evecs_t,
        "node_order": node_order_t,
    }