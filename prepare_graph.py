from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def load_gpickle(path: Path):
    """Load a NetworkX graph from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_gpickle(G: nx.Graph, path: Path) -> None:
    """Save a NetworkX graph to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def graph_to_simple_undirected(G_in) -> nx.Graph:

    G = nx.Graph()

    # Copy nodes and their attributes
    for n, data in G_in.nodes(data=True):
        G.add_node(n, **data)

    # Copy edges, ignoring direction and collapsing duplicates
    if G_in.is_multigraph():
        for u, v, _k, data in G_in.edges(keys=True, data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, **data)
    else:
        for u, v, data in G_in.edges(data=True):
            if not G.has_edge(u, v):
                G.add_edge(u, v, **data)

    return G


def remove_isolated_nodes(G: nx.Graph) -> Tuple[nx.Graph, List[int]]:
    """Remove all nodes with degree 0 and return the cleaned graph + removed node list."""
    isolated = list(nx.isolates(G))
    G = G.copy()
    G.remove_nodes_from(isolated)
    return G, isolated


def build_node_order(G: nx.Graph) -> List[int]:
    """Build a fixed node ordering. Here we simply sort by node ID."""
    return sorted(G.nodes())


def build_laplacian(G: nx.Graph, node_order: List[int]) -> np.ndarray:
    """
    Build the combinatorial graph Laplacian L = D - A
    in the fixed node order.
    """
    L_sparse = nx.laplacian_matrix(G, nodelist=node_order)
    return L_sparse.astype(float).toarray()


def eigendecompose_laplacian(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the full eigendecomposition of the symmetric Laplacian.

    Since L is symmetric for an undirected graph, we use eigh.
    Returns:
    - eigenvalues: shape (n,)
    - eigenvectors: shape (n, n)
    """
    evals, evecs = np.linalg.eigh(L)
    return evals, evecs


def maybe_save_torch(array: np.ndarray, path: Path) -> bool:
    """
    Save a NumPy array as a PyTorch tensor if torch is installed.
    Returns True if saved, False otherwise.
    """
    try:
        import torch
        tensor = torch.from_numpy(array)
        torch.save(tensor, path)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare canonical undirected graph, Laplacian, and eigendecomposition."
    )
    parser.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Path to final graph .gpickle file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/graph_spectrum"),
        help="Directory where outputs will be saved",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1) Load graph
    # --------------------------------------------------
    G_raw = load_gpickle(args.graph)

    print("Loaded graph")
    print(f"  Type: {type(G_raw)}")
    print(f"  Nodes: {G_raw.number_of_nodes():,}")
    print(f"  Edges: {G_raw.number_of_edges():,}")

    # --------------------------------------------------
    # 2) Convert to simple undirected canonical graph
    # --------------------------------------------------
    G = graph_to_simple_undirected(G_raw)

    print("\nConverted to simple undirected graph")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # --------------------------------------------------
    # 3) Remove isolated nodes just in case
    # --------------------------------------------------
    G, removed_isolates = remove_isolated_nodes(G)

    print("\nRemoved isolated nodes")
    print(f"  Removed: {len(removed_isolates):,}")
    print(f"  Remaining nodes: {G.number_of_nodes():,}")
    print(f"  Remaining edges: {G.number_of_edges():,}")

    # --------------------------------------------------
    # 4) Freeze node ordering
    # --------------------------------------------------
    node_order = build_node_order(G)
    node_to_index: Dict[int, int] = {int(node_id): i for i, node_id in enumerate(node_order)}

    print("\nNode ordering frozen")
    print(f"  Number of ordered nodes: {len(node_order):,}")
    print(f"  First 10 node IDs: {node_order[:10]}")

    # --------------------------------------------------
    # 5) Build Laplacian
    # --------------------------------------------------
    L = build_laplacian(G, node_order)

    print("\nLaplacian built")
    print(f"  Shape: {L.shape}")

    # --------------------------------------------------
    # 6) Eigendecomposition
    # --------------------------------------------------
    evals, evecs = eigendecompose_laplacian(L)

    print("\nEigendecomposition done")
    print(f"  Eigenvalues shape: {evals.shape}")
    print(f"  Eigenvectors shape: {evecs.shape}")
    print(f"  Smallest 10 eigenvalues: {evals[:10]}")

    # --------------------------------------------------
    # 7) Save graph and spectral objects
    # --------------------------------------------------
    save_gpickle(G, args.out_dir / "canonical_graph.gpickle")

    np.save(args.out_dir / "node_order.npy", np.array(node_order, dtype=np.int64))
    np.save(args.out_dir / "laplacian.npy", L)
    np.save(args.out_dir / "eigenvalues.npy", evals)
    np.save(args.out_dir / "eigenvectors.npy", evecs)

    torch_saved_node_order = maybe_save_torch(np.array(node_order, dtype=np.int64), args.out_dir / "node_order.pt")
    torch_saved_L = maybe_save_torch(L, args.out_dir / "laplacian.pt")
    torch_saved_evals = maybe_save_torch(evals, args.out_dir / "eigenvalues.pt")
    torch_saved_evecs = maybe_save_torch(evecs, args.out_dir / "eigenvectors.pt")

    with open(args.out_dir / "node_to_index.json", "w", encoding="utf-8") as f:
        json.dump(node_to_index, f, indent=2)

    # Basic metadata
    n_components = nx.number_connected_components(G)
    component_sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)

    metadata = {
        "source_graph": str(args.graph),
        "canonical_graph_type": "simple_undirected",
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "removed_isolated_nodes": len(removed_isolates),
        "n_connected_components": n_components,
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "node_ordering": "sorted_node_ids",
        "laplacian_type": "combinatorial_laplacian_D_minus_A",
        "eigenvalues_shape": list(evals.shape),
        "eigenvectors_shape": list(evecs.shape),
        "torch_saved": {
            "node_order.pt": torch_saved_node_order,
            "laplacian.pt": torch_saved_L,
            "eigenvalues.pt": torch_saved_evals,
            "eigenvectors.pt": torch_saved_evecs,
        },
    }

    with open(args.out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved files to:", args.out_dir)
    print("  - canonical_graph.gpickle")
    print("  - node_order.npy")
    print("  - node_to_index.json")
    print("  - laplacian.npy")
    print("  - eigenvalues.npy")
    print("  - eigenvectors.npy")
    if any(metadata["torch_saved"].values()):
        print("  - PyTorch .pt files too")

    print("\nDone.")


if __name__ == "__main__":
    main()