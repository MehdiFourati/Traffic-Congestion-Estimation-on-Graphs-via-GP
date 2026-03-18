#!/usr/bin/env python3
"""
Visualize the centroid graph as an abstract graph:
- nodes = centroids
- edges = straight centroid-to-centroid connections
- no road geometry is drawn

This is better when the graph is meant for GNN work and we do not care
about physical map realism.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx


def load_gpickle(path: Path) -> nx.Graph:
    """Load a graph stored as a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize centroid graph as an abstract graph.")
    parser.add_argument("--gpickle", type=Path, required=True, help="Path to graph .gpickle")
    parser.add_argument("--save", type=Path, default=None, help="Optional output PNG path")
    parser.add_argument("--show", action="store_true", help="Show plot")

    # Optional zoom
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    # Styling
    parser.add_argument("--node-size", type=float, default=8.0)
    parser.add_argument("--node-alpha", type=float, default=0.9)
    parser.add_argument("--edge-width", type=float, default=0.35)
    parser.add_argument("--edge-alpha", type=float, default=0.15)
    parser.add_argument("--no-edges", action="store_true", help="Do not draw edges")
    parser.add_argument("--max-edges", type=int, default=200000, help="Cap number of edges drawn")

    args = parser.parse_args()

    G = load_gpickle(args.gpickle)

    # -----------------------------
    # Collect node positions
    # -----------------------------
    node_x, node_y = [], []
    missing_node_xy = 0

    for _, data in G.nodes(data=True):
        if "x" in data and "y" in data:
            try:
                node_x.append(float(data["x"]))
                node_y.append(float(data["y"]))
            except Exception:
                missing_node_xy += 1
        else:
            missing_node_xy += 1

    if not node_x or not node_y:
        raise RuntimeError("No node coordinates found. Expected x/y attributes on nodes.")

    # -----------------------------
    # Collect straight edge segments
    # -----------------------------
    edge_segments = []

    if not args.no_edges:
        count = 0

        # Works for DiGraph / Graph
        if not G.is_multigraph():
            for u, v, data in G.edges(data=True):
                if count >= args.max_edges:
                    break

                du = G.nodes[u]
                dv = G.nodes[v]

                if "x" in du and "y" in du and "x" in dv and "y" in dv:
                    try:
                        edge_segments.append([
                            (float(du["x"]), float(du["y"])),
                            (float(dv["x"]), float(dv["y"]))
                        ])
                        count += 1
                    except Exception:
                        pass

        # Works for MultiGraph / MultiDiGraph
        else:
            for u, v, _k, data in G.edges(keys=True, data=True):
                if count >= args.max_edges:
                    break

                du = G.nodes[u]
                dv = G.nodes[v]

                if "x" in du and "y" in du and "x" in dv and "y" in dv:
                    try:
                        edge_segments.append([
                            (float(du["x"]), float(du["y"])),
                            (float(dv["x"]), float(dv["y"]))
                        ])
                        count += 1
                    except Exception:
                        pass

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10, 10))
    plt.title(
        f"Abstract centroid graph | nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}\n"
        f"missing_node_xy={missing_node_xy}"
    )
    ax = plt.gca()

    # Draw edges first
    if edge_segments:
        edge_lc = LineCollection(edge_segments, linewidths=args.edge_width, alpha=args.edge_alpha)
        ax.add_collection(edge_lc)

    # Draw nodes on top
    ax.scatter(node_x, node_y, s=args.node_size, alpha=args.node_alpha)

    # Zoom
    if args.xmin is not None and args.xmax is not None and args.ymin is not None and args.ymax is not None:
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
    else:
        xmin, xmax = min(node_x), max(node_x)
        ymin, ymax = min(node_y), max(node_y)
        pad_x = (xmax - xmin) * 0.02
        pad_y = (ymax - ymin) * 0.02
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.set_aspect("equal", adjustable="box")
    plt.axis("off")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved PNG to: {args.save}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()