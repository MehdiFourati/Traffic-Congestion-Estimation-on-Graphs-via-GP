from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx


def load_gpickle(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_bounds_from_segments(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for (x1, y1), (x2, y2) in segments:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    return min(xs), max(xs), min(ys), max(ys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize segments + intersection nodes (local).")
    parser.add_argument("--gpickle", type=Path, required=True, help="Path to intersections.gpickle")
    parser.add_argument("--save", type=Path, default=None, help="Optional: save figure to PNG")
    parser.add_argument("--show", action="store_true", help="Show plot in a local window")

    # Manual zoom bounds
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    # Styling
    parser.add_argument("--linewidth", type=float, default=0.6, help="Road segment line width")
    parser.add_argument("--edge-alpha", type=float, default=0.35, help="Road segment transparency")
    parser.add_argument("--node-size", type=float, default=6.0, help="Intersection node size (points^2)")
    parser.add_argument("--node-alpha", type=float, default=0.9, help="Intersection node transparency")
    args = parser.parse_args()

    G = load_gpickle(args.gpickle)

    # --- Collect road segments from edges ---
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    missing_geom = 0
    for _, _, _, data in G.edges(keys=True, data=True):
        try:
            x1 = float(data["from_x"])
            y1 = float(data["from_y"])
            x2 = float(data["to_x"])
            y2 = float(data["to_y"])
            segments.append(((x1, y1), (x2, y2)))
        except Exception:
            missing_geom += 1

    if not segments:
        raise RuntimeError("No segments found on edges. Expected from_x/from_y/to_x/to_y attributes.")

    # --- Collect node coordinates ---
    node_x = []
    node_y = []
    missing_nodes = 0
    for _, data in G.nodes(data=True):
        if "x" in data and "y" in data:
            try:
                node_x.append(float(data["x"]))
                node_y.append(float(data["y"]))
            except Exception:
                missing_nodes += 1
        else:
            missing_nodes += 1

    # --- Plot ---
    plt.figure(figsize=(10, 10))
    plt.title(
        f"Road segments + intersections | nodes={G.number_of_nodes():,}, edges={len(segments):,} "
        f"(missing_edge_geom={missing_geom}, missing_node_xy={missing_nodes})"
    )

    ax = plt.gca()

    # Draw segments
    lc = LineCollection(segments, linewidths=args.linewidth, alpha=args.edge_alpha)
    ax.add_collection(lc)

    # Draw intersections on top
    if node_x and node_y:
        ax.scatter(node_x, node_y, s=args.node_size, alpha=args.node_alpha)

    # Zoom handling
    if args.xmin is not None and args.xmax is not None and args.ymin is not None and args.ymax is not None:
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
    else:
        xmin, xmax, ymin, ymax = compute_bounds_from_segments(segments)
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