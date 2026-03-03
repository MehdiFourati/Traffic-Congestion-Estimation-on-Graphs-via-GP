#!/usr/bin/env python3
"""
Visualize the centroid/turn graph locally (no HTML):
- Draw road *segments* (from_x,from_y -> to_x,to_y) taken from node attributes
- Draw centroid nodes (x,y)
- Optionally draw turn edges as lines between centroids (can get dense)

Usage:
  python visualize_centroid_turn_graph.py --gpickle outputs/centroid_turn.gpickle --show
  python visualize_centroid_turn_graph.py --gpickle outputs/centroid_turn.gpickle --save outputs/centroid_turn.png --show

Disable turn edges (cleaner map):
  python visualize_centroid_turn_graph.py --gpickle outputs/centroid_turn.gpickle --no-turns --show

Zoom:
  python visualize_centroid_turn_graph.py --gpickle outputs/centroid_turn.gpickle --show --xmin ... --xmax ... --ymin ... --ymax ...
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

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
    parser = argparse.ArgumentParser(description="Visualize centroid/turn graph (local).")
    parser.add_argument("--gpickle", type=Path, required=True, help="Path to centroid_turn.gpickle")
    parser.add_argument("--save", type=Path, default=None, help="Optional: save figure to PNG")
    parser.add_argument("--show", action="store_true", help="Show plot in a local window")

    # Zoom bounds
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)

    # Styling
    parser.add_argument("--road-width", type=float, default=0.6)
    parser.add_argument("--road-alpha", type=float, default=0.25)
    parser.add_argument("--node-size", type=float, default=8.0)
    parser.add_argument("--node-alpha", type=float, default=0.9)

    # Turn edges (can be dense)
    parser.add_argument("--no-turns", action="store_true", help="Do not draw turn edges")
    parser.add_argument("--turn-width", type=float, default=0.35)
    parser.add_argument("--turn-alpha", type=float, default=0.08)
    parser.add_argument("--max-turn-edges", type=int, default=200000, help="Cap number of turn edges drawn")
    args = parser.parse_args()

    G = load_gpickle(args.gpickle)

    # --- Road segments from node attributes ---
    road_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    missing_road_geom = 0

    # Each node is a link; draw each link once
    for _, data in G.nodes(data=True):
        try:
            x1 = float(data["from_x"])
            y1 = float(data["from_y"])
            x2 = float(data["to_x"])
            y2 = float(data["to_y"])
            road_segments.append(((x1, y1), (x2, y2)))
        except Exception:
            missing_road_geom += 1

    if not road_segments:
        raise RuntimeError("No road segments found on nodes. Expected from_x/from_y/to_x/to_y node attributes.")

    # --- Centroid node positions ---
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

    # --- Turn edges as centroid-to-centroid lines (optional) ---
    turn_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    if not args.no_turns:
        count = 0
        for u, v, _k, _data in G.edges(keys=True, data=True):
            if count >= args.max_turn_edges:
                break
            du = G.nodes[u]
            dv = G.nodes[v]
            if "x" in du and "y" in du and "x" in dv and "y" in dv:
                try:
                    turn_segments.append(((float(du["x"]), float(du["y"])), (float(dv["x"]), float(dv["y"]))))
                    count += 1
                except Exception:
                    pass

    # --- Plot ---
    plt.figure(figsize=(10, 10))
    plt.title(
        f"Centroid/turn graph | nodes={G.number_of_nodes():,}, turns={G.number_of_edges():,}\n"
        f"roads={len(road_segments):,} (missing_road_geom={missing_road_geom}, missing_node_xy={missing_node_xy})"
    )
    ax = plt.gca()

    # Draw roads (physical geometry)
    road_lc = LineCollection(road_segments, linewidths=args.road_width, alpha=args.road_alpha)
    ax.add_collection(road_lc)

    # Draw turns (not physical, can be dense)
    if turn_segments:
        turn_lc = LineCollection(turn_segments, linewidths=args.turn_width, alpha=args.turn_alpha)
        ax.add_collection(turn_lc)

    # Draw centroids
    if node_x and node_y:
        ax.scatter(node_x, node_y, s=args.node_size, alpha=args.node_alpha)

    # Zoom
    if args.xmin is not None and args.xmax is not None and args.ymin is not None and args.ymax is not None:
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
    else:
        xmin, xmax, ymin, ymax = compute_bounds_from_segments(road_segments)
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