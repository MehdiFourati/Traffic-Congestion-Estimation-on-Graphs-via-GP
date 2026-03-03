from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import pickle
import pandas as pd
import networkx as nx

def save_gpickle(G: nx.Graph, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def snap_key(x: float, y: float, snap: float) -> Tuple[int, int]:
    return (int(round(x / snap)), int(round(y / snap)))


def build_intersection_graph(
    links_csv: Path,
    snap: float = 0.1,
    directed: bool = True,
) -> nx.MultiDiGraph:
    
    df = pd.read_csv(links_csv)

    # Choose graph type
    G: nx.MultiDiGraph
    if directed:
        G = nx.MultiDiGraph()
    else:
        # We'll still build as directed then convert at the end
        G = nx.MultiDiGraph()

    key_to_node: Dict[Tuple[int, int], int] = {}
    node_sum: Dict[int, Tuple[float, float, int]] = {} 
    next_node_id = 0

    def get_or_create_node(x: float, y: float) -> int:
        nonlocal next_node_id
        k = snap_key(x, y, snap)
        if k not in key_to_node:
            node_id = next_node_id
            next_node_id += 1
            key_to_node[k] = node_id
            node_sum[node_id] = (0.0, 0.0, 0)
            
            G.add_node(node_id, snap_kx=k[0], snap_ky=k[1])
        else:
            node_id = key_to_node[k]

        sx, sy, c = node_sum[node_id]
        node_sum[node_id] = (sx + float(x), sy + float(y), c + 1)
        return node_id

    # Build edges from links
    for row in df.itertuples(index=False):
        link_id = int(getattr(row, "id"))
        fx, fy = float(getattr(row, "from_x")), float(getattr(row, "from_y"))
        tx, ty = float(getattr(row, "to_x")), float(getattr(row, "to_y"))

        u = get_or_create_node(fx, fy)
        v = get_or_create_node(tx, ty)

        # Collect edge attributes
        edge_attr = {
            "link_id": link_id,
            "length": float(getattr(row, "length")),
            "out_ang": float(getattr(row, "out_ang")),
            "num_lanes": float(getattr(row, "num_lanes")),
            "from_x": fx,
            "from_y": fy,
            "to_x": tx,
            "to_y": ty,
        }

        # Use link_id as the edge key so parallel edges are distinct and reproducible but I am not sure it's the way to go actually
        G.add_edge(u, v, key=link_id, **edge_attr)

    # Finalize node coordinates as averages of contributing endpoints
    for node_id, (sx, sy, c) in node_sum.items():
        if c > 0:
            G.nodes[node_id]["x"] = sx / c
            G.nodes[node_id]["y"] = sy / c

    if not directed:
        G = nx.MultiGraph(G) 

    return G


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intersection graph from link_bboxes_clustered.csv")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("metadata"),
        help="Path to the metadata folder containing CSVs",
    )
    parser.add_argument(
        "--links-csv",
        type=Path,
        default=None,
        help="Override path to link_bboxes_clustered.csv",
    )
    parser.add_argument(
        "--snap",
        type=float,
        default=0.1,
        help="Snapping grid size for endpoint clustering (in coordinate units; try 0.1 or 1.0)",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Build an undirected graph instead of directed",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("intersection_graph"),
        help="Output path without extension (writes .graphml and .gpickle)",
    )
    args = parser.parse_args()

    links_csv = args.links_csv or (args.metadata_dir / "link_bboxes_clustered.csv")
    if not links_csv.exists():
        raise FileNotFoundError(f"Could not find: {links_csv}")

    G = build_intersection_graph(
        links_csv=links_csv,
        snap=args.snap,
        directed=not args.undirected,
    )

    out_base: Path = args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)

    
    nx.write_graphml(G, out_base.with_suffix(".graphml"))
    save_gpickle(G, out_base.with_suffix(".gpickle"))

    print("Intersection graph built")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    print(f"Wrote: {out_base.with_suffix('.graphml')}")
    print(f"Wrote: {out_base.with_suffix('.gpickle')}")


if __name__ == "__main__":
    main()