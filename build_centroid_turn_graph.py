from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import networkx as nx


def save_gpickle(G: nx.Graph, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_centroid_turn_graph(
    centroid_csv: Path,
    connections_csv: Path,
    links_csv: Path,
) -> nx.MultiDiGraph:
    # Load files
    df_c = pd.read_csv(centroid_csv)
    df_t = pd.read_csv(connections_csv)
    df_l = pd.read_csv(links_csv)

    # Normalize dtypes
    df_c["id"] = df_c["id"].astype(int)
    df_t["turn"] = df_t["turn"].astype(int)
    df_t["intersection"] = df_t["intersection"].astype(int)
    df_t["org"] = df_t["org"].astype(int)
    df_t["dst"] = df_t["dst"].astype(int)
    df_l["id"] = df_l["id"].astype(int)

    # Index link metadata for fast lookup
    links_by_id = df_l.set_index("id", drop=False)

    # Build node set from centroid file and turn file
    node_ids = set(df_c["id"].tolist())
    node_ids |= set(df_t["org"].tolist())
    node_ids |= set(df_t["dst"].tolist())

    # Centroid positions from centroid_pos.csv
    centroid_pos: Dict[int, Tuple[float, float]] = {
        int(r.id): (float(r.x), float(r.y)) for r in df_c.itertuples(index=False)
    }

    # Create directed multigraph (turns are directed; multiple turns between same pair possible)
    G = nx.MultiDiGraph()

    # Add nodes with attributes (centroid + segment geometry)
    for link_id in sorted(node_ids):
        node_attr = {"link_id": int(link_id)}

        # 1) Set node (x,y) centroid
        if link_id in centroid_pos:
            node_attr["x"] = float(centroid_pos[link_id][0])
            node_attr["y"] = float(centroid_pos[link_id][1])

        # 2) Attach geometry and other metadata from link_bboxes_clustered.csv
        if link_id in links_by_id.index:
            row = links_by_id.loc[link_id]

            node_attr["from_x"] = float(row["from_x"])
            node_attr["from_y"] = float(row["from_y"])
            node_attr["to_x"] = float(row["to_x"])
            node_attr["to_y"] = float(row["to_y"])

            # If centroid is missing, try c_x/c_y or midpoint of endpoints
            if "x" not in node_attr or "y" not in node_attr:
                if "c_x" in links_by_id.columns and "c_y" in links_by_id.columns:
                    try:
                        node_attr["x"] = float(row["c_x"])
                        node_attr["y"] = float(row["c_y"])
                    except Exception:
                        pass

            if "x" not in node_attr or "y" not in node_attr:
                node_attr["x"] = 0.5 * (node_attr["from_x"] + node_attr["to_x"])
                node_attr["y"] = 0.5 * (node_attr["from_y"] + node_attr["to_y"])

            # Add a few useful optional attributes if present
            for col in ["length", "out_ang", "num_lanes", "cluster", "grid_x", "grid_y", "grid_nb"]:
                if col in links_by_id.columns and pd.notna(row.get(col, None)):
                    node_attr[col] = row[col]

        else:
            # Node exists in turns but not in link metadata
            node_attr["missing_link_metadata"] = True

        G.add_node(link_id, **node_attr)

    # Add directed turn edges
    for r in df_t.itertuples(index=False):
        turn_id = int(r.turn)
        inter_id = int(r.intersection)
        org = int(r.org)
        dst = int(r.dst)
        length = float(r.length)

        G.add_edge(
            org,
            dst,
            key=turn_id,
            turn_id=turn_id,
            intersection_id=inter_id,
            length=length,
        )

    return G


def main() -> None:
    parser = argparse.ArgumentParser(description="Build centroid/turn graph (with segment geometry on nodes).")
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"), help="Path to metadata folder")
    parser.add_argument("--centroid-csv", type=Path, default=None, help="Override centroid_pos.csv path")
    parser.add_argument("--connections-csv", type=Path, default=None, help="Override connections.csv path")
    parser.add_argument("--links-csv", type=Path, default=None, help="Override link_bboxes_clustered.csv path")
    parser.add_argument("--out", type=Path, default=Path("outputs/centroid_turn.gpickle"), help="Output .gpickle path")
    args = parser.parse_args()

    centroid_csv = args.centroid_csv or (args.metadata_dir / "centroid_pos.csv")
    connections_csv = args.connections_csv or (args.metadata_dir / "connections.csv")
    links_csv = args.links_csv or (args.metadata_dir / "link_bboxes_clustered.csv")

    for p in [centroid_csv, connections_csv, links_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Could not find: {p}")

    G = build_centroid_turn_graph(centroid_csv, connections_csv, links_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_gpickle(G, args.out)

    print("Centroid/turn graph built")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()