from __future__ import annotations

import json
import pickle
import time
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Any

import wandb
import gpytorch
import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from Kernels.graph_matern_temporal import GraphTemporalKernel


# ============================================================
# CONFIG
# ============================================================

GRAPH_SPECTRUM_DIR = Path(
    r"C:\Users\USER\Documents\GitHub\Traffic-Congestion-Estimation-on-Graphs-via-GP\outputs\graph_spectrum"
)

GRAPH_PATH = GRAPH_SPECTRUM_DIR / "canonical_graph.gpickle"
PICKLE_DIR = Path(r"C:\Users\USER\Documents\GitHub\datasets\simbarca\all_agg_trimmed")

TRAIN_FILES = 70
VAL_FILES = 15
TEST_FILES = 16

OBS_FRACTION = 0.3

MIN_GRAPH_DEGREE = 3
MIN_AVAILABLE_NODES = 20

# ------------------------------------------------------------
# Local graph / central-node ablation
# ------------------------------------------------------------
USE_CENTRAL_NODE_RADIUS_FILTER = True
CENTRAL_NODE_ID = 8276
GRAPH_RADIUS_EDGES = 5
LOCAL_MIN_GRAPH_DEGREE = 0

# ------------------------------------------------------------
# Fixed observed-node setup
# ------------------------------------------------------------
USE_FIXED_OBSERVED_NODES = True

# If None, the code chooses one fixed random OBS_FRACTION subset of allowed nodes.
# Later, replace this by actual drone-monitored node IDs if you have them.
FIXED_OBSERVED_NODE_IDS = None

# ------------------------------------------------------------
# Historical-mean training ablation
# ------------------------------------------------------------
USE_HISTORICAL_MEAN_TRAINING = True

# ------------------------------------------------------------
# Temporal task construction
# ------------------------------------------------------------
TEMPORAL_WINDOW_STEPS = 10
CENTER_TIME_STRIDE = 5
MAX_NEIGHBOR_POINTS = 5000

# Since observed nodes are fixed, more than 1 task per center would duplicate tasks.
TRAIN_TASKS_PER_CENTER = 1
VAL_TASKS_PER_CENTER = 1
TEST_TASKS_PER_CENTER = 1

MAX_TRAIN_EVAL_TASKS = 300

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

NUM_EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 0.0
RANDOM_SEED = 42

RESAMPLE_TRAIN_MASKS = False

BASE_JITTER = 1e-5
MAX_JITTER_TRIES = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# WANDB CONFIG
# ============================================================

USE_WANDB = True

WANDB_PROJECT = "traffic-spatiotemporal-gp"
WANDB_ENTITY = None

WANDB_RUN_NAME = (
    f"central_{CENTRAL_NODE_ID}"
    f"_r_{GRAPH_RADIUS_EDGES}"
    f"_fixed_obs_{OBS_FRACTION}"
    f"_histmean_train"
    f"_tw_{TEMPORAL_WINDOW_STEPS}"
    f"_stride_{CENTER_TIME_STRIDE}"
    f"_lr_{LR}"
    f"_no_resampling"
)


# ============================================================
# UTILS
# ============================================================
def materialize_kernel(K: torch.Tensor) -> torch.Tensor:
    if hasattr(K, "to_dense"):
        return K.to_dense()
    if hasattr(K, "evaluate"):
        return K.evaluate()
    return K


def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def stabilize_covariance(
    cov: torch.Tensor,
    base_jitter: float = BASE_JITTER,
    max_tries: int = MAX_JITTER_TRIES,
) -> torch.Tensor:
    cov = symmetrize(cov)
    n = cov.shape[-1]
    I = torch.eye(n, device=cov.device, dtype=cov.dtype)

    jitter = base_jitter
    for _ in range(max_tries):
        test_cov = cov + jitter * I
        try:
            torch.linalg.cholesky(test_cov)
            return test_cov
        except RuntimeError:
            jitter *= 10.0

    return cov + jitter * I


def compute_metrics(df: pd.DataFrame, scale: float) -> dict[str, float]:
    if len(df) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "nmae": float("nan"),
            "nrmse": float("nan"),
        }

    mae = float(df["abs_error"].mean())
    rmse = float(np.sqrt(df["sq_error"].mean()))

    if scale <= 0:
        nmae = float("nan")
        nrmse = float("nan")
    else:
        nmae = mae / scale
        nrmse = rmse / scale

    return {
        "mae": mae,
        "rmse": rmse,
        "nmae": nmae,
        "nrmse": nrmse,
    }


def num_batches(n_items: int, batch_size: int) -> int:
    if n_items == 0:
        return 0
    return ceil(n_items / batch_size)


def iter_task_batches(tasks: list[dict], batch_size: int):
    for start in range(0, len(tasks), batch_size):
        yield tasks[start:start + batch_size]


def task_to_device(task: dict, device: torch.device) -> dict:
    x_obs = torch.tensor(task["x_obs_st"], dtype=torch.float32, device=device)
    y_obs = torch.tensor(task["y_obs_std"], dtype=torch.float32, device=device)

    x_tgt = torch.tensor(task["x_tgt_st"], dtype=torch.float32, device=device)
    y_tgt = torch.tensor(task["y_tgt_std"], dtype=torch.float32, device=device)

    return {
        "x_obs": x_obs,
        "y_obs": y_obs,
        "x_tgt": x_tgt,
        "y_tgt": y_tgt,
    }


def time_of_day_hours(index: pd.DatetimeIndex) -> np.ndarray:
    return (
        index.hour.to_numpy(dtype=np.float32)
        + index.minute.to_numpy(dtype=np.float32) / 60.0
        + index.second.to_numpy(dtype=np.float32) / 3600.0
    )


# ============================================================
# PICKLE / DATAFRAME LOADING
# ============================================================
def extract_pred_vtime_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, dict):
        candidate_keys = [
            "pred_vtime",
            "drone_vtime",
            "vtime",
            "vtime_pred",
        ]
        for key in candidate_keys:
            if key in obj and isinstance(obj[key], pd.DataFrame):
                return obj[key]

    raise ValueError(
        "Could not find a pred_vtime DataFrame in the pickle object. "
        "Please adapt extract_pred_vtime_df()."
    )


def load_pred_vtime_df(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    df = extract_pred_vtime_df(obj).copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    try:
        df.columns = pd.Index([int(c) for c in df.columns])
    except Exception:
        pass

    return df


def dataframe_to_sparse_rows(df: pd.DataFrame) -> dict:
    times = df.index
    time_hours = time_of_day_hours(times)

    row_dicts = []
    for _, row in df.iterrows():
        values = {}
        for col, val in row.items():
            if pd.notna(val):
                try:
                    values[int(col)] = float(val)
                except Exception:
                    pass
        row_dicts.append(values)

    return {
        "timestamps": list(times),
        "time_hours": time_hours.astype(np.float32),
        "rows": row_dicts,
    }


def build_time_series_cache(
    file_list: list[Path],
    cache_path: Path | None = None,
) -> list[dict]:
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached time-series rows from: {cache_path}")
        with open(cache_path, "rb") as f:
            out = pickle.load(f)
        print(f"Loaded {len(out)} cached files.")
        return out

    out = []

    for i, fp in enumerate(file_list, start=1):
        t0 = time.time()
        print(f"[{i:03d}/{len(file_list):03d}] Loading {fp.name} ...", flush=True)

        df = load_pred_vtime_df(fp)
        packed = dataframe_to_sparse_rows(df)

        out.append({
            "file": fp.name,
            "timestamps": packed["timestamps"],
            "time_hours": packed["time_hours"],
            "rows": packed["rows"],
        })

        print(
            f"    loaded in {time.time() - t0:.2f}s | "
            f"rows={len(packed['rows'])}",
            flush=True,
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
        print(f"Saved cached time-series rows to: {cache_path}")

    return out


def build_historical_mean_train_series(
    train_series: list[dict],
    allowed_nodes: set[int] | None = None,
    synthetic_date: str = "2000-01-01",
) -> list[dict]:
    """
    Build one synthetic training file by averaging the 70 training days.

    For each time-of-day and node:
        mean_value(time, node) = average over train files where that node exists.
    """
    sums_by_time = defaultdict(lambda: defaultdict(float))
    counts_by_time = defaultdict(lambda: defaultdict(int))

    for file_data in train_series:
        rows = file_data["rows"]
        time_hours = file_data["time_hours"]

        for t_hour, row_values in zip(time_hours, rows):
            time_key = int(round(float(t_hour) * 3600.0))

            for node_id, value in row_values.items():
                node_id = int(node_id)

                if allowed_nodes is not None and node_id not in allowed_nodes:
                    continue

                sums_by_time[time_key][node_id] += float(value)
                counts_by_time[time_key][node_id] += 1

    if len(sums_by_time) == 0:
        raise RuntimeError("Could not build historical mean series: no values found.")

    sorted_time_keys = sorted(sums_by_time.keys())

    timestamps = []
    time_hours = []
    mean_rows = []

    base_date = pd.Timestamp(synthetic_date)

    for time_key in sorted_time_keys:
        timestamp = base_date + pd.Timedelta(seconds=int(time_key))
        timestamps.append(timestamp)
        time_hours.append(float(time_key) / 3600.0)

        mean_row = {}

        for node_id, total in sums_by_time[time_key].items():
            count = counts_by_time[time_key][node_id]
            if count > 0:
                mean_row[int(node_id)] = float(total / count)

        mean_rows.append(mean_row)

    out = [{
        "file": "historical_mean_train_70_days",
        "timestamps": timestamps,
        "time_hours": np.asarray(time_hours, dtype=np.float32),
        "rows": mean_rows,
    }]

    print("\nBuilt historical-mean training series")
    print(f"Original train files:       {len(train_series)}")
    print(f"Synthetic mean files:       {len(out)}")
    print(f"Synthetic timestamps:       {len(mean_rows)}")
    print(f"Allowed nodes used:         {len(allowed_nodes) if allowed_nodes is not None else 'all'}")

    non_empty_rows = sum(1 for row in mean_rows if len(row) > 0)
    print(f"Non-empty mean rows:        {non_empty_rows}")

    return out


# ============================================================
# GRAPH + SPECTRAL OBJECTS
# ============================================================
def load_graph(graph_path: Path) -> nx.Graph:
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    if not isinstance(G, nx.Graph):
        raise TypeError(f"Loaded object is not a NetworkX graph, got: {type(G)}")

    return G


def _load_node_to_pos(spectrum_dir: Path) -> dict[int, int]:
    json_path = spectrum_dir / "node_to_index.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}

    node_order_pt = spectrum_dir / "node_order.pt"
    node_order_npy = spectrum_dir / "node_order.npy"

    if node_order_pt.exists():
        node_order_obj = torch.load(node_order_pt, map_location="cpu")
        if isinstance(node_order_obj, torch.Tensor):
            node_order = node_order_obj.tolist()
        else:
            node_order = list(node_order_obj)
        return {int(node): i for i, node in enumerate(node_order)}

    if node_order_npy.exists():
        node_order = np.load(node_order_npy, allow_pickle=True).tolist()
        return {int(node): i for i, node in enumerate(node_order)}

    raise FileNotFoundError(
        "Could not find node_to_index.json or node_order.pt/.npy in graph_spectrum."
    )


def _load_tensor_file(path: Path) -> torch.Tensor:
    if path.suffix == ".pt":
        obj = torch.load(path, map_location=DEVICE)
        if isinstance(obj, torch.Tensor):
            return obj.float()
        return torch.tensor(obj, dtype=torch.float32, device=DEVICE)

    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

    raise ValueError(f"Unsupported tensor file format: {path}")


def build_graph_objects(graph_path: Path, spectrum_dir: Path):
    G = load_graph(graph_path)

    eigvals_path = spectrum_dir / "eigenvalues.pt"
    eigvecs_path = spectrum_dir / "eigenvectors.pt"

    if not eigvals_path.exists():
        eigvals_path = spectrum_dir / "eigenvalues.npy"
    if not eigvecs_path.exists():
        eigvecs_path = spectrum_dir / "eigenvectors.npy"

    if not eigvals_path.exists():
        raise FileNotFoundError("Could not find eigenvalues.pt or eigenvalues.npy")
    if not eigvecs_path.exists():
        raise FileNotFoundError("Could not find eigenvectors.pt or eigenvectors.npy")

    eigenvalues = _load_tensor_file(eigvals_path).to(DEVICE)
    eigenvectors = _load_tensor_file(eigvecs_path).to(DEVICE)
    node_to_pos = _load_node_to_pos(spectrum_dir)

    if eigenvalues.ndim != 1:
        raise ValueError(f"eigenvalues must be 1D, got shape {tuple(eigenvalues.shape)}")
    if eigenvectors.ndim != 2:
        raise ValueError(f"eigenvectors must be 2D, got shape {tuple(eigenvectors.shape)}")
    if eigenvectors.shape[0] != eigenvectors.shape[1]:
        raise ValueError(f"eigenvectors must be square, got shape {tuple(eigenvectors.shape)}")
    if eigenvectors.shape[0] != eigenvalues.shape[0]:
        raise ValueError(
            f"Mismatch: eigenvectors has shape {tuple(eigenvectors.shape)} "
            f"but eigenvalues has shape {tuple(eigenvalues.shape)}"
        )

    graph_nodes = set(int(n) for n in G.nodes())
    mapped_nodes = set(node_to_pos.keys())

    missing_in_mapping = graph_nodes - mapped_nodes
    if missing_in_mapping:
        raise ValueError(
            "Some graph nodes are missing from node_to_index mapping. "
            f"Example missing nodes: {list(sorted(missing_in_mapping))[:10]}"
        )

    if len(node_to_pos) != eigenvalues.shape[0]:
        raise ValueError(
            f"node_to_pos has {len(node_to_pos)} entries but eigenvalues has length {eigenvalues.shape[0]}"
        )

    return G, node_to_pos, eigenvalues, eigenvectors


def get_degree_filtered_nodes(
    G: nx.Graph,
    node_to_pos: dict[int, int],
    min_degree: int,
) -> set[int]:
    allowed = {
        int(n)
        for n, deg in G.degree()
        if int(deg) >= int(min_degree) and int(n) in node_to_pos
    }

    if len(allowed) == 0:
        raise RuntimeError(
            f"No nodes left after applying degree >= {min_degree} filter."
        )

    return allowed


def get_radius_filtered_nodes(
    G: nx.Graph,
    node_to_pos: dict[int, int],
    central_node: int,
    radius_edges: int,
    local_min_degree: int = 0,
) -> set[int]:
    central_node = int(central_node)

    if central_node not in G:
        raise ValueError(f"central_node={central_node} is not in the graph.")

    if central_node not in node_to_pos:
        raise ValueError(f"central_node={central_node} is missing from node_to_pos.")

    lengths = nx.single_source_shortest_path_length(
        G,
        central_node,
        cutoff=int(radius_edges),
    )

    allowed = {
        int(n)
        for n in lengths.keys()
        if int(n) in node_to_pos and int(G.degree[n]) >= int(local_min_degree)
    }

    if len(allowed) == 0:
        raise RuntimeError(
            f"No nodes left inside radius {radius_edges} from central node {central_node}."
        )

    print("\nApplying central-node radius filter")
    print(f"Central node:              {central_node}")
    print(f"Central node degree:       {G.degree[central_node]}")
    print(f"Radius in graph edges:     {radius_edges}")
    print(f"Local min graph degree:    {local_min_degree}")
    print(f"Allowed local nodes:       {len(allowed)}")

    return allowed


def choose_fixed_observed_nodes(
    allowed_nodes: set[int],
    obs_fraction: float,
    seed: int,
) -> set[int]:
    """
    Choose one fixed observed-node set used for train/val/test.

    This simulates drone acquisition where the same streets/intersections
    are observed across all days.
    """
    allowed_sorted = sorted(int(n) for n in allowed_nodes)

    n_total = len(allowed_sorted)
    n_obs = int(round(obs_fraction * n_total))
    n_obs = max(1, n_obs)
    n_obs = min(n_obs, n_total - 1)

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(n_total, size=n_obs, replace=False)

    fixed_observed_nodes = {allowed_sorted[i] for i in chosen_idx}

    print("\nFixed observed-node setup")
    print(f"Total allowed nodes:        {n_total}")
    print(f"Fixed observed nodes:       {len(fixed_observed_nodes)}")
    print(f"Fixed target candidates:    {n_total - len(fixed_observed_nodes)}")
    print("First 20 fixed observed nodes:")
    print(sorted(fixed_observed_nodes)[:20])

    return fixed_observed_nodes


# ============================================================
# DATASET BUILDING
# ============================================================
def fit_standardization(
    train_series: list[dict],
    allowed_nodes: set[int],
) -> tuple[float, float]:
    vals = []

    for file_data in train_series:
        for row_values in file_data["rows"]:
            for node_id, value in row_values.items():
                if node_id in allowed_nodes:
                    vals.append(value)

    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        raise RuntimeError("No values available to fit normalization on train split.")

    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-8:
        std = 1.0

    return mean, std


def sample_obs_target_split(
    available_nodes: list[int],
    obs_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    n = len(available_nodes)
    if n < 2:
        return [], []

    n_obs = int(round(obs_fraction * n))
    n_obs = max(1, n_obs)
    n_obs = min(n_obs, n - 1)

    perm = rng.permutation(n)

    obs_idx = perm[:n_obs]
    tgt_idx = perm[n_obs:]

    obs_nodes = sorted(available_nodes[i] for i in obs_idx)
    tgt_nodes = sorted(available_nodes[i] for i in tgt_idx)

    return obs_nodes, tgt_nodes


def build_spatiotemporal_tasks(
    series_data: list[dict],
    node_to_pos: dict[int, int],
    allowed_nodes: set[int],
    y_mean: float,
    y_std: float,
    obs_fraction: float,
    num_tasks_per_center: int,
    base_seed: int,
    fixed_observed_nodes: set[int] | None = None,
    temporal_window_steps: int = TEMPORAL_WINDOW_STEPS,
    center_time_stride: int = CENTER_TIME_STRIDE,
    min_available_nodes: int = MIN_AVAILABLE_NODES,
    max_neighbor_points: int = MAX_NEIGHBOR_POINTS,
) -> list[dict]:
    """
    Build spatiotemporal reconstruction tasks.

    If fixed_observed_nodes is provided:
      - observed nodes are the same across train/val/test
      - targets are all available non-observed nodes

    If fixed_observed_nodes is None:
      - original random mask behavior is used
    """
    tasks = []
    mapped_nodes = set(node_to_pos.keys())
    usable_nodes = mapped_nodes & allowed_nodes

    for file_idx, file_data in enumerate(series_data):
        rows = file_data["rows"]
        timestamps = file_data["timestamps"]
        time_hours = file_data["time_hours"]

        if len(rows) == 0:
            continue

        center_indices = list(range(0, len(rows), center_time_stride))

        for center_idx in center_indices:
            center_row = rows[center_idx]
            center_time = float(time_hours[center_idx])

            center_nodes = sorted([n for n in center_row if n in usable_nodes])

            if len(center_nodes) < min_available_nodes:
                continue

            left = max(0, center_idx - temporal_window_steps)
            right = min(len(rows), center_idx + temporal_window_steps + 1)

            for task_rep in range(num_tasks_per_center):
                rng = np.random.default_rng(
                    base_seed + 100000 * file_idx + 1000 * center_idx + task_rep
                )

                if fixed_observed_nodes is not None:
                    obs_center_nodes = sorted(
                        n for n in center_nodes
                        if n in fixed_observed_nodes
                    )

                    tgt_nodes = sorted(
                        n for n in center_nodes
                        if n not in fixed_observed_nodes
                    )
                else:
                    obs_center_nodes, tgt_nodes = sample_obs_target_split(
                        available_nodes=center_nodes,
                        obs_fraction=obs_fraction,
                        rng=rng,
                    )

                if len(obs_center_nodes) == 0 or len(tgt_nodes) == 0:
                    continue

                obs_points = []

                for n in obs_center_nodes:
                    obs_points.append((int(n), center_time, float(center_row[n])))

                neighbor_points = []

                for ridx in range(left, right):
                    if ridx == center_idx:
                        continue

                    row_values = rows[ridx]
                    tval = float(time_hours[ridx])

                    # Leakage-safe:
                    # only the same fixed/observed nodes are available at neighbor times.
                    for n in obs_center_nodes:
                        if n in row_values:
                            neighbor_points.append(
                                (int(n), tval, float(row_values[n]))
                            )

                if len(neighbor_points) > max_neighbor_points:
                    chosen = rng.choice(
                        len(neighbor_points),
                        size=max_neighbor_points,
                        replace=False,
                    )
                    neighbor_points = [neighbor_points[i] for i in chosen]

                obs_points.extend(neighbor_points)

                if len(obs_points) == 0:
                    continue

                x_obs_st = np.array(
                    [[node_to_pos[n], t] for (n, t, _) in obs_points],
                    dtype=np.float32,
                )
                y_obs_raw = np.array([v for (_, _, v) in obs_points], dtype=np.float32)
                y_obs_std = ((y_obs_raw - y_mean) / y_std).astype(np.float32)

                x_tgt_st = np.array(
                    [[node_to_pos[n], center_time] for n in tgt_nodes],
                    dtype=np.float32,
                )
                y_tgt_raw = np.array([center_row[n] for n in tgt_nodes], dtype=np.float32)
                y_tgt_std = ((y_tgt_raw - y_mean) / y_std).astype(np.float32)

                tasks.append({
                    "file": file_data["file"],
                    "timestamp": timestamps[center_idx],
                    "center_time_hour": center_time,
                    "task_rep": int(task_rep),

                    "observed_center_nodes": [int(n) for n in obs_center_nodes],
                    "target_nodes": [int(n) for n in tgt_nodes],
                    "num_obs": int(len(obs_points)),
                    "num_targets": int(len(tgt_nodes)),

                    "x_obs_st": x_obs_st,
                    "x_tgt_st": x_tgt_st,
                    "y_obs_std": y_obs_std,
                    "y_tgt_std": y_tgt_std,

                    "y_obs_raw": y_obs_raw,
                    "y_tgt_raw": y_tgt_raw,
                })

    return tasks


# ============================================================
# SHARED CONDITIONAL SPATIOTEMPORAL GP MODEL
# ============================================================
class SharedConditionalSpatioTemporalGraphModel(torch.nn.Module):
    def __init__(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        base_jitter: float = BASE_JITTER,
    ):
        super().__init__()

        time_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

        self.kernel = GraphTemporalKernel(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            time_kernel=time_kernel,
        )

        self.raw_mean = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.raw_noise = torch.nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        self.base_jitter = float(base_jitter)

    @property
    def mean(self) -> torch.Tensor:
        return self.raw_mean

    @property
    def noise(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_noise) + 1e-8

    def _cov(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        K = self.kernel(x1, x2)
        return materialize_kernel(K)

    def conditional_distribution(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
        x_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = self.noise

        K_oo = self._cov(x_obs, x_obs)
        K_ot = self._cov(x_obs, x_tgt)
        K_to = K_ot.transpose(-1, -2)
        K_tt = self._cov(x_tgt, x_tgt)

        I_obs = torch.eye(K_oo.shape[-1], device=K_oo.device, dtype=K_oo.dtype)
        I_tgt = torch.eye(K_tt.shape[-1], device=K_tt.device, dtype=K_tt.dtype)

        K_oo_noisy = K_oo + noise * I_obs
        K_tt_noisy = K_tt + noise * I_tgt

        K_oo_noisy = stabilize_covariance(K_oo_noisy, base_jitter=self.base_jitter)
        K_tt_noisy = stabilize_covariance(K_tt_noisy, base_jitter=self.base_jitter)

        mean_obs = self.mean.expand(x_obs.shape[0])
        mean_tgt = self.mean.expand(x_tgt.shape[0])

        resid = (y_obs - mean_obs).unsqueeze(-1)

        alpha = torch.linalg.solve(K_oo_noisy, resid)
        pred_mean = mean_tgt.unsqueeze(-1) + K_to @ alpha
        pred_mean = pred_mean.squeeze(-1)

        pred_cov = K_tt_noisy - K_to @ torch.linalg.solve(K_oo_noisy, K_ot)
        pred_cov = stabilize_covariance(pred_cov, base_jitter=self.base_jitter)

        return pred_mean, pred_cov

    def task_nll(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
        x_tgt: torch.Tensor,
        y_tgt: torch.Tensor,
        normalize_by_targets: bool = True,
    ) -> torch.Tensor:
        pred_mean, pred_cov = self.conditional_distribution(x_obs, y_obs, x_tgt)
        dist = torch.distributions.MultivariateNormal(
            pred_mean,
            covariance_matrix=pred_cov,
        )
        nll = -dist.log_prob(y_tgt)

        if normalize_by_targets:
            nll = nll / max(int(y_tgt.numel()), 1)

        return nll


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_tasks(
    model: SharedConditionalSpatioTemporalGraphModel,
    tasks: list[dict],
    y_mean: float,
    y_std: float,
    batch_size: int = EVAL_BATCH_SIZE,
    split_name: str = "Eval",
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    model.eval()

    rows = []
    covariance_records = []

    batches = iter_task_batches(tasks, batch_size)
    total_batches = num_batches(len(tasks), batch_size)

    if show_progress:
        batches = tqdm(
            batches,
            total=total_batches,
            desc=split_name,
            unit="batch",
            leave=False,
        )

    for batch in batches:
        for task in batch:
            dev_task = task_to_device(task, DEVICE)

            pred_mean_std, pred_cov_std = model.conditional_distribution(
                x_obs=dev_task["x_obs"],
                y_obs=dev_task["y_obs"],
                x_tgt=dev_task["x_tgt"],
            )

            pred_mean_std_np = pred_mean_std.detach().cpu().numpy()
            pred_cov_std_np = pred_cov_std.detach().cpu().numpy()

            pred_var_std_np = np.diag(pred_cov_std_np)
            pred_std_std_np = np.sqrt(np.maximum(pred_var_std_np, 1e-12))

            pred_mean_raw = pred_mean_std_np * y_std + y_mean
            pred_std_raw = pred_std_std_np * y_std
            true_raw = task["y_tgt_raw"]

            covariance_records.append({
                "file": task["file"],
                "timestamp": str(task["timestamp"]),
                "center_time_hour": float(task["center_time_hour"]),
                "observed_center_nodes": [int(n) for n in task["observed_center_nodes"]],
                "target_nodes": [int(n) for n in task["target_nodes"]],
                "pred_mean_raw": pred_mean_raw.astype(float),
                "pred_cov_raw": (pred_cov_std_np * (y_std ** 2)).astype(float),
                "true_values_raw": true_raw.astype(float),
                "observed_values_raw": task["y_obs_raw"].astype(float),
            })

            for k, node_id in enumerate(task["target_nodes"]):
                rows.append({
                    "file": task["file"],
                    "timestamp": task["timestamp"],
                    "center_time_hour": float(task["center_time_hour"]),
                    "task_rep": int(task["task_rep"]),
                    "num_obs": int(task["num_obs"]),
                    "num_targets": int(task["num_targets"]),
                    "target_node": int(node_id),
                    "pred_mean": float(pred_mean_raw[k]),
                    "pred_std": float(pred_std_raw[k]),
                    "true_value": float(true_raw[k]),
                    "abs_error": float(abs(pred_mean_raw[k] - true_raw[k])),
                    "sq_error": float((pred_mean_raw[k] - true_raw[k]) ** 2),
                    "norm_abs_error": float(abs(pred_mean_raw[k] - true_raw[k]) / y_std),
                    "norm_sq_error": float(((pred_mean_raw[k] - true_raw[k]) / y_std) ** 2),
                })

    return pd.DataFrame(rows), covariance_records


# ============================================================
# MAIN
# ============================================================
def main():
    pickle_files = sorted(PICKLE_DIR.glob("*.pkl"))
    assert len(pickle_files) == 101, f"Expected 101 pickle files, found {len(pickle_files)}"
    assert TRAIN_FILES + VAL_FILES + TEST_FILES == 101

    train_files = pickle_files[:TRAIN_FILES]
    val_files = pickle_files[TRAIN_FILES:TRAIN_FILES + VAL_FILES]
    test_files = pickle_files[TRAIN_FILES + VAL_FILES:]

    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")

    print("First 3 train files:")
    for fp in train_files[:3]:
        print("  ", fp.name)

    G, node_to_pos, eigenvalues, eigenvectors = build_graph_objects(
        GRAPH_PATH,
        GRAPH_SPECTRUM_DIR,
    )

    print(f"Graph nodes: {len(G.nodes())}")
    print("Loaded precomputed spectrum:")
    print("  eigenvalues shape :", tuple(eigenvalues.shape))
    print("  eigenvectors shape:", tuple(eigenvectors.shape))
    print("  mapped nodes      :", len(node_to_pos))

    # --------------------------------------------------------
    # Node filtering
    # --------------------------------------------------------
    if USE_CENTRAL_NODE_RADIUS_FILTER:
        central_node = int(CENTRAL_NODE_ID)

        allowed_nodes = get_radius_filtered_nodes(
            G=G,
            node_to_pos=node_to_pos,
            central_node=central_node,
            radius_edges=GRAPH_RADIUS_EDGES,
            local_min_degree=LOCAL_MIN_GRAPH_DEGREE,
        )
    else:
        central_node = None

        allowed_nodes = get_degree_filtered_nodes(
            G=G,
            node_to_pos=node_to_pos,
            min_degree=MIN_GRAPH_DEGREE,
        )

    filtered_subgraph = G.subgraph(allowed_nodes).copy()

    print("\nFiltered graph summary")
    print(f"Allowed nodes:             {len(allowed_nodes)}")
    print(f"Filtered subgraph nodes:   {filtered_subgraph.number_of_nodes()}")
    print(f"Filtered subgraph edges:   {filtered_subgraph.number_of_edges()}")

    if USE_CENTRAL_NODE_RADIUS_FILTER:
        print(f"Central node used:         {central_node}")
        print(f"Graph radius used:         {GRAPH_RADIUS_EDGES}")
    else:
        print(f"Degree filter used:        degree >= {MIN_GRAPH_DEGREE}")

    # --------------------------------------------------------
    # Fixed observed nodes
    # --------------------------------------------------------
    if USE_FIXED_OBSERVED_NODES:
        if FIXED_OBSERVED_NODE_IDS is None:
            fixed_observed_nodes = choose_fixed_observed_nodes(
                allowed_nodes=allowed_nodes,
                obs_fraction=OBS_FRACTION,
                seed=RANDOM_SEED + 999,
            )
        else:
            fixed_observed_nodes = {
                int(n)
                for n in FIXED_OBSERVED_NODE_IDS
                if int(n) in allowed_nodes
            }

            if len(fixed_observed_nodes) == 0:
                raise RuntimeError("FIXED_OBSERVED_NODE_IDS produced an empty observed-node set.")

            print("\nUsing hard-coded fixed observed nodes")
            print(f"Fixed observed nodes: {len(fixed_observed_nodes)}")
            print("First 20 fixed observed nodes:")
            print(sorted(fixed_observed_nodes)[:20])
    else:
        fixed_observed_nodes = None

    fixed_observed_node_ids = (
        sorted(int(n) for n in fixed_observed_nodes)
        if fixed_observed_nodes is not None
        else []
    )

    # --------------------------------------------------------
    # Load sparse time series per file
    # --------------------------------------------------------
    cache_dir = Path("cache_spatiotemporal_rows")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_cache = cache_dir / "train_series.pkl"
    val_cache = cache_dir / "val_series.pkl"
    test_cache = cache_dir / "test_series.pkl"

    train_series = build_time_series_cache(train_files, cache_path=train_cache)
    val_series = build_time_series_cache(val_files, cache_path=val_cache)
    test_series = build_time_series_cache(test_files, cache_path=test_cache)

    print(f"Train files loaded: {len(train_series)}")
    print(f"Val files loaded:   {len(val_series)}")
    print(f"Test files loaded:  {len(test_series)}")

    train_series_raw = train_series

    if USE_HISTORICAL_MEAN_TRAINING:
        train_series_for_tasks = build_historical_mean_train_series(
            train_series=train_series_raw,
            allowed_nodes=allowed_nodes,
            synthetic_date="2000-01-01",
        )
    else:
        train_series_for_tasks = train_series_raw

    train_tasks_source = (
        "historical_mean_70_days"
        if USE_HISTORICAL_MEAN_TRAINING
        else "raw_train_days"
    )

    # --------------------------------------------------------
    # W&B run
    # --------------------------------------------------------
    run = None

    if USE_WANDB:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            config={
                "train_files": TRAIN_FILES,
                "val_files": VAL_FILES,
                "test_files": TEST_FILES,

                "obs_fraction": OBS_FRACTION,
                "min_graph_degree": MIN_GRAPH_DEGREE,
                "min_available_nodes": MIN_AVAILABLE_NODES,

                "use_central_node_radius_filter": USE_CENTRAL_NODE_RADIUS_FILTER,
                "central_node_id": CENTRAL_NODE_ID,
                "central_node_used": central_node,
                "graph_radius_edges": GRAPH_RADIUS_EDGES,
                "local_min_graph_degree": LOCAL_MIN_GRAPH_DEGREE,

                "use_fixed_observed_nodes": USE_FIXED_OBSERVED_NODES,
                "num_fixed_observed_nodes": len(fixed_observed_node_ids),
                "fixed_observed_node_ids": fixed_observed_node_ids,

                "use_historical_mean_training": USE_HISTORICAL_MEAN_TRAINING,
                "train_tasks_source": train_tasks_source,

                "temporal_window_steps": TEMPORAL_WINDOW_STEPS,
                "center_time_stride": CENTER_TIME_STRIDE,
                "max_neighbor_points": MAX_NEIGHBOR_POINTS,

                "train_tasks_per_center": TRAIN_TASKS_PER_CENTER,
                "val_tasks_per_center": VAL_TASKS_PER_CENTER,
                "test_tasks_per_center": TEST_TASKS_PER_CENTER,

                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,

                "num_epochs": NUM_EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "random_seed": RANDOM_SEED,

                "resample_train_masks": RESAMPLE_TRAIN_MASKS,

                "base_jitter": BASE_JITTER,
                "max_jitter_tries": MAX_JITTER_TRIES,

                "device": str(DEVICE),

                "initial_graph_nu": 1.0,
                "initial_graph_kappa": 1.0,
                "initial_time_lengthscale": 1.0,
                "initial_time_outputscale": 1.0,
            },
        )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------
    y_mean, y_std = fit_standardization(train_series_raw, allowed_nodes)

    print("\nNormalization")
    print(f"Train normalization mean = {y_mean:.6f}")
    print(f"Train normalization std  = {y_std:.6f}")
    print(f"Train task source        = {train_tasks_source}")

    # --------------------------------------------------------
    # Build fixed validation / test tasks from real raw days
    # --------------------------------------------------------
    val_tasks = build_spatiotemporal_tasks(
        series_data=val_series,
        node_to_pos=node_to_pos,
        allowed_nodes=allowed_nodes,
        y_mean=y_mean,
        y_std=y_std,
        obs_fraction=OBS_FRACTION,
        num_tasks_per_center=VAL_TASKS_PER_CENTER,
        base_seed=RANDOM_SEED + 10_000,
        fixed_observed_nodes=fixed_observed_nodes,
        temporal_window_steps=TEMPORAL_WINDOW_STEPS,
        center_time_stride=CENTER_TIME_STRIDE,
        min_available_nodes=MIN_AVAILABLE_NODES,
        max_neighbor_points=MAX_NEIGHBOR_POINTS,
    )

    test_tasks = build_spatiotemporal_tasks(
        series_data=test_series,
        node_to_pos=node_to_pos,
        allowed_nodes=allowed_nodes,
        y_mean=y_mean,
        y_std=y_std,
        obs_fraction=OBS_FRACTION,
        num_tasks_per_center=TEST_TASKS_PER_CENTER,
        base_seed=RANDOM_SEED + 20_000,
        fixed_observed_nodes=fixed_observed_nodes,
        temporal_window_steps=TEMPORAL_WINDOW_STEPS,
        center_time_stride=CENTER_TIME_STRIDE,
        min_available_nodes=MIN_AVAILABLE_NODES,
        max_neighbor_points=MAX_NEIGHBOR_POINTS,
    )

    # --------------------------------------------------------
    # Build fixed train tasks once
    # --------------------------------------------------------
    train_tasks = build_spatiotemporal_tasks(
        series_data=train_series_for_tasks,
        node_to_pos=node_to_pos,
        allowed_nodes=allowed_nodes,
        y_mean=y_mean,
        y_std=y_std,
        obs_fraction=OBS_FRACTION,
        num_tasks_per_center=TRAIN_TASKS_PER_CENTER,
        base_seed=RANDOM_SEED,
        fixed_observed_nodes=fixed_observed_nodes,
        temporal_window_steps=TEMPORAL_WINDOW_STEPS,
        center_time_stride=CENTER_TIME_STRIDE,
        min_available_nodes=MIN_AVAILABLE_NODES,
        max_neighbor_points=MAX_NEIGHBOR_POINTS,
    )

    if len(train_tasks) > MAX_TRAIN_EVAL_TASKS:
        rng = np.random.default_rng(RANDOM_SEED + 123_456)
        chosen = rng.choice(
            len(train_tasks),
            size=MAX_TRAIN_EVAL_TASKS,
            replace=False,
        )
        train_eval_tasks = [train_tasks[i] for i in chosen]
    else:
        train_eval_tasks = train_tasks

    print(f"\nFixed train tasks: {len(train_tasks)}")
    print(f"Train eval tasks:  {len(train_eval_tasks)}")
    print(f"Val tasks:         {len(val_tasks)}")
    print(f"Test tasks:        {len(test_tasks)}")

    if USE_WANDB:
        wandb.config.update(
            {
                "num_graph_nodes": len(G.nodes()),
                "num_mapped_nodes": len(node_to_pos),
                "num_filtered_nodes": len(allowed_nodes),
                "filtered_subgraph_edges": filtered_subgraph.number_of_edges(),
                "y_mean": y_mean,
                "y_std": y_std,

                "use_fixed_observed_nodes": USE_FIXED_OBSERVED_NODES,
                "num_fixed_observed_nodes": len(fixed_observed_node_ids),
                "fixed_observed_node_ids": fixed_observed_node_ids,

                "num_train_series_raw": len(train_series_raw),
                "num_train_series_for_tasks": len(train_series_for_tasks),

                "resample_train_masks": RESAMPLE_TRAIN_MASKS,
                "num_train_tasks": len(train_tasks),
                "num_train_eval_tasks": len(train_eval_tasks),
                "num_val_tasks": len(val_tasks),
                "num_test_tasks": len(test_tasks),
                "max_train_eval_tasks": MAX_TRAIN_EVAL_TASKS,
            },
            allow_val_change=True,
        )

    if len(train_tasks) == 0:
        raise RuntimeError(
            "No training tasks were built. Check fixed observed nodes, local radius, "
            "historical mean series, MIN_AVAILABLE_NODES, and missingness."
        )

    if len(val_tasks) == 0 or len(test_tasks) == 0:
        raise RuntimeError(
            "Validation/test tasks are empty. Check fixed observed nodes, local radius, "
            "missingness, and MIN_AVAILABLE_NODES."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = SharedConditionalSpatioTemporalGraphModel(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        base_jitter=BASE_JITTER,
    ).to(DEVICE)

    model.kernel.graph_kernel.nu = 1.0
    model.kernel.graph_kernel.kappa = 1.0

    model.kernel.time_kernel.base_kernel.lengthscale = 1.0
    model.kernel.time_kernel.outputscale = 1.0

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    epoch_bar = tqdm(range(1, NUM_EPOCHS + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()

        running_loss = 0.0
        seen_tasks = 0

        train_batches = iter_task_batches(train_tasks, TRAIN_BATCH_SIZE)
        total_train_batches = num_batches(len(train_tasks), TRAIN_BATCH_SIZE)

        train_batch_bar = tqdm(
            train_batches,
            total=total_train_batches,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]",
            unit="batch",
            leave=False,
        )

        for batch in train_batch_bar:
            optimizer.zero_grad(set_to_none=True)

            batch_loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

            for task in batch:
                dev_task = task_to_device(task, DEVICE)

                task_loss = model.task_nll(
                    x_obs=dev_task["x_obs"],
                    y_obs=dev_task["y_obs"],
                    x_tgt=dev_task["x_tgt"],
                    y_tgt=dev_task["y_tgt"],
                    normalize_by_targets=True,
                )
                batch_loss = batch_loss + task_loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()

            running_loss += float(batch_loss.detach().cpu()) * len(batch)
            seen_tasks += len(batch)

            train_batch_bar.set_postfix(
                avg_nll=f"{running_loss / max(seen_tasks, 1):.4f}"
            )

        avg_train_loss = running_loss / max(seen_tasks, 1)

        train_eval_df, _ = evaluate_tasks(
            model=model,
            tasks=train_eval_tasks,
            y_mean=y_mean,
            y_std=y_std,
            batch_size=EVAL_BATCH_SIZE,
            split_name=f"Epoch {epoch}/{NUM_EPOCHS} [train eval]",
            show_progress=True,
        )

        train_eval_metrics = compute_metrics(train_eval_df, scale=y_std)

        val_df, _ = evaluate_tasks(
            model=model,
            tasks=val_tasks,
            y_mean=y_mean,
            y_std=y_std,
            batch_size=EVAL_BATCH_SIZE,
            split_name=f"Epoch {epoch}/{NUM_EPOCHS} [val]",
            show_progress=True,
        )

        val_metrics = compute_metrics(val_df, scale=y_std)
        val_rmse = val_metrics["rmse"]

        history.append({
            "epoch": epoch,

            "train_nll": float(avg_train_loss),
            "train_mae": float(train_eval_metrics["mae"]),
            "train_rmse": float(train_eval_metrics["rmse"]),
            "train_nmae": float(train_eval_metrics["nmae"]),
            "train_nrmse": float(train_eval_metrics["nrmse"]),

            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_nmae": float(val_metrics["nmae"]),
            "val_nrmse": float(val_metrics["nrmse"]),

            "graph_nu": float(model.kernel.graph_kernel.nu.item()),
            "graph_kappa": float(model.kernel.graph_kernel.kappa.item()),
            "time_lengthscale": float(model.kernel.time_kernel.base_kernel.lengthscale.item()),
            "time_outputscale": float(model.kernel.time_kernel.outputscale.item()),
            "mean": float(model.mean.item()),
            "noise": float(model.noise.item()),

            "use_central_node_radius_filter": bool(USE_CENTRAL_NODE_RADIUS_FILTER),
            "central_node_used": int(central_node) if central_node is not None else None,
            "graph_radius_edges": int(GRAPH_RADIUS_EDGES),

            "use_fixed_observed_nodes": bool(USE_FIXED_OBSERVED_NODES),
            "num_fixed_observed_nodes": int(len(fixed_observed_node_ids)),

            "use_historical_mean_training": bool(USE_HISTORICAL_MEAN_TRAINING),
            "train_tasks_source": train_tasks_source,

            "resample_train_masks": bool(RESAMPLE_TRAIN_MASKS),
            "num_train_tasks": int(len(train_tasks)),
            "num_train_eval_tasks": int(len(train_eval_tasks)),
            "num_val_tasks": int(len(val_tasks)),
            "num_filtered_nodes": int(len(allowed_nodes)),
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model_state": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "val_metrics": dict(val_metrics),
            }

        if USE_WANDB:
            wandb.log(
                {
                    "epoch": epoch,

                    "train/nll": float(avg_train_loss),

                    "train_eval/mae": float(train_eval_metrics["mae"]),
                    "train_eval/rmse": float(train_eval_metrics["rmse"]),
                    "train_eval/nmae": float(train_eval_metrics["nmae"]),
                    "train_eval/nrmse": float(train_eval_metrics["nrmse"]),

                    "val/mae": float(val_metrics["mae"]),
                    "val/rmse": float(val_metrics["rmse"]),
                    "val/nmae": float(val_metrics["nmae"]),
                    "val/nrmse": float(val_metrics["nrmse"]),

                    "kernel/graph_nu": float(model.kernel.graph_kernel.nu.item()),
                    "kernel/graph_kappa": float(model.kernel.graph_kernel.kappa.item()),
                    "kernel/time_lengthscale": float(
                        model.kernel.time_kernel.base_kernel.lengthscale.item()
                    ),
                    "kernel/time_outputscale": float(
                        model.kernel.time_kernel.outputscale.item()
                    ),

                    "model/mean": float(model.mean.item()),
                    "model/noise": float(model.noise.item()),

                    "graph/use_central_node_radius_filter": bool(USE_CENTRAL_NODE_RADIUS_FILTER),
                    "graph/central_node_used": int(central_node) if central_node is not None else -1,
                    "graph/radius_edges": int(GRAPH_RADIUS_EDGES),

                    "obs/use_fixed_observed_nodes": bool(USE_FIXED_OBSERVED_NODES),
                    "obs/num_fixed_observed_nodes": int(len(fixed_observed_node_ids)),

                    "training/use_historical_mean_training": bool(USE_HISTORICAL_MEAN_TRAINING),
                    "training/train_tasks_source": train_tasks_source,

                    "tasks/resample_train_masks": bool(RESAMPLE_TRAIN_MASKS),
                    "tasks/num_train_tasks": int(len(train_tasks)),
                    "tasks/num_train_eval_tasks": int(len(train_eval_tasks)),
                    "tasks/num_val_tasks": int(len(val_tasks)),
                    "tasks/num_filtered_nodes": int(len(allowed_nodes)),

                    "best/val_rmse": float(best_val_rmse),
                    "best/epoch": int(best_epoch),
                },
                step=epoch,
            )

        epoch_bar.set_postfix(
            train_nll=f"{avg_train_loss:.4f}",
            val_nmae=f"{val_metrics['nmae']:.4f}",
            val_nrmse=f"{val_metrics['nrmse']:.4f}",
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train NLL: {avg_train_loss:.4f} | "
            f"Train MAE: {train_eval_metrics['mae']:.4f} | "
            f"Train RMSE: {train_eval_metrics['rmse']:.4f} | "
            f"Train NMAE: {train_eval_metrics['nmae']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val NMAE: {val_metrics['nmae']:.4f} | "
            f"Val NRMSE: {val_metrics['nrmse']:.4f} | "
            f"graph_nu: {model.kernel.graph_kernel.nu.item():.4f} | "
            f"graph_kappa: {model.kernel.graph_kernel.kappa.item():.4f} | "
            f"time_lengthscale: {model.kernel.time_kernel.base_kernel.lengthscale.item():.4f} | "
            f"time_outputscale: {model.kernel.time_kernel.outputscale.item():.4f} | "
            f"mean: {model.mean.item():.4f} | "
            f"noise: {model.noise.item():.4f}"
        )

    if best_state is None:
        raise RuntimeError("No best checkpoint was saved.")

    print(f"\nRestoring best model from epoch {best_state['epoch']}")
    print("Best validation metrics:", best_state["val_metrics"])

    model.load_state_dict(best_state["model_state"])

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------
    val_df, val_cov_records = evaluate_tasks(
        model=model,
        tasks=val_tasks,
        y_mean=y_mean,
        y_std=y_std,
        batch_size=EVAL_BATCH_SIZE,
        split_name="Final validation",
        show_progress=True,
    )

    test_df, test_cov_records = evaluate_tasks(
        model=model,
        tasks=test_tasks,
        y_mean=y_mean,
        y_std=y_std,
        batch_size=EVAL_BATCH_SIZE,
        split_name="Final test",
        show_progress=True,
    )

    val_metrics = compute_metrics(val_df, scale=y_std)
    test_metrics = compute_metrics(test_df, scale=y_std)

    if USE_WANDB:
        wandb.log(
            {
                "final/val_mae": float(val_metrics["mae"]),
                "final/val_rmse": float(val_metrics["rmse"]),
                "final/val_nmae": float(val_metrics["nmae"]),
                "final/val_nrmse": float(val_metrics["nrmse"]),

                "final/test_mae": float(test_metrics["mae"]),
                "final/test_rmse": float(test_metrics["rmse"]),
                "final/test_nmae": float(test_metrics["nmae"]),
                "final/test_nrmse": float(test_metrics["nrmse"]),

                "final/best_epoch": int(best_epoch),
                "final/best_val_rmse": float(best_val_rmse),

                "final/use_fixed_observed_nodes": bool(USE_FIXED_OBSERVED_NODES),
                "final/num_fixed_observed_nodes": int(len(fixed_observed_node_ids)),

                "final/use_historical_mean_training": bool(USE_HISTORICAL_MEAN_TRAINING),
                "final/train_tasks_source": train_tasks_source,
            }
        )

        wandb.summary["best_epoch"] = int(best_epoch)
        wandb.summary["best_val_rmse"] = float(best_val_rmse)
        wandb.summary["test_nmae"] = float(test_metrics["nmae"])
        wandb.summary["test_nrmse"] = float(test_metrics["nrmse"])
        wandb.summary["use_fixed_observed_nodes"] = bool(USE_FIXED_OBSERVED_NODES)
        wandb.summary["num_fixed_observed_nodes"] = int(len(fixed_observed_node_ids))
        wandb.summary["use_historical_mean_training"] = bool(USE_HISTORICAL_MEAN_TRAINING)
        wandb.summary["train_tasks_source"] = train_tasks_source

    print("\nValidation")
    print(f"MAE:   {val_metrics['mae']:.4f}")
    print(f"RMSE:  {val_metrics['rmse']:.4f}")
    print(f"NMAE:  {val_metrics['nmae']:.4f}")
    print(f"NRMSE: {val_metrics['nrmse']:.4f}")

    print("\nTest")
    print(f"MAE:   {test_metrics['mae']:.4f}")
    print(f"RMSE:  {test_metrics['rmse']:.4f}")
    print(f"NMAE:  {test_metrics['nmae']:.4f}")
    print(f"NRMSE: {test_metrics['nrmse']:.4f}")

    if len(test_df) > 0:
        print("\nTest by target node (top 20 by count)")
        by_target = (
            test_df.groupby("target_node")
            .agg(
                count=("target_node", "size"),
                mae=("abs_error", "mean"),
                rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            )
            .sort_values("count", ascending=False)
            .head(20)
        )
        print(by_target)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    out_dir = Path(
        f"outputs_spatiotemporal_gp_central_{CENTRAL_NODE_ID}"
        f"_r_{GRAPH_RADIUS_EDGES}"
        f"_fixed_obs"
        f"_histmean_train"
        f"_no_resampling"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "training_history.csv", index=False)

    torch.save(best_state, out_dir / "best_checkpoint.pt")

    val_df.to_csv(out_dir / "val_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    torch.save(val_cov_records, out_dir / "val_task_covariances.pt")
    torch.save(test_cov_records, out_dir / "test_task_covariances.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),

            "use_central_node_radius_filter": USE_CENTRAL_NODE_RADIUS_FILTER,
            "central_node_used": central_node,
            "graph_radius_edges": GRAPH_RADIUS_EDGES,
            "local_min_graph_degree": LOCAL_MIN_GRAPH_DEGREE,

            "use_fixed_observed_nodes": USE_FIXED_OBSERVED_NODES,
            "fixed_observed_node_ids": fixed_observed_node_ids,

            "use_historical_mean_training": USE_HISTORICAL_MEAN_TRAINING,
            "train_tasks_source": train_tasks_source,

            "resample_train_masks": RESAMPLE_TRAIN_MASKS,

            "obs_fraction": OBS_FRACTION,
            "min_graph_degree": MIN_GRAPH_DEGREE,
            "temporal_window_steps": TEMPORAL_WINDOW_STEPS,
            "center_time_stride": CENTER_TIME_STRIDE,
            "max_neighbor_points": MAX_NEIGHBOR_POINTS,
            "train_tasks_per_center": TRAIN_TASKS_PER_CENTER,
            "val_tasks_per_center": VAL_TASKS_PER_CENTER,
            "test_tasks_per_center": TEST_TASKS_PER_CENTER,
            "min_available_nodes": MIN_AVAILABLE_NODES,
            "num_filtered_nodes": len(allowed_nodes),
            "y_mean": y_mean,
            "y_std": y_std,
            "allowed_nodes": sorted(int(n) for n in allowed_nodes),
            "node_to_pos": node_to_pos,
        },
        out_dir / "shared_model.pt",
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Saved outputs to: {out_dir.resolve()}")

    if USE_WANDB:
        artifact = wandb.Artifact(
            name=f"spatiotemporal-gp-{wandb.run.id}",
            type="model",
            metadata={
                "best_epoch": int(best_epoch),
                "best_val_rmse": float(best_val_rmse),
                "test_nmae": float(test_metrics["nmae"]),
                "test_nrmse": float(test_metrics["nrmse"]),

                "obs_fraction": OBS_FRACTION,
                "min_graph_degree": MIN_GRAPH_DEGREE,
                "min_available_nodes": MIN_AVAILABLE_NODES,

                "use_central_node_radius_filter": USE_CENTRAL_NODE_RADIUS_FILTER,
                "central_node_used": int(central_node) if central_node is not None else -1,
                "graph_radius_edges": GRAPH_RADIUS_EDGES,
                "local_min_graph_degree": LOCAL_MIN_GRAPH_DEGREE,

                "use_fixed_observed_nodes": USE_FIXED_OBSERVED_NODES,
                "num_fixed_observed_nodes": len(fixed_observed_node_ids),

                "use_historical_mean_training": USE_HISTORICAL_MEAN_TRAINING,
                "train_tasks_source": train_tasks_source,

                "temporal_window_steps": TEMPORAL_WINDOW_STEPS,
                "center_time_stride": CENTER_TIME_STRIDE,
                "lr": LR,
                "resample_train_masks": RESAMPLE_TRAIN_MASKS,
            },
        )

        artifact.add_file(str(out_dir / "best_checkpoint.pt"))
        artifact.add_file(str(out_dir / "shared_model.pt"))
        artifact.add_file(str(out_dir / "training_history.csv"))
        artifact.add_file(str(out_dir / "val_predictions.csv"))
        artifact.add_file(str(out_dir / "test_predictions.csv"))

        wandb.log_artifact(artifact)

        wandb.finish()


if __name__ == "__main__":
    main()