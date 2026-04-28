from __future__ import annotations

import json
import pickle
import time
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

# Fraction of center-time available nodes revealed to the model
OBS_FRACTION = 0.3

# Only keep graph nodes whose degree in the canonical graph is >= this threshold
MIN_GRAPH_DEGREE = 3

# Minimum number of available nodes at the center timestamp to build a task
MIN_AVAILABLE_NODES = 50

# ------------------------------------------------------------
# Temporal task construction
# ------------------------------------------------------------
# For each center timestamp, use rows in [center - w, ..., center + w]
TEMPORAL_WINDOW_STEPS = 2

# Build one task every CENTER_TIME_STRIDE timestamps inside each file
CENTER_TIME_STRIDE = 5

# Cap the number of non-center temporal observations to avoid huge covariances
MAX_NEIGHBOR_POINTS = 2500

# Number of random masks per center timestamp
TRAIN_TASKS_PER_CENTER = 1
VAL_TASKS_PER_CENTER = 1
TEST_TASKS_PER_CENTER = 1

MAX_TRAIN_EVAL_TASKS = 300

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

NUM_EPOCHS = 15
LR = 0.003
WEIGHT_DECAY = 0.0
RANDOM_SEED = 42

# Numerical stabilization
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
WANDB_ENTITY = None  # Put your W&B username/team here if needed, else keep None

WANDB_RUN_NAME = (
    f"deg_ge_{MIN_GRAPH_DEGREE}"
    f"_obs_{OBS_FRACTION}"
    f"_tw_{TEMPORAL_WINDOW_STEPS}"
    f"_stride_{CENTER_TIME_STRIDE}"
    f"_lr_{LR}"
)



# ============================================================
# UTILS
# ============================================================
def materialize_kernel(K: torch.Tensor) -> torch.Tensor:
    """
    Convert a lazy covariance object into a dense tensor when needed.
    """
    if hasattr(K, "to_dense"):
        return K.to_dense()
    if hasattr(K, "evaluate"):
        return K.evaluate()
    return K


def symmetrize(M: torch.Tensor) -> torch.Tensor:
    """
    Force symmetry numerically.
    """
    return 0.5 * (M + M.transpose(-1, -2))


def stabilize_covariance(
    cov: torch.Tensor,
    base_jitter: float = BASE_JITTER,
    max_tries: int = MAX_JITTER_TRIES,
) -> torch.Tensor:
    """
    Add enough jitter so Cholesky succeeds.
    """
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
    """
    Compute raw and normalized regression metrics.
    scale is usually the train standard deviation y_std.
    """
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


def node_ids_to_positions(node_ids: list[int], node_to_pos: dict[int, int]) -> list[int]:
    """
    Convert graph node ids into spectral positions used by the eigenvector matrix.
    """
    return [node_to_pos[n] for n in node_ids]


def task_to_device(task: dict, device: torch.device) -> dict:
    """
    Convert one CPU task into device tensors.

    Spatiotemporal format:
      x[..., 0] = node position in spectral ordering
      x[..., 1] = time
    """
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
    """
    Convert a DatetimeIndex to float hours in [0, 24).
    """
    return (
        index.hour.to_numpy(dtype=np.float32)
        + index.minute.to_numpy(dtype=np.float32) / 60.0
        + index.second.to_numpy(dtype=np.float32) / 3600.0
    )


# ============================================================
# PICKLE / DATAFRAME LOADING
# ============================================================
def extract_pred_vtime_df(obj: Any) -> pd.DataFrame:
    """
    Adapt this if your pickle structure differs.

    Expected:
      - either the pickle itself is a DataFrame
      - or it is a dict containing a DataFrame under one of these keys
    """
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

    # Force datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Try to cast columns to integer node ids
    try:
        df.columns = pd.Index([int(c) for c in df.columns])
    except Exception:
        pass

    return df


def dataframe_to_sparse_rows(df: pd.DataFrame) -> dict:
    """
    Convert one dataframe into a sparse row representation to reduce overhead.

    Output:
      {
        "timestamps": [...],
        "time_hours": np.ndarray shape (T,),
        "rows": list[dict[int, float]] length T
      }
    """
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
    """
    Build a cached sparse time-series representation for each file.
    """
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


# ============================================================
# GRAPH + SPECTRAL OBJECTS
# ============================================================
def load_graph(graph_path: Path) -> nx.Graph:
    """
    Load the canonical graph from disk.
    """
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    if not isinstance(G, nx.Graph):
        raise TypeError(f"Loaded object is not a NetworkX graph, got: {type(G)}")

    return G


def _load_node_to_pos(spectrum_dir: Path) -> dict[int, int]:
    """
    Prefer node_to_index.json because it is explicit.
    Fall back to node_order.pt / node_order.npy if needed.
    """
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
    """
    Load a tensor saved either as .pt or .npy.
    """
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
    """
    Load:
      - graph topology from canonical_graph.gpickle
      - precomputed spectral objects from graph_spectrum
    """
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

    # Basic consistency checks
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
    """
    Keep only nodes whose degree in the canonical graph is >= min_degree
    and that are present in the spectral mapping.
    """
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


# ============================================================
# DATASET BUILDING
# ============================================================
def fit_standardization(
    train_series: list[dict],
    allowed_nodes: set[int],
) -> tuple[float, float]:
    """
    Fit normalization stats using only the allowed node subset.
    """
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
    """
    Randomly split available center-time nodes into:
      - observed center nodes
      - target center nodes
    """
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
    temporal_window_steps: int = TEMPORAL_WINDOW_STEPS,
    center_time_stride: int = CENTER_TIME_STRIDE,
    min_available_nodes: int = MIN_AVAILABLE_NODES,
    max_neighbor_points: int = MAX_NEIGHBOR_POINTS,
) -> list[dict]:
    """
    Build spatiotemporal reconstruction tasks.

    For each chosen center timestamp:
      - use only allowed graph nodes
      - choose a random observed subset among nodes available at the center time
      - predict all other available nodes at the center time
      - additionally provide observations from nearby timestamps inside the window

    Inputs are 2D points:
      [node_position, time_hour]
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

                obs_center_nodes, tgt_nodes = sample_obs_target_split(
                    available_nodes=center_nodes,
                    obs_fraction=obs_fraction,
                    rng=rng,
                )

                if len(obs_center_nodes) == 0 or len(tgt_nodes) == 0:
                    continue

                obs_points = []

                # Center-time observed points
                for n in obs_center_nodes:
                    obs_points.append((int(n), center_time, float(center_row[n])))

                # Neighbor-time observed points
                neighbor_points = []
                for ridx in range(left, right):
                    if ridx == center_idx:
                        continue

                    row_values = rows[ridx]
                    tval = float(time_hours[ridx])

                    for n, v in row_values.items():
                        if n in usable_nodes:
                            neighbor_points.append((int(n), tval, float(v)))

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
    """
    Shared GraphTemporalKernel + shared constant mean + shared noise.

    Input format:
      x[..., 0] = node position in spectral ordering
      x[..., 1] = time
    """
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

        # Global mean for standardized process
        self.raw_mean = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Positive observation noise via softplus
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
        x_obs: torch.Tensor,   # (n_obs, 2)
        y_obs: torch.Tensor,   # (n_obs,)
        x_tgt: torch.Tensor,   # (n_tgt, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the predictive Gaussian distribution:

            y_tgt | y_obs ~ N(pred_mean, pred_cov)

        where both observed and target quantities are noisy observations.
        """
        noise = self.noise

        # Kernel blocks
        K_oo = self._cov(x_obs, x_obs)              # (n_obs, n_obs)
        K_ot = self._cov(x_obs, x_tgt)              # (n_obs, n_tgt)
        K_to = K_ot.transpose(-1, -2)               # (n_tgt, n_obs)
        K_tt = self._cov(x_tgt, x_tgt)              # (n_tgt, n_tgt)

        I_obs = torch.eye(K_oo.shape[-1], device=K_oo.device, dtype=K_oo.dtype)
        I_tgt = torch.eye(K_tt.shape[-1], device=K_tt.device, dtype=K_tt.dtype)

        K_oo_noisy = K_oo + noise * I_obs
        K_tt_noisy = K_tt + noise * I_tgt

        K_oo_noisy = stabilize_covariance(K_oo_noisy, base_jitter=self.base_jitter)
        K_tt_noisy = stabilize_covariance(K_tt_noisy, base_jitter=self.base_jitter)

        mean_obs = self.mean.expand(x_obs.shape[0])   # (n_obs,)
        mean_tgt = self.mean.expand(x_tgt.shape[0])   # (n_tgt,)

        resid = (y_obs - mean_obs).unsqueeze(-1)      # (n_obs, 1)

        alpha = torch.linalg.solve(K_oo_noisy, resid)                     # (n_obs, 1)
        pred_mean = mean_tgt.unsqueeze(-1) + K_to @ alpha                # (n_tgt, 1)
        pred_mean = pred_mean.squeeze(-1)                                # (n_tgt,)

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
        """
        Negative log-likelihood for one spatiotemporal reconstruction task.
        """
        pred_mean, pred_cov = self.conditional_distribution(x_obs, y_obs, x_tgt)
        dist = torch.distributions.MultivariateNormal(pred_mean, covariance_matrix=pred_cov)
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
    """
    Evaluate a list of tasks.

    Returns:
      - row-level dataframe with one row per predicted node
      - covariance records, one per task
    """
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
    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Graph objects
    # --------------------------------------------------------
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
    # Degree-based node filtering
    # --------------------------------------------------------
    allowed_nodes = get_degree_filtered_nodes(
        G=G,
        node_to_pos=node_to_pos,
        min_degree=MIN_GRAPH_DEGREE,
    )

    filtered_subgraph = G.subgraph(allowed_nodes).copy()

    print(f"\nApplying degree filter: degree >= {MIN_GRAPH_DEGREE}")
    print(f"Allowed nodes after filtering: {len(allowed_nodes)}")
    print(f"Filtered subgraph nodes      : {filtered_subgraph.number_of_nodes()}")
    print(f"Filtered subgraph edges      : {filtered_subgraph.number_of_edges()}")

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
    # Fit normalization on train only, using filtered nodes only
    # --------------------------------------------------------
    y_mean, y_std = fit_standardization(train_series, allowed_nodes)

    print(f"\nTrain normalization over degree>={MIN_GRAPH_DEGREE} nodes only")
    print(f"Train normalization mean = {y_mean:.6f}")
    print(f"Train normalization std  = {y_std:.6f}")

    # --------------------------------------------------------
    # Build fixed validation / test tasks
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
        temporal_window_steps=TEMPORAL_WINDOW_STEPS,
        center_time_stride=CENTER_TIME_STRIDE,
        min_available_nodes=MIN_AVAILABLE_NODES,
        max_neighbor_points=MAX_NEIGHBOR_POINTS,
    )

    # Warm-up build to verify train tasks exist
    warmup_train_tasks = build_spatiotemporal_tasks(
        series_data=train_series,
        node_to_pos=node_to_pos,
        allowed_nodes=allowed_nodes,
        y_mean=y_mean,
        y_std=y_std,
        obs_fraction=OBS_FRACTION,
        num_tasks_per_center=TRAIN_TASKS_PER_CENTER,
        base_seed=RANDOM_SEED,
        temporal_window_steps=TEMPORAL_WINDOW_STEPS,
        center_time_stride=CENTER_TIME_STRIDE,
        min_available_nodes=MIN_AVAILABLE_NODES,
        max_neighbor_points=MAX_NEIGHBOR_POINTS,
    )

    # Fixed train subset used only for monitoring training prediction metrics.
    # This stays the same across epochs, unlike the resampled training masks.
    if len(warmup_train_tasks) > MAX_TRAIN_EVAL_TASKS:
        rng = np.random.default_rng(RANDOM_SEED + 123_456)
        chosen = rng.choice(
            len(warmup_train_tasks),
            size=MAX_TRAIN_EVAL_TASKS,
            replace=False,
        )
        train_eval_tasks = [warmup_train_tasks[i] for i in chosen]
    else:
        train_eval_tasks = warmup_train_tasks

    print(f"\nWarm-up train tasks: {len(warmup_train_tasks)}")
    print(f"Train eval tasks:    {len(train_eval_tasks)}")
    print(f"Val tasks:           {len(val_tasks)}")
    print(f"Test tasks:          {len(test_tasks)}")

    if USE_WANDB:
        wandb.config.update(
            {
                "num_graph_nodes": len(G.nodes()),
                "num_mapped_nodes": len(node_to_pos),
                "num_filtered_nodes": len(allowed_nodes),
                "filtered_subgraph_edges": filtered_subgraph.number_of_edges(),
                "y_mean": y_mean,
                "y_std": y_std,
                "num_warmup_train_tasks": len(warmup_train_tasks),
                "num_train_eval_tasks": len(train_eval_tasks),
                "num_val_tasks": len(val_tasks),
                "num_test_tasks": len(test_tasks),
                "max_train_eval_tasks": MAX_TRAIN_EVAL_TASKS,
            },
            allow_val_change=True,
        )

    if len(warmup_train_tasks) == 0:
        raise RuntimeError(
            "No training tasks were built for spatiotemporal reconstruction. Check:\n"
            "  - degree filter maybe too strict\n"
            "  - node id alignment\n"
            "  - missingness in the rows\n"
            "  - MIN_AVAILABLE_NODES\n"
            "  - OBS_FRACTION\n"
            "  - CENTER_TIME_STRIDE / TEMPORAL_WINDOW_STEPS"
        )

    if len(val_tasks) == 0 or len(test_tasks) == 0:
        raise RuntimeError(
            "Validation/test tasks are empty. Check degree filter, missingness, and MIN_AVAILABLE_NODES."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = SharedConditionalSpatioTemporalGraphModel(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        base_jitter=BASE_JITTER,
    ).to(DEVICE)

    # Graph kernel hyperparameters
    model.kernel.graph_kernel.nu = 1.0
    model.kernel.graph_kernel.kappa = 1.0

    # Time kernel hyperparameters
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

        # Resample training masks every epoch
        train_tasks = build_spatiotemporal_tasks(
            series_data=train_series,
            node_to_pos=node_to_pos,
            allowed_nodes=allowed_nodes,
            y_mean=y_mean,
            y_std=y_std,
            obs_fraction=OBS_FRACTION,
            num_tasks_per_center=TRAIN_TASKS_PER_CENTER,
            base_seed=RANDOM_SEED + 100_000 * epoch,
            temporal_window_steps=TEMPORAL_WINDOW_STEPS,
            center_time_stride=CENTER_TIME_STRIDE,
            min_available_nodes=MIN_AVAILABLE_NODES,
            max_neighbor_points=MAX_NEIGHBOR_POINTS,
        )

        if len(train_tasks) == 0:
            raise RuntimeError(f"No training tasks built at epoch {epoch}.")

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

            train_batch_bar.set_postfix(avg_nll=f"{running_loss / max(seen_tasks, 1):.4f}")

        avg_train_loss = running_loss / max(seen_tasks, 1)

        # -------------------------
        # Train evaluation
        # -------------------------
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

        # -------------------------
        # Validation
        # -------------------------
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

            "num_train_tasks": int(len(train_tasks)),
            "num_train_eval_tasks": int(len(train_eval_tasks)),
            "num_val_tasks": int(len(val_tasks)),
            "num_filtered_nodes": int(len(allowed_nodes)),
        })

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

                    "tasks/num_train_tasks": int(len(train_tasks)),
                    "tasks/num_train_eval_tasks": int(len(train_eval_tasks)),
                    "tasks/num_val_tasks": int(len(val_tasks)),

                    "best/val_rmse": float(best_val_rmse),
                    "best/epoch": int(best_epoch),
                },
                step=epoch,
            )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "val_metrics": dict(val_metrics),
            }

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
            }
        )

        wandb.summary["best_epoch"] = int(best_epoch)
        wandb.summary["best_val_rmse"] = float(best_val_rmse)
        wandb.summary["test_nmae"] = float(test_metrics["nmae"])
        wandb.summary["test_nrmse"] = float(test_metrics["nrmse"])

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
    out_dir = Path(f"outputs_spatiotemporal_gp_deg_ge_{MIN_GRAPH_DEGREE}_matern_0")
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
                "temporal_window_steps": TEMPORAL_WINDOW_STEPS,
                "center_time_stride": CENTER_TIME_STRIDE,
                "lr": LR,
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