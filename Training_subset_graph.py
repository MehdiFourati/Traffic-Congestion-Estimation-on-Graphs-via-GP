import json
import pickle
import time
from math import ceil
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from Kernels.graph_matern import GraphMaternKernel


# ============================================================
# CONFIG
# ============================================================

GRAPH_SPECTRUM_DIR = Path(
    r"C:\Users\USER\Documents\GitHub\Traffic-Congestion-Estimation-on-Graphs-via-GP\outputs\graph_spectrum"
)

GRAPH_PATH = GRAPH_SPECTRUM_DIR / "canonical_graph.gpickle"
PICKLE_DIR = Path(r"C:\Users\USER\Documents\GitHub\datasets\simbarca\all_agg_trimmed")

TARGET_TIME = "09:00"

TRAIN_FILES = 70
VAL_FILES = 15
TEST_FILES = 16

OBS_FRACTION = 0.3

# Only keep graph nodes whose degree in the canonical graph is >= this threshold
MIN_GRAPH_DEGREE = 3

# For training, masks are resampled every epoch.
# For validation/test, masks stay fixed.
TRAIN_MASKS_PER_DAY = 1
VAL_MASKS_PER_DAY = 1
TEST_MASKS_PER_DAY = 1

# Skip rows that do not contain enough available graph nodes
MIN_AVAILABLE_NODES = 50

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

NUM_EPOCHS = 40
LR = 0.03
WEIGHT_DECAY = 0.0
RANDOM_SEED = 42

# Numerical stabilization
BASE_JITTER = 1e-5
MAX_JITTER_TRIES = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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

    # Final try; if it still fails later, the matrix is genuinely problematic.
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
    """
    Yield consecutive batches of tasks.
    """
    for start in range(0, len(tasks), batch_size):
        yield tasks[start:start + batch_size]


def node_ids_to_positions(node_ids: list[int], node_to_pos: dict[int, int]) -> list[int]:
    """
    Convert graph node ids into positions used by the spectral objects.
    """
    return [node_to_pos[n] for n in node_ids]


def task_to_device(task: dict, device: torch.device) -> dict:
    """
    Convert one CPU task into device tensors on demand.

    For the whole-graph reconstruction task:
      - x_obs has shape (n_obs, 1)
      - y_obs has shape (n_obs,)
      - x_tgt has shape (n_tgt, 1)
      - y_tgt has shape (n_tgt,)
    """
    x_obs = torch.tensor(task["x_obs_pos"], dtype=torch.float32, device=device).unsqueeze(-1)
    y_obs = torch.tensor(task["y_obs_std"], dtype=torch.float32, device=device)

    x_tgt = torch.tensor(task["x_tgt_pos"], dtype=torch.float32, device=device).unsqueeze(-1)
    y_tgt = torch.tensor(task["y_tgt_std"], dtype=torch.float32, device=device)

    return {
        "x_obs": x_obs,
        "y_obs": y_obs,
        "x_tgt": x_tgt,
        "y_tgt": y_tgt,
    }


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


def select_row_nearest_time(df: pd.DataFrame, target_time: str) -> pd.Series:
    """
    Pick the row whose timestamp is nearest to the requested clock time.
    """
    if len(df) == 0:
        raise ValueError("Empty dataframe.")

    hour, minute = map(int, target_time.split(":"))
    base_day = df.index[0].normalize()
    desired_ts = base_day + pd.Timedelta(hours=hour, minutes=minute)

    idx = np.argmin(np.abs(df.index - desired_ts))
    return df.iloc[idx]


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
def build_day_rows(
    file_list: list[Path],
    target_time: str,
    cache_path: Path | None = None,
) -> list[dict]:
    """
    Build one selected row per file/day at the chosen fixed clock time.

    If cache_path exists, load the cached rows instead of re-reading all pickles.
    """
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached day rows from: {cache_path}")
        with open(cache_path, "rb") as f:
            rows = pickle.load(f)
        print(f"Loaded {len(rows)} cached rows.")
        return rows

    rows = []

    for i, fp in enumerate(file_list, start=1):
        t0 = time.time()
        print(f"[{i:03d}/{len(file_list):03d}] Loading {fp.name} ...", flush=True)

        df = load_pred_vtime_df(fp)

        t1 = time.time()
        print(
            f"    loaded dataframe in {t1 - t0:.2f}s | shape={df.shape}",
            flush=True,
        )

        row = select_row_nearest_time(df, target_time)

        values = {}
        for col, val in row.items():
            if pd.notna(val):
                try:
                    values[int(col)] = float(val)
                except Exception:
                    pass

        rows.append({
            "file": fp.name,
            "timestamp": row.name,
            "values": values,
        })

        print(
            f"    extracted fixed-time row in {time.time() - t1:.2f}s | "
            f"non-missing nodes={len(values)}",
            flush=True,
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(rows, f)
        print(f"Saved cached day rows to: {cache_path}")

    return rows


def fit_standardization(
    train_rows: list[dict],
    allowed_nodes: set[int],
) -> tuple[float, float]:
    """
    Fit normalization stats using only the allowed node subset
    (here: degree >= MIN_GRAPH_DEGREE in the canonical graph).
    """
    vals = []

    for day in train_rows:
        for node_id, value in day["values"].items():
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
    Randomly split available nodes into:
      - observed nodes
      - target nodes
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


def build_whole_graph_tasks(
    day_rows: list[dict],
    node_to_pos: dict[int, int],
    allowed_nodes: set[int],
    y_mean: float,
    y_std: float,
    obs_fraction: float,
    num_masks_per_day: int,
    base_seed: int,
    min_available_nodes: int = MIN_AVAILABLE_NODES,
) -> list[dict]:
    """
    Build whole-graph reconstruction tasks on the filtered node subset.

    For each selected day/time row:
      - keep only nodes that:
          1) appear in this row,
          2) exist in the graph spectral mapping,
          3) satisfy the degree filter
      - choose a random observed subset of those nodes
      - predict all remaining filtered nodes

    One day can produce several tasks via different masks.
    """
    tasks = []
    mapped_nodes = set(node_to_pos.keys())
    usable_nodes = mapped_nodes & allowed_nodes

    for day_idx, day in enumerate(day_rows):
        values = day["values"]

        available_nodes = sorted([n for n in values if n in usable_nodes])

        if len(available_nodes) < min_available_nodes:
            continue

        for mask_idx in range(num_masks_per_day):
            rng = np.random.default_rng(base_seed + 10000 * day_idx + mask_idx)

            obs_nodes, tgt_nodes = sample_obs_target_split(
                available_nodes=available_nodes,
                obs_fraction=obs_fraction,
                rng=rng,
            )

            if len(obs_nodes) == 0 or len(tgt_nodes) == 0:
                continue

            x_obs_pos = node_ids_to_positions(obs_nodes, node_to_pos)
            x_tgt_pos = node_ids_to_positions(tgt_nodes, node_to_pos)

            y_obs_raw = np.array([values[n] for n in obs_nodes], dtype=np.float32)
            y_tgt_raw = np.array([values[n] for n in tgt_nodes], dtype=np.float32)

            y_obs_std = ((y_obs_raw - y_mean) / y_std).astype(np.float32)
            y_tgt_std = ((y_tgt_raw - y_mean) / y_std).astype(np.float32)

            tasks.append({
                "file": day["file"],
                "timestamp": day["timestamp"],
                "mask_id": int(mask_idx),

                "observed_nodes": [int(n) for n in obs_nodes],
                "target_nodes": [int(n) for n in tgt_nodes],
                "num_obs": len(obs_nodes),
                "num_targets": len(tgt_nodes),

                "x_obs_pos": [int(p) for p in x_obs_pos],
                "x_tgt_pos": [int(p) for p in x_tgt_pos],
                "y_obs_std": y_obs_std,
                "y_tgt_std": y_tgt_std,

                "y_obs_raw": y_obs_raw,
                "y_tgt_raw": y_tgt_raw,
            })

    return tasks


# ============================================================
# SHARED CONDITIONAL GP MODEL
# ============================================================
class SharedConditionalGraphModel(torch.nn.Module):
    """
    Shared graph Matérn kernel + shared constant mean + shared noise.

    For each task:
      - observe a subset of graph nodes
      - predict the remaining nodes

    We optimize the conditional Gaussian negative log-likelihood:
      -log p(y_target | y_observed)
    """
    def __init__(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        base_jitter: float = BASE_JITTER,
    ):
        super().__init__()

        self.kernel = GraphMaternKernel(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

        # Global mean for the standardized process
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
        return materialize_kernel(self.kernel(x1, x2))

    def conditional_distribution(
        self,
        x_obs: torch.Tensor,   # shape (n_obs, 1)
        y_obs: torch.Tensor,   # shape (n_obs,)
        x_tgt: torch.Tensor,   # shape (n_tgt, 1)
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
        Negative log-likelihood for one whole-graph reconstruction task.
        """
        pred_mean, pred_cov = self.conditional_distribution(x_obs, y_obs, x_tgt)
        dist = torch.distributions.MultivariateNormal(pred_mean, covariance_matrix=pred_cov)
        nll = -dist.log_prob(y_tgt)

        # This keeps the scale of the loss more comparable across tasks,
        # since target size can vary.
        if normalize_by_targets:
            nll = nll / max(int(y_tgt.numel()), 1)

        return nll


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_tasks(
    model: SharedConditionalGraphModel,
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
                "mask_id": int(task["mask_id"]),
                "observed_nodes": [int(n) for n in task["observed_nodes"]],
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
                    "mask_id": int(task["mask_id"]),
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
    # Build one selected row per file/day at the target time
    # --------------------------------------------------------
    cache_dir = Path("cache_fixed_time_rows")
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_time = TARGET_TIME.replace(":", "_")

    train_cache = cache_dir / f"train_rows_{safe_time}.pkl"
    val_cache = cache_dir / f"val_rows_{safe_time}.pkl"
    test_cache = cache_dir / f"test_rows_{safe_time}.pkl"

    train_rows = build_day_rows(train_files, TARGET_TIME, cache_path=train_cache)
    val_rows = build_day_rows(val_files, TARGET_TIME, cache_path=val_cache)
    test_rows = build_day_rows(test_files, TARGET_TIME, cache_path=test_cache)

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows:   {len(val_rows)}")
    print(f"Test rows:  {len(test_rows)}")

    # --------------------------------------------------------
    # Fit normalization on train only, using filtered nodes only
    # --------------------------------------------------------
    y_mean, y_std = fit_standardization(train_rows, allowed_nodes)

    print(f"\nTrain normalization over degree>={MIN_GRAPH_DEGREE} nodes only")
    print(f"Train normalization mean = {y_mean:.6f}")
    print(f"Train normalization std  = {y_std:.6f}")

    # --------------------------------------------------------
    # Build fixed validation / test tasks on the filtered node set
    # --------------------------------------------------------
    val_tasks = build_whole_graph_tasks(
        day_rows=val_rows,
        node_to_pos=node_to_pos,
        allowed_nodes=allowed_nodes,
        y_mean=y_mean,
        y_std=y_std,
        obs_fraction=OBS_FRACTION,
        num_masks_per_day=VAL_MASKS_PER_DAY,
        base_seed=RANDOM_SEED + 10_000,
        min_available_nodes=MIN_AVAILABLE_NODES,
    )

    test_tasks = build_whole_graph_tasks(
        day_rows=test_rows,
        node_to_pos=node_to_pos,
        allowed_nodes=allowed_nodes,
        y_mean=y_mean,
        y_std=y_std,
        obs_fraction=OBS_FRACTION,
        num_masks_per_day=TEST_MASKS_PER_DAY,
        base_seed=RANDOM_SEED + 20_000,
        min_available_nodes=MIN_AVAILABLE_NODES,
    )

    # Warm-up build to verify train tasks exist
    warmup_train_tasks = build_whole_graph_tasks(
        day_rows=train_rows,
        node_to_pos=node_to_pos,
        allowed_nodes=allowed_nodes,
        y_mean=y_mean,
        y_std=y_std,
        obs_fraction=OBS_FRACTION,
        num_masks_per_day=TRAIN_MASKS_PER_DAY,
        base_seed=RANDOM_SEED,
        min_available_nodes=MIN_AVAILABLE_NODES,
    )

    print(f"\nWarm-up train tasks: {len(warmup_train_tasks)}")
    print(f"Val tasks:           {len(val_tasks)}")
    print(f"Test tasks:          {len(test_tasks)}")

    if len(warmup_train_tasks) == 0:
        raise RuntimeError(
            "No training tasks were built for whole-graph reconstruction. Check:\n"
            "  - degree filter maybe too strict\n"
            "  - node id alignment\n"
            "  - missingness in the selected time row\n"
            "  - MIN_AVAILABLE_NODES\n"
            "  - OBS_FRACTION"
        )

    if len(val_tasks) == 0 or len(test_tasks) == 0:
        raise RuntimeError(
            "Validation/test tasks are empty. Check degree filter, missingness, and MIN_AVAILABLE_NODES."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = SharedConditionalGraphModel(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        base_jitter=BASE_JITTER,
    ).to(DEVICE)

    # Optional initialization of kernel hyperparameters
    model.kernel.nu = 1.0
    model.kernel.kappa = 1.0
    model.kernel.outputscale = 1.0

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

        # Resample training masks every epoch on filtered nodes only
        train_tasks = build_whole_graph_tasks(
            day_rows=train_rows,
            node_to_pos=node_to_pos,
            allowed_nodes=allowed_nodes,
            y_mean=y_mean,
            y_std=y_std,
            obs_fraction=OBS_FRACTION,
            num_masks_per_day=TRAIN_MASKS_PER_DAY,
            base_seed=RANDOM_SEED + 100_000 * epoch,
            min_available_nodes=MIN_AVAILABLE_NODES,
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
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_nmae": float(val_metrics["nmae"]),
            "val_nrmse": float(val_metrics["nrmse"]),
            "nu": float(model.kernel.nu.item()),
            "kappa": float(model.kernel.kappa.item()),
            "outputscale": float(model.kernel.outputscale.item()),
            "mean": float(model.mean.item()),
            "noise": float(model.noise.item()),
            "num_train_tasks": int(len(train_tasks)),
            "num_val_tasks": int(len(val_tasks)),
            "num_filtered_nodes": int(len(allowed_nodes)),
        })

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
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val NMAE: {val_metrics['nmae']:.4f} | "
            f"Val NRMSE: {val_metrics['nrmse']:.4f} | "
            f"nu: {model.kernel.nu.item():.4f} | "
            f"kappa: {model.kernel.kappa.item():.4f} | "
            f"outputscale: {model.kernel.outputscale.item():.4f} | "
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
    out_dir = Path(f"outputs_whole_graph_fixed_time_deg_ge_{MIN_GRAPH_DEGREE}")
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
            "target_time": TARGET_TIME,
            "obs_fraction": OBS_FRACTION,
            "min_graph_degree": MIN_GRAPH_DEGREE,
            "train_masks_per_day": TRAIN_MASKS_PER_DAY,
            "val_masks_per_day": VAL_MASKS_PER_DAY,
            "test_masks_per_day": TEST_MASKS_PER_DAY,
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


if __name__ == "__main__":
    main()