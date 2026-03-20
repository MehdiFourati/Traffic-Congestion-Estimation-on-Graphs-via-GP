from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import time
import json
import networkx as nx
import numpy as np
import pandas as pd
import torch
import gpytorch

# ------------------------------------------------------------------
# ADAPT THIS IMPORT TO YOUR PROJECT
# ------------------------------------------------------------------
from Kernels.graph_matern import GraphMaternKernel


# ============================================================
# CONFIG
# ============================================================
from pathlib import Path

# Folder that contains canonical_graph.gpickle, eigenvalues.pt, etc.
GRAPH_SPECTRUM_DIR = Path(
    r"C:\Users\USER\Documents\GitHub\Traffic-Congestion-Estimation-on-Graphs-via-GP\outputs\graph_spectrum"
)

GRAPH_PATH = GRAPH_SPECTRUM_DIR / "canonical_graph.gpickle"
PICKLE_DIR = Path(r"C:\Users\USER\Documents\GitHub\datasets\simbarca\all_agg_trimmed")

TARGET_TIME = "21:00"
MIN_DEGREE = 4

TRAIN_FILES = 70
VAL_FILES = 15
TEST_FILES = 16

NUM_EPOCHS = 2
LR = 0.03
WEIGHT_DECAY = 0.0
RANDOM_SEED = 42

MIN_NEIGHBORS_PRESENT = 1

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
    Convert a GPyTorch lazy covariance object into a dense tensor
    when we need manual linear algebra.
    """
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
    """
    Make covariance numerically safer for Cholesky / MVN operations.
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

    # Final try; let it fail later if truly broken
    return cov + jitter * I


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {"mae": float("nan"), "rmse": float("nan")}

    mae = float(df["abs_error"].mean())
    rmse = float(np.sqrt(df["sq_error"].mean()))
    return {"mae": mae, "rmse": rmse}


def node_ids_to_positions(node_ids: list[int], node_to_pos: dict[int, int]) -> list[int]:
    return [node_to_pos[n] for n in node_ids]


# ============================================================
# PICKLE / DATAFRAME LOADING
# ============================================================
def extract_pred_vtime_df(obj: Any) -> pd.DataFrame:
    """
    ADAPT THIS if your pickle structure differs.

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
    Pick the row whose timestamp is nearest to the requested clock time
    within a file/day.
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

        # JSON keys may be strings
        node_to_pos = {int(k): int(v) for k, v in raw.items()}
        return node_to_pos

    # Fallback: reconstruct mapping from node_order
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
      - the graph topology from canonical_graph.gpickle
      - the precomputed spectral objects from graph_spectrum

    This avoids recomputing eigenvalues/eigenvectors.
    """
    G = load_graph(graph_path)

    # Prefer .pt, fallback to .npy
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

    eligible_centers = sorted([int(n) for n in G.nodes() if G.degree[n] >= MIN_DEGREE])

    return G, node_to_pos, eigenvalues, eigenvectors, eligible_centers


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
            f"    loaded dataframe in {t1 - t0:.2f}s | "
            f"shape={df.shape}",
            flush=True
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
            flush=True
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(rows, f)
        print(f"Saved cached day rows to: {cache_path}")

    return rows


def get_relevant_nodes(G: nx.Graph, eligible_centers: list[int]) -> set[int]:
    """
    Nodes whose values are relevant for this task family:
      - eligible centers
      - neighbors of eligible centers
    """
    relevant = set()

    for c in eligible_centers:
        relevant.add(c)
        for nbr in G.neighbors(c):
            relevant.add(nbr)

    return relevant


def fit_standardization(
    train_rows: list[dict],
    relevant_nodes: set[int],
) -> tuple[float, float]:
    """
    Fit standardization stats using only the training split.
    """
    vals = []

    for day in train_rows:
        for node_id, value in day["values"].items():
            if node_id in relevant_nodes:
                vals.append(value)

    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        raise RuntimeError("No values available to fit normalization on train split.")

    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-8:
        std = 1.0

    return mean, std


def build_tasks(
    day_rows: list[dict],
    G: nx.Graph,
    eligible_centers: list[int],
    node_to_pos: dict[int, int],
    y_mean: float,
    y_std: float,
) -> list[dict]:
    """
    For each (day, center), create one task:

      observed:
        - center node value only

      targets:
        - all observed neighbors of that center in that day-row

    This matches:
      "condition on the center node only, predict its neighbors"
    """
    tasks = []

    for day in day_rows:
        values = day["values"]

        for center in eligible_centers:
            if center not in values:
                continue

            observed_neighbors = sorted([nbr for nbr in G.neighbors(center) if nbr in values])

            if len(observed_neighbors) < MIN_NEIGHBORS_PRESENT:
                continue

            center_pos = node_to_pos[center]
            neighbor_positions = node_ids_to_positions(observed_neighbors, node_to_pos)

            y_center_raw = float(values[center])
            y_neighbors_raw = np.array([values[nbr] for nbr in observed_neighbors], dtype=float)

            y_center_std = (y_center_raw - y_mean) / y_std
            y_neighbors_std = (y_neighbors_raw - y_mean) / y_std

            tasks.append({
                "file": day["file"],
                "timestamp": day["timestamp"],
                "center_node": center,
                "target_neighbors": observed_neighbors,
                "num_targets": len(observed_neighbors),
                "x_obs": torch.tensor([[center_pos]], dtype=torch.float32, device=DEVICE),
                "y_obs": torch.tensor([y_center_std], dtype=torch.float32, device=DEVICE),
                "x_tgt": torch.tensor(neighbor_positions, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
                "y_tgt": torch.tensor(y_neighbors_std, dtype=torch.float32, device=DEVICE),
                "y_obs_raw": y_center_raw,
                "y_tgt_raw": y_neighbors_raw,
            })

    return tasks


# ============================================================
# SHARED CONDITIONAL GP MODEL
# ============================================================
class SharedConditionalGraphModel(torch.nn.Module):
    """
    Shared graph kernel + shared constant mean + shared observation noise.

    For each task:
      observe center node value
      predict neighbor values

    We optimize the conditional Gaussian negative log-likelihood:
      -log p(y_neighbors | y_center)
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

        # Positive noise via softplus
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
        x_obs: torch.Tensor,   # shape (1, 1)
        y_obs: torch.Tensor,   # shape (1,)
        x_tgt: torch.Tensor,   # shape (m, 1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the predictive Gaussian distribution:

            y_tgt | y_obs ~ N(pred_mean, pred_cov)

        where both observed and target quantities are noisy observations.
        """
        noise = self.noise

        # Kernel blocks
        K_oo = self._cov(x_obs, x_obs)        # (1,1)
        K_ot = self._cov(x_obs, x_tgt)        # (1,m)
        K_to = K_ot.transpose(-1, -2)         # (m,1)
        K_tt = self._cov(x_tgt, x_tgt)        # (m,m)

        # Add observation noise on observed and target sides
        I_obs = torch.eye(K_oo.shape[-1], device=K_oo.device, dtype=K_oo.dtype)
        I_tgt = torch.eye(K_tt.shape[-1], device=K_tt.device, dtype=K_tt.dtype)

        K_oo_noisy = K_oo + noise * I_obs
        K_tt_noisy = K_tt + noise * I_tgt

        K_oo_noisy = stabilize_covariance(K_oo_noisy, base_jitter=self.base_jitter)
        K_tt_noisy = stabilize_covariance(K_tt_noisy, base_jitter=self.base_jitter)

        mean_obs = self.mean.expand(x_obs.shape[0])   # (1,)
        mean_tgt = self.mean.expand(x_tgt.shape[0])   # (m,)

        resid = (y_obs - mean_obs).unsqueeze(-1)      # (1,1)

        alpha = torch.linalg.solve(K_oo_noisy, resid)     # (1,1)
        pred_mean = mean_tgt.unsqueeze(-1) + K_to @ alpha
        pred_mean = pred_mean.squeeze(-1)                 # (m,)

        pred_cov = K_tt_noisy - K_to @ torch.linalg.solve(K_oo_noisy, K_ot)
        pred_cov = stabilize_covariance(pred_cov, base_jitter=self.base_jitter)

        return pred_mean, pred_cov

    def task_nll(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
        x_tgt: torch.Tensor,
        y_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log-likelihood for one (day, center) task.
        """
        pred_mean, pred_cov = self.conditional_distribution(x_obs, y_obs, x_tgt)
        dist = torch.distributions.MultivariateNormal(pred_mean, covariance_matrix=pred_cov)
        return -dist.log_prob(y_tgt)


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_tasks(
    model: SharedConditionalGraphModel,
    tasks: list[dict],
    y_mean: float,
    y_std: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns:
      - a row-level dataframe with one row per predicted neighbor
      - a list of full predictive covariance records, one per task
    """
    model.eval()

    rows = []
    covariance_records = []

    for task in tasks:
        pred_mean_std, pred_cov_std = model.conditional_distribution(
            x_obs=task["x_obs"],
            y_obs=task["y_obs"],
            x_tgt=task["x_tgt"],
        )

        pred_mean_std_np = pred_mean_std.detach().cpu().numpy()
        pred_cov_std_np = pred_cov_std.detach().cpu().numpy()

        pred_var_std_np = np.diag(pred_cov_std_np)
        pred_std_std_np = np.sqrt(np.maximum(pred_var_std_np, 1e-12))

        pred_mean_raw = pred_mean_std_np * y_std + y_mean
        pred_std_raw = pred_std_std_np * y_std
        true_raw = task["y_tgt_raw"]

        # Save full covariance for this (day, center) task
        covariance_records.append({
            "file": task["file"],
            "timestamp": str(task["timestamp"]),
            "center_node": int(task["center_node"]),
            "target_neighbors": [int(n) for n in task["target_neighbors"]],
            "pred_mean_raw": pred_mean_raw.astype(float),
            "pred_cov_raw": (pred_cov_std_np * (y_std ** 2)).astype(float),
            "true_values_raw": true_raw.astype(float),
            "center_value_raw": float(task["y_obs_raw"]),
        })

        # Save one row per neighbor
        for k, nbr in enumerate(task["target_neighbors"]):
            rows.append({
                "file": task["file"],
                "timestamp": task["timestamp"],
                "center_node": int(task["center_node"]),
                "center_value": float(task["y_obs_raw"]),
                "target_neighbor": int(nbr),
                "pred_mean": float(pred_mean_raw[k]),
                "pred_std": float(pred_std_raw[k]),
                "true_value": float(true_raw[k]),
                "abs_error": float(abs(pred_mean_raw[k] - true_raw[k])),
                "sq_error": float((pred_mean_raw[k] - true_raw[k]) ** 2),
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
    G, node_to_pos, eigenvalues, eigenvectors, eligible_centers = build_graph_objects(
    GRAPH_PATH,
    GRAPH_SPECTRUM_DIR,
    )

    print(f"Eligible centers with degree >= {MIN_DEGREE}: {len(eligible_centers)}")
    print("First 10 eligible centers and degrees:")
    for n in eligible_centers[:10]:
        print(f"  node={n}, degree={G.degree[n]}")

    print("Loaded precomputed spectrum:")
    print("  eigenvalues shape :", tuple(eigenvalues.shape))
    print("  eigenvectors shape:", tuple(eigenvectors.shape))
    print("  mapped nodes      :", len(node_to_pos))

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
    # Fit normalization on train only
    # --------------------------------------------------------
    relevant_nodes = get_relevant_nodes(G, eligible_centers)
    y_mean, y_std = fit_standardization(train_rows, relevant_nodes)

    print(f"Train normalization mean = {y_mean:.6f}")
    print(f"Train normalization std  = {y_std:.6f}")

    # --------------------------------------------------------
    # Build tasks
    # --------------------------------------------------------
    train_tasks = build_tasks(train_rows, G, eligible_centers, node_to_pos, y_mean, y_std)
    val_tasks = build_tasks(val_rows, G, eligible_centers, node_to_pos, y_mean, y_std)
    test_tasks = build_tasks(test_rows, G, eligible_centers, node_to_pos, y_mean, y_std)

    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks:   {len(val_tasks)}")
    print(f"Test tasks:  {len(test_tasks)}")

    if len(train_tasks) == 0:
        raise RuntimeError(
            "No training tasks were built. Check:\n"
            "  - file ordering\n"
            "  - target time selection\n"
            "  - dataframe key in the pickles\n"
            "  - node id alignment"
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

    best_state = None
    best_val_rmse = float("inf")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=DEVICE)

        for task in train_tasks:
            task_loss = model.task_nll(
                x_obs=task["x_obs"],
                y_obs=task["y_obs"],
                x_tgt=task["x_tgt"],
                y_tgt=task["y_tgt"],
            )
            total_loss = total_loss + task_loss

        total_loss = total_loss / len(train_tasks)
        total_loss.backward()
        optimizer.step()

        # Validation
        val_df, _ = evaluate_tasks(model, val_tasks, y_mean, y_std)
        val_metrics = compute_metrics(val_df)
        val_rmse = val_metrics["rmse"]

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == NUM_EPOCHS:
            print(
                f"Epoch {epoch:03d} | "
                f"Train NLL: {total_loss.item():.4f} | "
                f"Val RMSE: {val_rmse:.4f} | "
                f"nu: {model.kernel.nu.item():.4f} | "
                f"kappa: {model.kernel.kappa.item():.4f} | "
                f"outputscale: {model.kernel.outputscale.item():.4f} | "
                f"mean: {model.mean.item():.4f} | "
                f"noise: {model.noise.item():.4f}"
            )

    # --------------------------------------------------------
    # Restore best state
    # --------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------
    val_df, val_cov_records = evaluate_tasks(model, val_tasks, y_mean, y_std)
    test_df, test_cov_records = evaluate_tasks(model, test_tasks, y_mean, y_std)

    val_metrics = compute_metrics(val_df)
    test_metrics = compute_metrics(test_df)

    print("\nValidation")
    print(f"MAE:  {val_metrics['mae']:.4f}")
    print(f"RMSE: {val_metrics['rmse']:.4f}")

    print("\nTest")
    print(f"MAE:  {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")

    if len(test_df) > 0:
        print("\nTest by center node (top 20 by count)")
        by_center = (
            test_df.groupby("center_node")
            .agg(
                count=("center_node", "size"),
                mae=("abs_error", "mean"),
                rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
            )
            .sort_values("count", ascending=False)
            .head(20)
        )
        print(by_center)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    out_dir = Path("outputs_multi_center_fixed_time")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df.to_csv(out_dir / "val_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # Full predictive covariance per (day, center) task
    torch.save(val_cov_records, out_dir / "val_task_covariances.pt")
    torch.save(test_cov_records, out_dir / "test_task_covariances.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_time": TARGET_TIME,
            "min_degree": MIN_DEGREE,
            "eligible_centers": eligible_centers,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        out_dir / "shared_model.pt",
    )

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()