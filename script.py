"""
fNIRS 3-Class Classification Take-Home Problem

This script demonstrates a complete pipeline for classifying mental states from
functional near-infrared spectroscopy (fNIRS) brain imaging data:

1. Load fNIRS data (HbO/HbR concentrations) and stimulus protocols
2. Preprocess with highpass filtering
3. Compute single-trial beta estimates using GLM with canonical HRF
4. Train a neural network classifier

The task: Classify brain activity into 3 mental states:
- Mental Arithmetic
- Singing a song  
- WhatsApp

Data format:
- reconstruction_data.npz: Contains 'hbo', 'hbr' (4D: x,y,z,time), 'time' (1D)
- protocol.json: List of {stimulus, t_start, t_end} trial definitions
"""

import gc
import json
import shutil
import tempfile
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy.stats import gamma
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Update this path to where you extracted the data
DATA_DIR = Path("/data/thomas/lys_work_sample_data")
SAMPLING_FREQ = 4.762  # Hz (fNIRS sampling rate)
HIGHPASS_CUTOFF = 0.01  # Hz
NUM_CLASSES = 3
LABEL_MAP = {"Mental Arithmetic": 0, "Singing a song": 1, "WhatsApp": 2}


@dataclass
class Trial:
    """Represents a single trial with stimulus label and timing."""
    stimulus: str
    t_start: float
    t_end: float
    session_idx: int
    label: int  # 0, 1, or 2


def load_all_sessions_memmap(
    data_dir: Path,
    fs: float = SAMPLING_FREQ,
    cutoff_hz: float = HIGHPASS_CUTOFF,
) -> tuple[np.memmap, dict[int, tuple[int, int]], list[np.ndarray], list[Trial], Path]:
    """Load all sessions with memmap to keep peak memory low-ish.
    
    Uses mmap_mode='r' to load npz data without full RAM allocation, then filters
    slice-by-slice.
    
    Returns:
        all_hbo: Memmap of all HbO data, shape (x, y, z, total_timepoints)
        session_offsets: Dict mapping session_idx -> (offset_in_time_axis, n_timepoints)
        time_list: List of time vectors, one per session
        all_trials: All trials across sessions
        temp_dir: Path to temp directory (caller should clean up with shutil.rmtree)
    """
    session_dirs = sorted(data_dir.glob("session-*"))
    temp_dir = Path(tempfile.mkdtemp())
    
    session_data: dict[int, tuple[Path, int]] = {}
    time_list: list[np.ndarray] = []
    all_trials: list[Trial] = []
    spatial_shape: tuple[int, int, int] | None = None
    
    print("Loading sessions...")
    
    for session_dir in session_dirs:
        session_idx = int(session_dir.name.split("-")[1])
        
        # Load with mmap_mode='r' - no RAM used until slicing
        npz = np.load(session_dir / "reconstruction_data.npz", mmap_mode="r")
        hbo_mmap = npz["hbo"]  # (x, y, z, time), memory-mapped
        time = np.asarray(npz["time"])  # time vector needs to be loaded
        
        with open(session_dir / "protocol.json") as f:
            protocol = json.load(f)
        
        trials = [
            Trial(
                stimulus=p["stimulus"],
                t_start=p["t_start"],
                t_end=p["t_end"],
                session_idx=session_idx,
                label=LABEL_MAP[p["stimulus"]],
            )
            for p in protocol
        ]
        
        # Get spatial shape from first session
        if spatial_shape is None:
            spatial_shape = hbo_mmap.shape[:3]  # (x, y, z)
        
        n_tp = hbo_mmap.shape[3]
        
        # Filter slice-by-slice, writing directly to memmap (~300MB peak RAM)
        memmap_path = temp_dir / f"session_{session_idx}.dat"
        highpass_filter_to_memmap(hbo_mmap, memmap_path, fs=fs, cutoff_hz=cutoff_hz)
        
        session_data[session_idx] = (memmap_path, n_tp)
        time_list.append(time)
        all_trials.extend(trials)
        
        print(f"  Session {session_idx}: {n_tp} timepoints, {len(trials)} trials")
        
        del npz, hbo_mmap
        gc.collect()
    
    # Combine temp memmaps into final memmap with shape (x, y, z, total_time)
    total_tp = sum(n_tp for _, n_tp in session_data.values())
    print(f"\nTotal timepoints: {total_tp}, combining into final memmap...")
    
    all_hbo = np.memmap(
        str(temp_dir / "all_hbo.dat"),
        dtype="float32",
        mode="w+",
        shape=spatial_shape + (total_tp,),  # (x, y, z, total_time)
    )
    
    session_offsets: dict[int, tuple[int, int]] = {}
    offset = 0
    for sess_idx in sorted(session_data.keys()):
        memmap_path, n_tp = session_data[sess_idx]
        temp_memmap = np.memmap(
            str(memmap_path), dtype="float32", mode="r", shape=spatial_shape + (n_tp,)
        )
        all_hbo[..., offset : offset + n_tp] = temp_memmap
        session_offsets[sess_idx] = (offset, n_tp)
        offset += n_tp
        del temp_memmap
        memmap_path.unlink()  # Delete temp file
    
    all_hbo.flush()
    
    print(f"Total: {len(session_data)} sessions, {len(all_trials)} trials")
    return all_hbo, session_offsets, time_list, all_trials, temp_dir


def highpass_filter_to_memmap(
    hbo_mmap: np.ndarray,
    output_path: Path,
    fs: float,
    cutoff_hz: float = 0.01,
    order: int = 4,
) -> np.memmap:
    """Filter along time axis slice-by-slice, writing directly to memmap.
    
    Processes one x-slice at a time to minimize RAM usage.
    
    Args:
        hbo_mmap: Input data with shape (x, y, z, time), typically a memory-mapped array
        output_path: Path where the output memmap will be written
        fs: Sampling frequency in Hz
        cutoff_hz: Cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Memory-mapped filtered data with same shape as input
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(order, normalized_cutoff, btype="highpass", output="sos")
    
    x_dim = hbo_mmap.shape[0]
    output = np.memmap(
        str(output_path), dtype="float32", mode="w+", shape=hbo_mmap.shape
    )
    
    for i in range(x_dim):
        slice_data = np.asarray(hbo_mmap[i])  # (y, z, time), loads from mmap
        output[i] = sosfiltfilt(sos, slice_data, axis=2).astype(np.float32)
    
    output.flush()
    return output


def canonical_hrf(t: np.ndarray, tau: float = 1.0, ratio: float = 1/6, 
                  p1: int = 5, p2: int = 16,
                  initial_dip_ratio: float = 0.2, initial_dip_shape: int = 3) -> np.ndarray:
    """Compute canonical hemodynamic response function (triple-gamma model).
    
    The HRF models the delayed and dispersed blood flow response to neural activity,
    including an initial dip, main peak, and post-stimulus undershoot.
    
    Args:
        t: Time points in seconds
        tau: Time scaling parameter
        ratio: Undershoot to peak ratio
        p1: Shape parameter for peak response
        p2: Shape parameter for undershoot
        initial_dip_ratio: Amplitude of initial dip relative to peak
        initial_dip_shape: Shape parameter for initial dip
        
    Returns:
        HRF values normalized to peak amplitude of 1
    """
    t = np.asarray(t)
    t_scaled = t / tau
    
    # Triple-gamma: initial dip, peak, and undershoot
    peak = gamma.pdf(t_scaled, p1)
    undershoot = gamma.pdf(t_scaled, p2)
    initial_dip = gamma.pdf(t_scaled, initial_dip_shape)
    hrf = -initial_dip_ratio * initial_dip + peak - ratio * undershoot
    
    # Normalize to unit peak
    hrf = hrf / np.max(np.abs(hrf))
    return hrf


def make_trial_regressor(t_start: float, t_end: float, n_timepoints: int, 
                         fs: float, hrf: np.ndarray, session_time_offset: float = 0.0) -> np.ndarray:
    """Create a single trial regressor (boxcar convolved with HRF).
    
    Args:
        t_start: Trial start time in seconds (absolute protocol time)
        t_end: Trial end time in seconds (absolute protocol time)
        n_timepoints: Total number of timepoints in the session
        fs: Sampling frequency in Hz
        hrf: Pre-computed HRF kernel
        session_time_offset: Start time of the session (to convert absolute times to indices)
        
    Returns:
        Regressor array of shape (n_timepoints,)
    """
    # Create boxcar function (1 during stimulus, 0 otherwise)
    boxcar = np.zeros(n_timepoints)
    start_idx = int(np.round((t_start - session_time_offset) * fs))
    end_idx = int(np.round((t_end - session_time_offset) * fs))
    start_idx = max(0, min(start_idx, n_timepoints))
    end_idx = max(0, min(end_idx, n_timepoints))
    boxcar[start_idx:end_idx] = 1.0
    
    # Convolve with HRF and truncate to original length
    regressor = np.convolve(boxcar, hrf, mode="full")[:n_timepoints]
    return regressor


def make_design_matrix(trials: list[Trial], n_timepoints: int, fs: float, 
                       session_time_offset: float) -> np.ndarray:
    """Build GLM design matrix with one regressor per trial.
    
    Args:
        trials: List of Trial objects (must all be from same session)
        n_timepoints: Number of timepoints in the session
        fs: Sampling frequency in Hz
        session_time_offset: Start time of the session (first protocol interval start)
        
    Returns:
        Design matrix of shape (n_timepoints, n_trials)
    """
    # Pre-compute HRF (32 seconds duration)
    hrf_duration = 32.0
    hrf_time = np.arange(0, hrf_duration, 1/fs)
    hrf = canonical_hrf(hrf_time)
    
    n_trials = len(trials)
    design = np.zeros((n_timepoints, n_trials), dtype=np.float32)
    
    for i, trial in enumerate(trials):
        design[:, i] = make_trial_regressor(
            trial.t_start, trial.t_end, n_timepoints, fs, hrf, session_time_offset
        )
    
    return design


def compute_voxel_mask_from_memmap(
    all_hbo: np.memmap,
    session_offsets: dict[int, tuple[int, int]],
    include_session_indices: set[int],
    percentile_threshold: float = 1.0,
    chunk_size: int = 500,
) -> np.ndarray:
    """Compute voxel mask from memmap using chunked variance computation.
    
    Uses Var(X) = E[X^2] - E[X]^2 formula, accumulating sums in chunks to
    avoid loading all data into memory.
    
    Args:
        all_hbo: Memmap of all HbO data, shape (x, y, z, total_timepoints)
        session_offsets: Dict mapping session_idx -> (offset_in_time_axis, n_timepoints)
        include_session_indices: Which sessions to include in mask computation
        percentile_threshold: Exclude voxels below this variance percentile
        chunk_size: Number of timepoints to process at once
        
    Returns:
        Boolean mask of shape (x, y, z)
    """
    spatial_shape = all_hbo.shape[:3]  # (x, y, z)
    sum_x = np.zeros(spatial_shape, dtype=np.float64)
    sum_x2 = np.zeros(spatial_shape, dtype=np.float64)
    total_tp = 0
    
    for sess_idx in include_session_indices:
        if sess_idx not in session_offsets:
            continue
        offset, n_tp = session_offsets[sess_idx]
        total_tp += n_tp
        
        for start in range(0, n_tp, chunk_size):
            end = min(start + chunk_size, n_tp)
            # Data is (x, y, z, time), slice along last axis
            chunk = all_hbo[..., offset + start : offset + end].astype(np.float64)
            sum_x += chunk.sum(axis=3)
            sum_x2 += (chunk**2).sum(axis=3)
    
    if total_tp == 0:
        raise ValueError("No timepoints found for specified sessions")
    
    mean = sum_x / total_tp
    variance = (sum_x2 / total_tp) - (mean**2)
    
    threshold = max(np.percentile(variance, percentile_threshold), 1e-10)
    mask = variance >= threshold
    
    n_kept = mask.sum()
    n_total = mask.size
    print(f"Voxel mask: keeping {n_kept}/{n_total} voxels ({100*n_kept/n_total:.1f}%)")
    return mask


def compute_betas_ols(design: np.ndarray, hbo_masked: np.ndarray) -> np.ndarray:
    """Compute GLM betas using ordinary least squares.
    
    Solves: Y = X @ beta + error
    Solution: beta = (X'X)^-1 @ X' @ Y
    
    Args:
        design: Design matrix, shape (n_timepoints, n_trials)
        hbo_masked: Masked brain data, shape (n_timepoints, n_voxels)
        
    Returns:
        Beta coefficients, shape (n_trials, n_voxels)
    """
    # Compute (X'X)^-1 @ X'
    XtX = design.T @ design
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    
    # beta = (X'X)^-1 @ X' @ Y
    betas = XtX_inv @ design.T @ hbo_masked
    return betas


def compute_session_betas_from_memmap(
    all_hbo: np.memmap,
    session_offset: int,
    n_timepoints: int,
    trials: list[Trial],
    mask: np.ndarray,
    fs: float,
    time_vector: np.ndarray,
    chunk_size: int = 500,
) -> np.ndarray:
    """Compute single-trial betas for one session from memmap.
    
    Args:
        all_hbo: Memmap of all HbO data, shape (x, y, z, total_timepoints)
        session_offset: Start index of this session in the time axis
        n_timepoints: Number of timepoints in this session
        trials: List of Trial objects for this session
        mask: Voxel mask, shape (x, y, z)
        fs: Sampling frequency
        time_vector: Time vector for this session
        chunk_size: Number of timepoints to load at once
        
    Returns:
        Betas, shape (n_trials, n_voxels)
    """
    session_time_offset = time_vector[0]
    mask_flat = mask.ravel()
    n_voxels = mask_flat.sum()
    
    # Load masked data in chunks, transposing from (x,y,z,time) to (time,voxels)
    hbo_masked = np.zeros((n_timepoints, n_voxels), dtype=np.float32)
    for start in range(0, n_timepoints, chunk_size):
        end = min(start + chunk_size, n_timepoints)
        # Data is (x, y, z, time), slice along last axis and transpose
        chunk = all_hbo[..., session_offset + start : session_offset + end].astype(np.float32)
        # Transpose to (time, x, y, z) then flatten spatial dims
        chunk_transposed = np.transpose(chunk, (3, 0, 1, 2))  # (chunk_time, x, y, z)
        chunk_flat = chunk_transposed.reshape(end - start, -1)  # (chunk_time, voxels)
        hbo_masked[start:end] = chunk_flat[:, mask_flat]
    
    # Build design matrix
    design = make_design_matrix(trials, n_timepoints, fs, session_time_offset)
    
    # Compute betas
    betas = compute_betas_ols(design, hbo_masked)
    return betas


class Classifier(nn.Module):
    def __init__(self, n_voxels: int, hidden_dim: int = 512, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_voxels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(train_betas: np.ndarray, train_labels: np.ndarray,
                     val_betas: np.ndarray, val_labels: np.ndarray,
                     epochs: int = 10, batch_size: int = 32, lr: float = 0.001,
                     device: str = "cuda", seed: int = 42) -> nn.Module:
    """
    Args:
        train_betas: Training features, shape (n_train, n_voxels)
        train_labels: Training labels, shape (n_train,)
        val_betas: Validation features
        val_labels: Validation labels
        epochs: Maximum training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
        seed: Random seed for reproducibility
        
    Returns:
        Trained model (best validation accuracy checkpoint)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Normalize features
    scaler = StandardScaler()
    train_betas_norm = scaler.fit_transform(train_betas).astype(np.float32)
    val_betas_norm = scaler.transform(val_betas).astype(np.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_betas_norm),
        torch.from_numpy(train_labels).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_betas_t = torch.from_numpy(val_betas_norm).to(device)
    val_labels_t = torch.from_numpy(val_labels).long().to(device)
    
    # Initialize model
    n_voxels = train_betas.shape[1]
    model = Classifier(n_voxels, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for betas_batch, labels_batch in train_loader:
            betas_batch = betas_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(betas_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_betas_t)
            val_loss = criterion(val_logits, val_labels_t).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels_t).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.1%}")
    
    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    return model, scaler


def generate_cv_folds(
    session_indices: list[int], sessions_per_fold: int = 2
) -> list[tuple[set[int], set[int]]]:
    """Generate cross-validation folds by partitioning sessions.
    
    Args:
        session_indices: List of all session indices
        sessions_per_fold: Number of sessions to hold out per fold
        
    Returns:
        List of (train_sessions, val_sessions) tuples
    """
    n_sessions = len(session_indices)
    n_folds = n_sessions // sessions_per_fold
    
    folds = []
    for i in range(n_folds):
        start_idx = i * sessions_per_fold
        end_idx = start_idx + sessions_per_fold
        val_sessions = set(session_indices[start_idx:end_idx])
        train_sessions = set(session_indices) - val_sessions
        folds.append((train_sessions, val_sessions))
    
    return folds


def run_single_fold(
    fold_idx: int,
    train_session_indices: set[int],
    val_session_indices: set[int],
    all_hbo: np.memmap,
    session_offsets: dict[int, tuple[int, int]],
    time_list: list[np.ndarray],
    all_trials: list[Trial],
    session_indices: list[int],
) -> dict:
    """Run training and evaluation for a single CV fold."""
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}: val_sessions={sorted(val_session_indices)}")
    print(f"{'='*60}")
    
    mask = compute_voxel_mask_from_memmap(all_hbo, session_offsets, train_session_indices)
    
    all_betas = []
    all_labels = []
    all_session_idx = []
    
    for i, sess_idx in enumerate(session_indices):
        session_trials = [t for t in all_trials if t.session_idx == sess_idx]
        if len(session_trials) == 0:
            continue
        
        offset, n_tp = session_offsets[sess_idx]
        betas = compute_session_betas_from_memmap(
            all_hbo, offset, n_tp, session_trials, mask, SAMPLING_FREQ, time_list[i]
        )
        all_betas.append(betas)
        all_labels.extend([t.label for t in session_trials])
        all_session_idx.extend([sess_idx] * len(session_trials))
    
    all_betas = np.vstack(all_betas)
    all_labels = np.array(all_labels)
    all_session_idx = np.array(all_session_idx)
    
    train_mask = np.isin(all_session_idx, list(train_session_indices))
    val_mask = np.isin(all_session_idx, list(val_session_indices))
    
    train_betas = all_betas[train_mask]
    train_labels = all_labels[train_mask]
    val_betas = all_betas[val_mask]
    val_labels = all_labels[val_mask]
    
    print(f"Train: {len(train_betas)} trials, Val: {len(val_betas)} trials")
    
    model, scaler = train_classifier(train_betas, train_labels, val_betas, val_labels)
    
    model.eval()
    device = next(model.parameters()).device
    val_betas_norm = scaler.transform(val_betas).astype(np.float32)
    val_betas_t = torch.from_numpy(val_betas_norm).to(device)
    
    with torch.no_grad():
        val_preds = model(val_betas_t).argmax(dim=1).cpu().numpy()
    
    accuracy = (val_preds == val_labels).mean()
    
    per_class_accuracy = {}
    for label_name, label_idx in LABEL_MAP.items():
        class_mask = val_labels == label_idx
        if class_mask.sum() > 0:
            per_class_accuracy[label_name] = (val_preds[class_mask] == label_idx).mean()
    
    print(f"Fold {fold_idx + 1} accuracy: {accuracy:.1%}")
    
    return {
        "fold_idx": fold_idx,
        "val_sessions": val_session_indices,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "val_preds": val_preds,
        "val_labels": val_labels,
    }


def print_cv_summary(fold_results: list[dict]) -> None:
    """Print summary statistics across all CV folds."""
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)
    
    accuracies = [r["accuracy"] for r in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print(f"\nOverall: {mean_acc:.1%} +/- {std_acc:.1%}")
    print(f"Chance level: {1/NUM_CLASSES:.1%}")
    
    print("\nPer-fold results:")
    for r in fold_results:
        print(f"  Fold {r['fold_idx'] + 1} (sessions {sorted(r['val_sessions'])}): {r['accuracy']:.1%}")
    
    print("\nPer-class accuracy (averaged across folds):")
    for label_name in LABEL_MAP.keys():
        class_accs = [r["per_class_accuracy"].get(label_name, np.nan) for r in fold_results]
        valid_accs = [a for a in class_accs if not np.isnan(a)]
        if valid_accs:
            mean_class_acc = np.mean(valid_accs)
            std_class_acc = np.std(valid_accs)
            print(f"  {label_name}: {mean_class_acc:.1%} +/- {std_class_acc:.1%}")


def run_pipeline(data_dir: Path = DATA_DIR, sessions_per_fold: int = 2):
    """Run N-fold cross-validation pipeline.
    
    Args:
        data_dir: Path to data directory containing session folders
        sessions_per_fold: Number of sessions to hold out per fold
    """
    print("=" * 60)
    print("fNIRS 3-Class Classification Pipeline (N-Fold CV)")
    print("=" * 60)
    
    print("\n[1/2] Loading and filtering data...")
    all_hbo, session_offsets, time_list, all_trials, temp_dir = load_all_sessions_memmap(data_dir)
    
    try:
        session_indices = sorted(session_offsets.keys())
        folds = generate_cv_folds(session_indices, sessions_per_fold)
        
        print(f"\nTotal sessions: {len(session_indices)}")
        print(f"Number of folds: {len(folds)}")
        print(f"Sessions per fold: {sessions_per_fold}")
        
        print("\n[2/2] Running cross-validation...")
        fold_results = []
        for fold_idx, (train_sessions, val_sessions) in enumerate(folds):
            result = run_single_fold(
                fold_idx=fold_idx,
                train_session_indices=train_sessions,
                val_session_indices=val_sessions,
                all_hbo=all_hbo,
                session_offsets=session_offsets,
                time_list=time_list,
                all_trials=all_trials,
                session_indices=session_indices,
            )
            fold_results.append(result)
        
        print_cv_summary(fold_results)
        
        return fold_results
    
    finally:
        del all_hbo
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    run_pipeline()

