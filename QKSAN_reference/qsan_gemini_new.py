from __future__ import annotations

import argparse
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from pathlib import Path
from typing import List, Sequence, Tuple

import pennylane as qml

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from data_loader import build_standard_loaders

# -------------------------
# Repro
# -------------------------
def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------
# Helper Functions from data_loader.py
# -------------------------
def _normalize_image_size(image_size: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a length-2 sequence")
    return (int(image_size[0]), int(image_size[1]))


# -------------------------
# PCam Dataset from data_loader.py (Modified for QSAN)
# -------------------------
class PCamDataset(torch.utils.data.Dataset):
    """
    PatchCamelyon (Camelyon16) patches stored as .npy under class0/class1 folders.
    Modified to be compatible with QSAN's filter_remap_dataset logic.
    """

    CLASS_MAP = [("class0", 0), ("class1", 1)]

    def __init__(
        self,
        root: str | Path,
        image_size: Sequence[int] | int,
        samples_per_class: int | None = None,
        seed: int = 42,
        force_grayscale: bool = True
    ) -> None:
        self.root = Path(root)
        self.image_size = _normalize_image_size(image_size)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.force_grayscale = force_grayscale
        
        self.items = self._gather_items()
        self.labels = [lbl for _, lbl in self.items]
        
        # QSAN's filter_remap_dataset relies on accessing .targets or .data
        self.targets = self.labels 
        # We don't load all data into memory (.data) to save RAM, 
        # but filter_remap_dataset handles subsets via targets mostly.
        
        self.num_channels = 1 if force_grayscale else self._infer_channels()

    def _gather_items(self) -> List[Tuple[Path, int]]:
        rng = np.random.default_rng(self.seed)
        items: List[Tuple[Path, int]] = []
        for cname, label in self.CLASS_MAP:
            cdir = self.root / cname
            if not cdir.is_dir():
                continue
            paths = sorted(cdir.rglob("*.npy"))
            # Note: We load all paths first, filtering happens later or here if desired.
            # QSAN script handles subsampling via filter_remap_dataset, 
            # but usually limits are small there.
            
            # If samples_per_class is set here, it pre-filters files
            if self.samples_per_class is not None and len(paths) > self.samples_per_class:
                rng.shuffle(paths)
                paths = paths[: self.samples_per_class]
            for p in paths:
                items.append((p, label))
        
        if not items:
            # Fallback for empty init or checking existence without crashing immediately if optional
            pass 
            
        rng.shuffle(items)
        return items

    def _infer_channels(self) -> int:
        if not self.items: return 3
        sample = np.load(self.items[0][0])
        if sample.ndim == 2:
            return 1
        if sample.ndim == 3:
            return sample.shape[-1]
        return 3

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        arr = np.load(path) # Expected [H, W, C] or [H, W]
        
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        
        # Conversion to Tensor [C, H, W]
        img = torch.from_numpy(arr).permute(2, 0, 1).float()
        
        if img.max() > 1.0:
            img = img / 255.0
            
        # Resize
        img = TF.resize(img, self.image_size, antialias=True)

        # Grayscale conversion for QSAN (needs 1 channel for 16 qubits on 4x4)
        if self.force_grayscale and img.shape[0] == 3:
            img = TF.rgb_to_grayscale(img)

        # QSAN expects (img, label)
        return img, label


# -------------------------
# Dataset filtering + remap
# -------------------------
def filter_remap_dataset(ds, classes: list[int], per_class_limit: int | None, seed: int):
    """
    Keep only samples with label in `classes` and remap labels to 0..K-1.
    Handles both standard torchvision datasets and the custom PCamDataset.
    """
    rng = np.random.default_rng(seed)

    # read targets
    if isinstance(ds.targets, list):
        y = np.array(ds.targets, dtype=np.int64)
    else:
        # Check if targets is a tensor or numpy array
        if torch.is_tensor(ds.targets):
            y = ds.targets.detach().cpu().numpy().astype(np.int64)
        else:
             y = np.array(ds.targets, dtype=np.int64)

    classes = list(classes)
    keep = np.isin(y, classes)
    idx_all = np.where(keep)[0]

    if per_class_limit is None:
        idx_keep = idx_all
    else:
        idx_keep = []
        for c in classes:
            idx_c = idx_all[y[idx_all] == c]
            rng.shuffle(idx_c)
            idx_keep.extend(idx_c[:per_class_limit].tolist())
        idx_keep = np.array(idx_keep, dtype=np.int64)

    # remap labels
    c2n = {c: i for i, c in enumerate(classes)}
    y_new = np.array([c2n[int(y[i])] for i in idx_keep], dtype=np.int64)

    # subset data
    # Case 1: Standard Torchvision (has .data)
    if hasattr(ds, "data") and not isinstance(ds, PCamDataset):
        ds.data = ds.data[idx_keep]
        if isinstance(ds.targets, list):
            ds.targets = y_new.tolist()
        else:
            ds.targets = torch.tensor(y_new, dtype=torch.long)
    
    # Case 2: PCamDataset (list of paths in self.items)
    elif isinstance(ds, PCamDataset):
        # Filter items list based on kept indices
        ds.items = [ds.items[i] for i in idx_keep]
        ds.labels = y_new.tolist()
        ds.targets = ds.labels # Update alias
    
    else:
        # Fallback for generic datasets if they don't have .data exposed
        # This part might need custom logic if using other datasets
        pass

    return ds


# -------------------------
# Metrics
# -------------------------
def compute_metrics_binary(y_true: np.ndarray, prob_pos: np.ndarray) -> dict:
    y_pred = (prob_pos >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    try:
        auroc = roc_auc_score(y_true, prob_pos)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(y_true, prob_pos)
    except ValueError:
        auprc = float("nan")

    return {
        "Accuracy": float(acc),
        "AUROC": float(auroc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 score": float(f1),
        "AUPRC": float(auprc),
    }


def compute_metrics_multiclass(y_true: np.ndarray, prob: np.ndarray) -> dict:
    # prob: [N,K]
    K = prob.shape[1]
    y_pred = prob.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    y_oh = label_binarize(y_true, classes=list(range(K)))
    try:
        auroc = roc_auc_score(y_oh, prob, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_oh, prob, average="macro")
    except ValueError:
        auprc = float("nan")

    return {
        "Accuracy": float(acc),
        "AUROC": float(auroc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 score": float(f1),
        "AUPRC": float(auprc),
    }


def _to_grayscale(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 3:
        return img
    if img.shape[0] == 1:
        return img
    if img.shape[0] == 3:
        return TF.rgb_to_grayscale(img)
    return img.mean(dim=0, keepdim=True)


def collate_flatten(batch):
    xs = []
    ys = []
    for item in batch:
        if len(item) == 3:
            img, y, _ = item
        else:
            img, y = item
        img = _to_grayscale(img)
        if torch.is_tensor(y):
            y = int(y.view(-1)[0].item())
        else:
            y = int(y)
        xs.append(img.view(-1))
        ys.append(y)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def compute_loss(model: nn.Module, loader, device: torch.device, loss_fn):
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def resolve_split_counts(args, num_classes: int):
    use_counts = any(c is not None for c in (args.train_count, args.val_count, args.test_count))
    if use_counts:
        if not all(c is not None for c in (args.train_count, args.val_count, args.test_count)):
            raise ValueError("train_count, val_count, and test_count must all be set when using count-based split.")
        return args.train_count, args.val_count, args.test_count, args.samples_per_label

    if args.use_ratios:
        return None, None, None, args.samples_per_label

    train_count = args.train_per_class * num_classes
    val_count = args.val_per_class * num_classes
    test_count = args.test_per_class * num_classes
    samples_per_label = args.samples_per_label
    if samples_per_label is None:
        samples_per_label = args.train_per_class + args.val_per_class + args.test_per_class
    return train_count, val_count, test_count, samples_per_label


# -------------------------
# QSAN Model
# -------------------------
class QSANModel(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int, num_classes: int, shots: int = 0):
        super().__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.shots = None if shots == 0 else int(shots)

        # Wires:
        # q: 0..n-1
        # k: n..2n-1
        # work: 2n
        # out: 2n+1
        self.q_wires = list(range(0, n_qubits))
        self.k_wires = list(range(n_qubits, 2 * n_qubits))
        self.work = 2 * n_qubits
        self.out = 2 * n_qubits + 1
        self.total_wires = 2 * n_qubits + 2

        self.dev_qls = qml.device("default.qubit", wires=self.total_wires, shots=self.shots)
        self.dev_val = qml.device("default.qubit", wires=n_qubits, shots=self.shots)

        # Trainable circuit weights (torch parameters)
        wshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.w_q = nn.Parameter(0.01 * torch.randn(*wshape))
        self.w_k = nn.Parameter(0.01 * torch.randn(*wshape))
        self.w_v = nn.Parameter(0.01 * torch.randn(*wshape))

        # Classification head
        self.cls = nn.Linear(n_qubits, num_classes)

        # ---- QNodes (Torch interface + parameter-shift OK) ----
        @qml.qnode(self.dev_qls, interface="torch", diff_method="parameter-shift")
        def qls_prob(xq, xk, wq, wk):
            qml.AmplitudeEmbedding(xq, wires=self.q_wires, pad_with=0.0, normalize=True)
            qml.StronglyEntanglingLayers(wq, wires=self.q_wires)

            qml.AmplitudeEmbedding(xk, wires=self.k_wires, pad_with=0.0, normalize=True)
            qml.StronglyEntanglingLayers(wk, wires=self.k_wires)

            for qw, kw in zip(self.q_wires, self.k_wires):
                qml.Toffoli(wires=[qw, kw, self.work])
                qml.CNOT(wires=[self.work, self.out])
                qml.Toffoli(wires=[qw, kw, self.work])

            return qml.expval(qml.PauliZ(self.out))

        @qml.qnode(self.dev_val, interface="torch", diff_method="parameter-shift")
        def value_expvals(x, wv):
            qml.AmplitudeEmbedding(x, wires=range(n_qubits), pad_with=0.0, normalize=True)
            qml.StronglyEntanglingLayers(wv, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qls_prob = qls_prob
        self.value_expvals = value_expvals

    def forward(self, x_vec: torch.Tensor) -> torch.Tensor:
        B, D = x_vec.shape
        x_vec = x_vec.float()

        v_list = []
        p_list = []
        for b in range(B):
            vb = self.value_expvals(x_vec[b], self.w_v)
            vb = torch.stack(vb).float()
            v_list.append(vb)

            ez = self.qls_prob(x_vec[b], x_vec[b], self.w_q, self.w_k)
            pb = 0.5 * (1.0 - ez)
            p_list.append(pb.float())

        V = torch.stack(v_list, dim=0)       # [B, n_qubits]
        P = torch.stack(p_list, dim=0)       # [B]
        gated = V * P.unsqueeze(-1)          # [B, n_qubits]

        logits = self.cls(gated)             # [B, C]
        return logits


# -------------------------
# Train/Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, num_classes: int):
    model.eval()
    ys = []
    probs = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pr = F.softmax(logits, dim=1)
        ys.append(yb.detach().cpu().numpy())
        probs.append(pr.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    prob = np.concatenate(probs, axis=0)

    if num_classes == 2:
        return compute_metrics_binary(y_true, prob[:, 1])
    return compute_metrics_multiclass(y_true, prob)


def run_experiment(args, dataset_choice: str, lr: float, batch_size: int):
    seed_all(args.seed)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    K = len(args.classes)
    if K < 2:
        raise ValueError("Need at least 2 classes.")

    train_count, val_count, test_count, samples_per_label = resolve_split_counts(args, K)

    (_, train_set), (_, val_set), (_, test_set) = build_standard_loaders(
        dataset_choice=dataset_choice,
        image_size=args.img_side,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        seed=args.seed,
        dataset_labels=args.classes,
        samples_per_label=samples_per_label,
        pcam_root=args.pcam_root,
        data_root=args.data_root,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_flatten,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_flatten,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_flatten,
    )

    # qubits for 16-dim amplitude embedding -> 4
    feature_dim = args.img_side * args.img_side
    n_qubits = int(round(math.log2(feature_dim)))
    if 2 ** n_qubits != feature_dim:
        raise ValueError(
            f"img_side^2 ({feature_dim}) must be power of two for amplitude embedding (e.g. 4x4 -> 16)."
        )

    model = QSANModel(n_qubits=n_qubits, n_layers=args.q_layers, num_classes=K, shots=args.shots).to(device)

    # Nesterov Momentum SGD
    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    use_val = len(val_set) > 0
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    print(f"[QSAN] dataset={dataset_choice} classes={args.classes} K={K}")
    print(f"[QSAN] train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print(f"[QSAN] resize={args.img_side}x{args.img_side} -> dim={feature_dim} qubits={n_qubits}")
    print(f"[QSAN] epochs={args.epochs} batch={batch_size} lr={lr} nesterov(m={args.momentum}) device={device}")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_loss = compute_loss(model, val_loader, device, loss_fn) if use_val else float("nan")
        if use_val:
            if val_loss + args.early_stop_delta < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"[early-stop] no val improvement for {args.early_stop_patience} epochs.")
                break

        if ep == 1 or ep % args.eval_interval == 0 or ep == args.epochs:
            tr_m = evaluate(model, train_loader, device, num_classes=K)
            te_m = evaluate(model, test_loader, device, num_classes=K)
            print(f"[epoch {ep:03d}] loss={float(np.mean(losses)):.4f} val_loss={val_loss:.4f}")
            print("  train:", " | ".join(f"{k}={v:.4f}" for k, v in tr_m.items()))
            print("  test :", " | ".join(f"{k}={v:.4f}" for k, v in te_m.items()))

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    else:
        model.load_state_dict(best_state)

    print("\n[Final Test Metrics]")
    final = evaluate(model, test_loader, device, num_classes=K)
    for k, v in final.items():
        print(f"{k:<10}: {v:.6f}")

    return {
        "dataset": dataset_choice,
        "lr": lr,
        "batch_size": batch_size,
        "best_val_loss": best_val_loss if use_val else float("nan"),
        "test_metrics": final,
    }


def main():
    p = argparse.ArgumentParser("QSAN IEEE-style (4x4 resize) simulator training")

    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fmnist", "cifar10", "pcam"])
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Root folder for datasets (pcam expects class0/class1 or train/test/val subfolders)",
    )
    p.add_argument(
        "--pcam-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "pcam",
        help="Root folder containing PCam class0/class1 .npy patches",
    )
    p.add_argument("--classes", type=int, nargs="+", default=[0, 1], help="labels to keep (remapped to 0..K-1)")
    p.add_argument("--samples-per-label", type=int, default=None, help="cap samples per label before split")
    p.add_argument("--train-per-class", type=int, default=50)
    p.add_argument("--val-per-class", type=int, default=30)
    p.add_argument("--test-per-class", type=int, default=30)
    p.add_argument("--train-count", type=int, default=None)
    p.add_argument("--val-count", type=int, default=None)
    p.add_argument("--test-count", type=int, default=None)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--use-ratios", action="store_true", help="use ratio-based split instead of count-based")

    p.add_argument("--img-side", type=int, default=4, help="paper uses 4x4 resizing")
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=5)

    # quantum
    p.add_argument("--q-layers", type=int, default=1)
    p.add_argument("--shots", type=int, default=0, help="0=analytic, else finite shots")

    # optimizer (Nesterov)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--early-stop-delta", type=float, default=0.0)

    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")

    p.add_argument("--grid-search", action="store_true")
    p.add_argument("--grid-datasets", type=str, nargs="+", default=["mnist", "fmnist", "cifar10", "pcam"])
    p.add_argument("--grid-lrs", type=float, nargs="+", default=[0.01, 0.05])
    p.add_argument("--grid-batch-sizes", type=int, nargs="+", default=[32, 64])

    args = p.parse_args()
    if args.grid_search:
        results = []
        for dataset_choice in args.grid_datasets:
            for lr in args.grid_lrs:
                for bs in args.grid_batch_sizes:
                    print("\n==============================")
                    print(f"[grid] dataset={dataset_choice} lr={lr} batch={bs}")
                    results.append(run_experiment(args, dataset_choice, lr, bs))
        print("\n==== Grid Summary (best val loss) ====")
        for r in results:
            acc = r["test_metrics"]["Accuracy"]
            print(
                f"{r['dataset']} lr={r['lr']} batch={r['batch_size']} "
                f"best_val_loss={r['best_val_loss']:.4f} test_acc={acc:.4f}"
            )
    else:
        run_experiment(args, args.dataset, args.lr, args.batch_size)


if __name__ == "__main__":
    main()
