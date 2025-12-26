# qsann_visualize_testsets.py
# Visualize original + overlay for each fold’s test set
# Matches the split/seed used when checkpoints were saved in 64x64_with_4x4_patch_test.py.

import os
import glob
import math
import random
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold

import pennylane as qml
import io
import pandas as pd

# -----------------------
# Config (match training)
# -----------------------
Denc = 24
D = 2
S = 256
n = 4
num_layers = 1
image_size = 128
patch_size = (8, 8)
patch_shape = (16, 16)
TITLE = "mixed_128x128_with_8x8_patch_prognosis"
N_SPLITS = 5
RANDOM_STATE = 42

# Choose which split to reproduce: "stratified" (default) or "grouped"
SPLIT_MODE = "grouped"  # or "grouped"

DATA_ROOT = "/home/junyeollee/.jupyter/QSANN/data/prognosis"
CSV_PATH = "/home/junyeollee/.jupyter/QSANN/data/prognosis/sev_intial_detail_ver1.csv"
MODEL_DIR = "/home/junyeollee/.jupyter/mets/model_results"
OUT_BASE = "/home/junyeollee/.jupyter/QSANN/visualization/quadrant/30epoch"
OUT_STROMA = os.path.join(OUT_BASE, "favorable")
OUT_TUMOR = os.path.join(OUT_BASE, "poor")
os.makedirs(OUT_STROMA, exist_ok=True)
os.makedirs(OUT_TUMOR, exist_ok=True)

# Additional 4-way buckets by GT/Pred
OUT_GT0_PRED0 = os.path.join(OUT_BASE, "GT0_Pred0")
OUT_GT0_PRED1 = os.path.join(OUT_BASE, "GT0_Pred1")
OUT_GT1_PRED0 = os.path.join(OUT_BASE, "GT1_Pred0")
OUT_GT1_PRED1 = os.path.join(OUT_BASE, "GT1_Pred1")
for d in [OUT_GT0_PRED0, OUT_GT0_PRED1, OUT_GT1_PRED0, OUT_GT1_PRED1]:
    os.makedirs(d, exist_ok=True)

device = torch.device("cpu")
random.seed(42)
torch.manual_seed(42)

# -----------------------
# Dataset utils
# -----------------------
def split_into_non_overlapping_patches(image: np.ndarray, patch_size: Tuple[int, int] = patch_size) -> np.ndarray:
    patches = []
    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]].flatten()
            patches.append(patch)
    return np.array(patches)  # (256, 48) for RGB

# -----------------------
# Prognosis loader (same as training)
# -----------------------
def read_csv_flexible(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8", "cp949", "euc-kr", "iso-8859-1"]
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    with open(csv_path, "rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        text = data.decode("latin1", errors="ignore")
    return pd.read_csv(io.StringIO(text))

def load_prognosis_quadrant_pngs(
    root: str = DATA_ROOT,
    csv_path: str = CSV_PATH,
    seed: int = 42,
):
    df = read_csv_flexible(csv_path)
    label_col = "label (1:poor prognosis, 0:favorable prognosis)"
    if label_col not in df.columns or "sample_name" not in df.columns:
        raise ValueError(f"CSV must contain 'sample_name' and '{label_col}' columns")
    label_map = {
        str(row["sample_name"]).strip().lower(): int(row[label_col])
        for _, row in df.iterrows()
        if pd.notna(row["sample_name"]) and pd.notna(row[label_col])
    }

    Xs: List[np.ndarray] = []
    ys: List[int] = []
    groups: List[str] = []
    paths: List[str] = []

    rng = np.random.default_rng(seed)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    total_files = 0
    used_files = 0
    skipped_no_label = 0

    for patient in sorted(os.listdir(root)):
        pdir = os.path.join(root, patient)
        if not os.path.isdir(pdir):
            continue
        files = [
            os.path.join(pdir, f)
            for f in os.listdir(pdir)
            if f.lower().startswith("quadrant_") and os.path.splitext(f)[1].lower() in exts
        ]
        for f in files:
            total_files += 1
            base = os.path.basename(f)
            name_wo_ext, _ = os.path.splitext(base)
            key = name_wo_ext[len("quadrant_") :].strip()
            key_norm = key.lower()
            label = label_map.get(key_norm)
            if label is None:
                folder_norm = str(patient).strip().lower()
                label = label_map.get(folder_norm)
            if label is None:
                skipped_no_label += 1
                continue
            try:
                img = Image.open(f).convert("RGB")
                if img.size != (image_size, image_size):
                    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
                arr = np.array(img, dtype=np.float32)
                Xs.append(arr)
                ys.append(int(label))
                groups.append(key)
                paths.append(f)
                used_files += 1
            except Exception:
                continue

    if not Xs:
        raise RuntimeError(f"No quadrant images with labels found under {root}")

    print(
        f"[Loader] scanned={total_files}, used={used_files}, skipped_no_label={skipped_no_label}, "
        f"unique_patients={len(set(groups))}"
    )

    X = torch.tensor(np.stack(Xs, axis=0), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.int64)
    return X, y, groups, paths

def get_grouped_splits(y: torch.Tensor, groups: List[str]):
    uniq = len(set(groups))
    eff_splits = min(N_SPLITS, uniq)
    idx = np.arange(y.shape[0])
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=eff_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(sgkf.split(idx, y.numpy(), groups=np.array(groups)))
    except Exception:
        from sklearn.model_selection import GroupKFold
        gk = GroupKFold(n_splits=eff_splits)
        return list(gk.split(idx, y.numpy(), groups=np.array(groups)))

# -----------------------
# Model (match training)
# -----------------------
class TorchLayer(nn.Module):
    def __init__(self, qnode, weights):
        super().__init__()
        self.qnode = qnode
        self.qnode.interface = "torch"
        self.q_weight = nn.Parameter(weights["weights"])
        self._input_arg = "inputs"
    def forward(self, inputs, loc_angles=None, texture_angles=None):
        return self._evaluate_qnode(inputs, loc_angles, texture_angles)
    def _evaluate_qnode(self, x, loc_angles=None, texture_angles=None):
        kwargs = {self._input_arg: x, "weights": self.q_weight.to(x.device)}
        if loc_angles is not None:
            kwargs["loc_angles"] = loc_angles.to(x.device)
        if texture_angles is not None:
            kwargs["texture_angles"] = texture_angles.to(x.device)
        res = self.qnode(**kwargs)
        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)
        if isinstance(res, (list, tuple)) and len(res) > 0:
            if isinstance(res[0], torch.Tensor) and res[0].ndim >= 1:
                return torch.stack(list(res), dim=-1).type(x.dtype)
            else:
                return torch.hstack(list(res)).type(x.dtype)
        return torch.tensor(res, dtype=x.dtype, device=x.device)

class QSAL_pennylane(torch.nn.Module):
    def __init__(self, S, n, Denc, D):
        super().__init__()
        self.seq_num = S
        self.num_pixel_qubits = n
        self.num_q = self.num_pixel_qubits
        self.Denc = Denc
        self.D = D
        self.d = (2 + self.D) * self.num_q
        self.init_params_Q = nn.Parameter((np.pi / 4) * (2 * torch.randn(self.d) - 1))
        self.init_params_K = nn.Parameter((np.pi / 4) * (2 * torch.randn(self.d) - 1))
        self.init_params_V = nn.Parameter((np.pi / 4) * (2 * torch.randn(self.d) - 1))
        self.dev = qml.device("lightning.qubit", wires=self.num_q)
        self.qnode_v = qml.QNode(self.circuit_v, self.dev, interface="torch", diff_method="adjoint")
        self.qnode_q = qml.QNode(self.circuit_qk, self.dev, interface="torch", diff_method="adjoint")
        self.qnode_k = self.qnode_q
        self.to_Q = TorchLayer(self.qnode_q, {"weights": self.init_params_Q})
        self.to_K = TorchLayer(self.qnode_k, {"weights": self.init_params_K})
        self.to_V = TorchLayer(self.qnode_v, {"weights": self.init_params_V})
        self.alpha = None
        for p in self.parameters():
            p.data = p.data.to(torch.device("cpu"))
        self.register_buffer("pos_coords", self._create_pos_coords())
    def _create_pos_coords(self):
        patch_side = int(math.isqrt(self.seq_num)) if hasattr(math, "isqrt") else int(np.sqrt(self.seq_num))
        pos_coords = []
        for r in range(patch_side):
            for c in range(patch_side):
                pos_coords.append([r / (patch_side - 1), c / (patch_side - 1)])
        return torch.tensor(pos_coords)
    def circuit_v(self, inputs, weights, loc_angles=None, texture_angles=None):
        s = (inputs / 255.0) * math.pi
        enc_len = 2 * self.num_pixel_qubits * self.Denc
        if s.shape[-1] > enc_len:
            s = s[..., :enc_len]
        elif s.shape[-1] < enc_len:
            s = F.pad(s, (0, enc_len - s.shape[-1]), mode="constant", value=0.0)
        for stage in range(self.Denc):
            base = 2 * self.num_pixel_qubits * stage
            for q in range(self.num_pixel_qubits):
                qml.RX(s[..., base + 2 * q], q)
                qml.RY(s[..., base + 2 * q + 1], q)
            if stage != self.Denc - 1:
                for q in range(self.num_pixel_qubits):
                    qml.CNOT(wires=(q, (q + 1) % self.num_pixel_qubits))
        indx = 0
        for j in range(self.num_q):
            if indx < weights.shape[0]:
                qml.RX(weights[indx], j)
            if indx + 1 < weights.shape[0]:
                qml.RY(weights[indx + 1], j)
            indx += 2
        for _ in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                if indx < weights.shape[0]:
                    qml.RY(weights[indx], j)
                    indx += 1
        expvals = []
        for i in range(self.num_q):
            expvals.append(qml.expval(qml.PauliX(i)))
            expvals.append(qml.expval(qml.PauliY(i)))
            expvals.append(qml.expval(qml.PauliZ(i)))
        return expvals
    def circuit_qk(self, inputs, weights, loc_angles=None, texture_angles=None):
        s = (inputs / 255.0) * math.pi
        enc_len = 2 * self.num_pixel_qubits * self.Denc
        if s.shape[-1] > enc_len:
            s = s[..., :enc_len]
        elif s.shape[-1] < enc_len:
            s = F.pad(s, (0, enc_len - s.shape[-1]), mode="constant", value=0.0)
        for stage in range(self.Denc):
            base = 2 * self.num_pixel_qubits * stage
            for q in range(self.num_pixel_qubits):
                qml.RX(s[..., base + 2 * q], q)
                qml.RY(s[..., base + 2 * q + 1], q)
            if stage != self.Denc - 1:
                for q in range(self.num_pixel_qubits):
                    qml.CNOT(wires=(q, (q + 1) % self.num_pixel_qubits))
        indx = 0
        for j in range(self.num_q):
            if indx < weights.shape[0]:
                qml.RX(weights[indx], j)
            if indx + 1 < weights.shape[0]:
                qml.RY(weights[indx + 1], j)
            indx += 2
        for _ in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                if indx < weights.shape[0]:
                    qml.RY(weights[indx], j)
                    indx += 1
        expvals = []
        for i in range(self.num_q):
            expvals.append(qml.expval(qml.PauliX(i)))
            expvals.append(qml.expval(qml.PauliY(i)))
            expvals.append(qml.expval(qml.PauliZ(i)))
        return expvals
    def forward(self, input):
        device = self.to_Q.q_weight.device
        input = input.to(device)
        B, S_, d_in = input.shape
        flat_input = input.view(B * S_, d_in)
        def _run_mb(func, x, mb: int = 4096):
            outs = []
            for i in range(0, x.shape[0], mb):
                outs.append(func(x[i: i + mb]))
            return torch.cat(outs, dim=0)
        Q_flat = _run_mb(self.to_Q, flat_input)
        K_flat = _run_mb(self.to_K, flat_input)
        V_flat = _run_mb(self.to_V, flat_input)
        Q = Q_flat.view(B, S_, -1)
        K = K_flat.view(B, S_, -1)
        V = V_flat.view(B, S_, -1)
        QQ = (Q * Q).sum(-1).unsqueeze(2)
        KK = (K * K).sum(-1).unsqueeze(1)
        QK = torch.bmm(Q, K.transpose(1, 2))
        dist2 = QQ + KK - 2.0 * QK
        self.alpha = torch.exp(-dist2)
        self.alpha = self.alpha / (self.alpha.sum(dim=-1, keepdim=True) + 1e-12)
        output = torch.bmm(self.alpha, V)
        return output.to(device)

class QSANN_pennylane(torch.nn.Module):
    def __init__(self, S, n, Denc, D, num_layers):
        super().__init__()
        self.qsal_lst = [QSAL_pennylane(S, n, Denc, D) for _ in range(num_layers)]
        self.qnn = nn.Sequential(*self.qsal_lst)
    def forward(self, input):
        return self.qnn(input)

class QSANN_text_classifier(torch.nn.Module):
    def __init__(self, S, n, Denc, D, num_layers):
        super().__init__()
        self.Qnn = QSANN_pennylane(S, n, Denc, D, num_layers)
        embed_dim = 3 * n
        self.final_layer = nn.Linear(S * embed_dim, 1)
        self.final_layer = self.final_layer.float()
    def forward(self, input, return_attention: bool = False):
        x = self.Qnn(input)
        if return_attention:
            attention = self.Qnn.qsal_lst[-1].alpha
        x = torch.flatten(x, start_dim=1)
        output = self.final_layer(x)
        if return_attention:
            return output, attention
        return output



def find_checkpoint_for_fold(fold: int) -> str:


    # 2) 정확한 기본 경로 우선
    preferred = os.path.join(MODEL_DIR, f"{TITLE}_fold{fold}.pth")
    if os.path.exists(preferred):
        return preferred

    # 4) 더 이상은 허용하지 않고 명확히 에러
    raise FileNotFoundError(
        f"No checkpoint found for fold {fold} with TITLE='{TITLE}'. "
        f"Tried: {preferred}"
    )


def save_images(orig_rgb: np.ndarray, heatmap_2d: np.ndarray, out_dir: str, base_name: str):
    scale_y = orig_rgb.shape[0] / heatmap_2d.shape[0]
    scale_x = orig_rgb.shape[1] / heatmap_2d.shape[1]
    resized_heatmap = zoom(heatmap_2d, (scale_y, scale_x))
    hm_min, hm_max = resized_heatmap.min(), resized_heatmap.max()
    if hm_max > hm_min:
        resized_heatmap = (resized_heatmap - hm_min) / (hm_max - hm_min)
    else:
        resized_heatmap = np.zeros_like(resized_heatmap)
    # original
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(orig_rgb.astype(np.uint8))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(out_dir, f"{base_name}_original.png"), bbox_inches="tight", pad_inches=0)
    plt.close()
    # overlay
    plt.figure(figsize=(4, 4))
    plt.imshow(orig_rgb.astype(np.uint8))
    plt.imshow(resized_heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(out_dir, f"{base_name}_overlay.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

def visualize_fold(fold_idx: int, test_indices: np.ndarray, X: torch.Tensor, y: torch.Tensor, paths: List[str]):
    ckpt = find_checkpoint_for_fold(fold_idx + 1)
    if ckpt is None:
        print(f"[WARN] No checkpoint for fold {fold_idx+1}. Skipping.")
        return
    print(f"[INFO] Fold {fold_idx+1}: using checkpoint {ckpt}")
    model = QSANN_text_classifier(S, n, Denc, D, num_layers).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    for idx in test_indices:
        orig_img = X[idx].numpy()  # (128,128,3)
        label = int(y[idx].item())
        base_name = os.path.splitext(os.path.basename(paths[idx]))[0]
        sample_patches = split_into_non_overlapping_patches(orig_img)  # (256, 48)
        sample_tensor = torch.tensor(sample_patches, dtype=torch.float32).unsqueeze(0)  # (1,256,48)
        with torch.no_grad():
            logits, attn = model(sample_tensor, return_attention=True)  # (1,1), (1,S,S)
            prob = torch.sigmoid(logits).item()
            pred_label = 1 if prob >= 0.5 else 0
        row_sums = attn.squeeze(0).sum(dim=0).cpu().numpy()  # (S,)
        heatmap_2d = row_sums.reshape(patch_shape)
        # Route by GT/Pred
        if label == 0 and pred_label == 0:
            out_dir = OUT_GT0_PRED0
        elif label == 0 and pred_label == 1:
            out_dir = OUT_GT0_PRED1
        elif label == 1 and pred_label == 0:
            out_dir = OUT_GT1_PRED0
        else:
            out_dir = OUT_GT1_PRED1
        save_images(orig_img, heatmap_2d, out_dir, f"fold{fold_idx+1}_{base_name}")
    print(f"[INFO] Fold {fold_idx+1}: saved {len(test_indices)} samples.")

def main():
    print("[INFO] Loading prognosis dataset (CSV-labeled, grouped folds)...")
    X, y, groups, paths = load_prognosis_quadrant_pngs(DATA_ROOT, CSV_PATH)
    print(f"[INFO] Loaded: X={tuple(X.shape)}, y={tuple(y.shape)}, unique_patients={len(set(groups))}")
    splits = get_grouped_splits(y, groups)
    for fold_idx, (_, test_idx) in enumerate(splits):
        visualize_fold(fold_idx, test_idx, X, y, paths)
    print(f"[DONE] Saved under:\n- {OUT_STROMA}\n- {OUT_TUMOR}")

if __name__ == "__main__":
    main()
