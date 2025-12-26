import argparse
import random
import sys
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import torch

# 경로 설정
CURRENT_DIR = Path(__file__).resolve().parent
REVISION_ROOT = CURRENT_DIR.parent
REPO_ROOT = REVISION_ROOT.parent.parent
if str(REVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(REVISION_ROOT))

from data_loader import build_pcam_dataset, build_standard_image_dataset, split_dataset

# ==========================================
# 설정 (6 Qubits, 8 Features, 6 Layers)
# ==========================================
N_QUBITS_PER_REG = 3    
N_QUBITS = N_QUBITS_PER_REG * 2  # 6 qubits
DATA_DIM = 8                     # PCA 8차원
LAYERS = 6                       # [수정] 논문 Table 2: 6 Layers

LEARNING_RATE = 0.09    
BATCH_SIZE = 30         
STEPS = 120             
GAMMA = 0.9             
CLASSES = [0, 1]        

dev = qml.device("default.qubit", wires=N_QUBITS + N_QUBITS_PER_REG)

# ==========================================
# Data Loader (기존과 동일)
# ==========================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QKSAN 6-Layer Sharing Implementation")
    parser.add_argument("--dataset-choice", type=str, default="mnist")
    parser.add_argument("--dataset-labels", type=int, nargs="*", default=CLASSES)
    parser.add_argument("--samples-per-label", type=int, default=550)
    parser.add_argument("--medmnist-subset", type=str, default="organamnist")
    parser.add_argument("--pcam-root", type=Path, default=REPO_ROOT / "data" / "camelyon" / "3channel")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    return parser

def load_data(args):
    # (데이터 로딩 부분은 이전 코드와 동일하여 생략, 실행 시 포함됨)
    print(f"Loading data... Target Dimension (PCA): {DATA_DIM}")
    if args.dataset_choice != "pcam":
        full_dataset = build_standard_image_dataset(
            dataset_choice=args.dataset_choice,
            image_size=args.image_size,
            dataset_labels=args.dataset_labels,
            samples_per_label=args.samples_per_label,
            medmnist_subset=args.medmnist_subset,
        )
    else:
        full_dataset = build_pcam_dataset(
            image_size=args.image_size,
            samples_per_class=args.samples_per_label,
            root=args.pcam_root,
            seed=args.seed,
        )

    flattened_images = []
    labels = []
    for img, lbl, _ in full_dataset:
        img_np = img.detach().cpu().numpy() if torch.is_tensor(img) else np.array(img)
        flattened_images.append(img_np.reshape(-1))
        labels.append(int(lbl))
    
    X_raw = np.asarray(flattened_images, dtype=np.float32)
    labels = np.array(labels)
    unique_labels = sorted(set(labels))
    
    print(f"Applying PCA to reduce dimensions to {DATA_DIM}...")
    pca = PCA(n_components=DATA_DIM, random_state=args.seed)
    X_pca = pca.fit_transform(X_raw)

    normalizer = Normalizer()
    X_norm = normalizer.fit_transform(X_pca).astype(np.float64)
    y_signed = np.where(labels == unique_labels[0], -1, 1)

    train_set, val_set, test_set = split_dataset(
        full_dataset, train_ratio=10/11, val_ratio=0.0, test_ratio=1/11, seed=args.seed
    )
    
    train_idx = np.array(train_set.indices)
    test_idx = np.array(test_set.indices)
    
    X_train, y_train = X_norm[train_idx], y_signed[train_idx]
    X_test, y_test = X_norm[test_idx], y_signed[test_idx]

    return X_train, y_train, None, None, X_test, y_test

# ==========================================
# 3. Model Definition: 6 Layers with Sharing
# ==========================================

def he_ansatz_shared(params, wires, depth=6):
    """
    Hardware-Efficient Ansatz with Parameter Sharing.
    params: 1D array of parameters available for this ansatz block.
    depth: Number of layers to stack (Paper: 6).
    """
    n_wires = len(wires)
    n_params = len(params)
    
    # 파라미터 인덱스 카운터
    # (파라미터 수가 적으므로 modulo 연산으로 순환 사용)
    cnt = 0
    
    for layer in range(depth):
        # 1. Hadamard (First layer only or every layer? Usually first)
        # Fig 3b shows H at the beginning.
        if layer == 0:
            for wire in wires:
                qml.Hadamard(wires=wire)
        
        # 2. Rotations (Rz, Ry) - Source 263
        for wire in wires:
            # Rz
            qml.RZ(params[cnt % n_params], wires=wire)
            cnt += 1
            # Ry
            qml.RY(params[cnt % n_params], wires=wire)
            cnt += 1
            
        # 3. Entanglement (CRy)
        # Linear entanglement: 0-1, 1-2
        for i in range(n_wires - 1):
            qml.CRY(params[cnt % n_params], wires=[wires[i], wires[i+1]])
            cnt += 1

@qml.qnode(dev, interface="autograd")
def qksan_circuit(data, params):
    reg1 = [0, 1, 2]
    reg2 = [3, 4, 5]
    
    # --- Parameter Unpacking (Total 11) ---
    # 논문의 총 파라미터 11개를 블록별로 적절히 분배
    # 예: Reg1(4개), Reg1_Inv(4개), Reg2(2개), Link(1개)
    # 파라미터 수가 매우 적으므로, 각 블록은 할당된 소수의 파라미터를 
    # 6레이어 동안 계속 재사용(Sharing)하며 학습합니다.
    
    p1 = params[0:4]   # Reg1 Ansatz용 4개
    p2 = params[4:8]   # Reg1 Inverse용 4개
    p3 = params[8:10]  # Reg2 Ansatz용 2개
    p4 = params[10]    # Link용 1개
    
    # --- Register 1 ---
    qml.AmplitudeEmbedding(features=data, wires=reg1, normalize=True)
    
    # 6 Layers Ansatz (Shared params)
    he_ansatz_shared(p1, reg1, depth=LAYERS)
    
    # Inverse Ansatz
    qml.adjoint(he_ansatz_shared)(p2, reg1, depth=LAYERS)
    
    qml.adjoint(qml.AmplitudeEmbedding)(features=data, wires=reg1, normalize=True)
    
    # --- DMP Measurement ---
    m0 = qml.measure(0)
    m1 = qml.measure(1)
    m2 = qml.measure(2)
    
    # --- Register 2 ---
    qml.AmplitudeEmbedding(features=data, wires=reg2, normalize=True)
    he_ansatz_shared(p3, reg2, depth=LAYERS)
    
    # --- Link ---
    def apply_link():
        for wire in reg2:
            qml.RX(p4, wires=wire)
            qml.RY(p4, wires=wire) # Flexibility with same param

    qml.cond((m0 == 0) & (m1 == 0) & (m2 == 0), apply_link)()
    
    return qml.expval(qml.PauliZ(wires=reg2[-1]))

# ==========================================
# 4. Main
# ==========================================
def cost(params, X_batch, y_batch):
    preds = [qksan_circuit(x, params) for x in X_batch]
    return np.mean((y_batch - np.stack(preds)) ** 2)

def accuracy(params, X_data, y_data):
    preds = [qksan_circuit(x, params) for x in X_data]
    pred_labels = np.where(np.sign(preds) >= 0, 1, -1)
    return np.mean(pred_labels == y_data)

def main():
    args = build_parser().parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    X_train, y_train, _, _, X_test, y_test = load_data(args)
    
    # [설정] 11개 파라미터 (논문 Table 2)
    params = np.random.uniform(low=-0.1, high=0.1, size=(11,), requires_grad=True)
    
    opt = qml.NesterovMomentumOptimizer(stepsize=LEARNING_RATE, momentum=GAMMA)
    
    print(f"\nStarting Training (6 Layers, 11 Params with Sharing)...")
    print(f"Data: {DATA_DIM} dim | Layers: {LAYERS} | Total Params: {len(params)}")
    
    for step in range(STEPS):
        batch_idx = np.random.randint(0, len(X_train), (BATCH_SIZE,))
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        params, loss_val = opt.step_and_cost(lambda p: cost(p, X_batch, y_batch), params)
        
        if step % 10 == 0 or step == STEPS - 1:
            train_acc = accuracy(params, X_batch, y_batch)
            print(f"Step {step:3d} | Cost: {loss_val:.4f} | Train Acc: {train_acc:.2%}")
            
    test_acc = accuracy(params, X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    main()