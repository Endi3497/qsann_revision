import argparse
import random
import sys
from pathlib import Path

import pennylane as qml
from pennylane import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import torch

CURRENT_DIR = Path(__file__).resolve().parent
REVISION_ROOT = CURRENT_DIR.parent
REPO_ROOT = REVISION_ROOT.parent.parent
if str(REVISION_ROOT) not in sys.path:
    sys.path.insert(0, str(REVISION_ROOT))

from data_loader import build_pcam_dataset, build_standard_image_dataset, split_dataset

# ==========================================
# 1. Experimental Settings (Source 578, Tab 2)
# ==========================================
# 논문의 설정값들을 상수로 정의합니다.
N_QUBITS = 4            # Fig 6: q0~q3
N_FEATURES = 4          # 2 Qubits per register -> 2^2 = 4 amplitudes
LEARNING_RATE = 0.09    # Tab 2
BATCH_SIZE = 30         # Tab 2
STEPS = 120             # Tab 2
GAMMA = 0.9             # Nesterov Momentum Term
CLASSES = [0, 1]        # Binary Classification
DEFAULT_DATASET_CHOICE = "mnist"
DEFAULT_DATASET_LABELS = CLASSES
DEFAULT_SAMPLES_PER_LABEL = 550
DEFAULT_MEDMNIST_SUBSET = "organamnist"
DEFAULT_IMAGE_SIZE = 28
TRAIN_RATIO = 10/11
VAL_RATIO = 0.0
TEST_RATIO = 1/11
DEFAULT_PCAM_ROOT = REPO_ROOT / "data" / "camelyon" / "3channel"

# Device setup (Ideal simulation)
dev = qml.device("default.qubit", wires=N_QUBITS + 2)

# ==========================================
# 2. Data Preparation (Source 457-462)
# ==========================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QKSAN reference using shared QSANN data loader")
    parser.add_argument(
        "--dataset-choice",
        type=str,
        default=DEFAULT_DATASET_CHOICE,
        choices=["mnist", "fmnist", "cifar10", "medmnist", "pcam"],
        help="Source dataset handled by QSANN data_loader.",
    )
    parser.add_argument(
        "--dataset-labels",
        type=int,
        nargs="*",
        default=DEFAULT_DATASET_LABELS,
        help="Two labels to keep (binary); required for torchvision datasets.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=DEFAULT_SAMPLES_PER_LABEL,
        help="Per-label cap; set to <=0 to keep all samples.",
    )
    parser.add_argument(
        "--medmnist-subset",
        type=str,
        default=DEFAULT_MEDMNIST_SUBSET,
        choices=["pathmnist", "dermamnist", "retinamnist", "bloodmnist", "organamnist"],
        help="Required when dataset-choice=medmnist.",
    )
    parser.add_argument(
        "--pcam-root",
        type=Path,
        default=DEFAULT_PCAM_ROOT,
        help="Root folder containing PCam/Camelyon16 class0/class1 .npy patches.",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Resize target for images.")
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=42, help="Seed used for dataset split and parameter init.")
    return parser


def load_data(args: argparse.Namespace):
    print("Loading data via QSANN shared data loader...")
    dataset_choice = args.dataset_choice.lower()
    samples_per_label = args.samples_per_label if args.samples_per_label and args.samples_per_label > 0 else None

    if dataset_choice != "pcam":
        if args.dataset_labels is None or len(args.dataset_labels) != 2:
            raise ValueError("QKSAN is binary; provide exactly two --dataset-labels.")
        if dataset_choice == "medmnist" and args.medmnist_subset is None:
            raise ValueError("--medmnist-subset is required when dataset-choice=medmnist.")
        full_dataset = build_standard_image_dataset(
            dataset_choice=dataset_choice,
            image_size=args.image_size,
            dataset_labels=args.dataset_labels,
            samples_per_label=samples_per_label,
            medmnist_subset=args.medmnist_subset,
        )
    else:
        full_dataset = build_pcam_dataset(
            image_size=args.image_size,
            samples_per_class=samples_per_label,
            root=args.pcam_root,
            seed=args.seed,
        )

    labels = np.array(getattr(full_dataset, "labels", []), dtype=int)
    unique_labels = sorted(set(labels.tolist()))
    if len(unique_labels) != 2:
        raise ValueError(f"Binary QKSAN expects two classes; got labels {unique_labels}.")

    # Flatten all images once, then reuse the same indices as QSANN's split_dataset.
    flattened_images = []
    for img, _, _ in full_dataset:
        img_np = img.detach().cpu().numpy() if torch.is_tensor(img) else np.array(img)
        flattened_images.append(img_np.reshape(-1))
    X_raw = np.asarray(flattened_images, dtype=np.float32)

    # PCA + L2 normalization to prepare amplitude-encoded vectors.
    pca = PCA(n_components=N_FEATURES, random_state=args.seed)
    X_pca = pca.fit_transform(X_raw)
    normalizer = Normalizer()
    X_norm = normalizer.fit_transform(X_pca).astype(np.float64)
    y_signed = np.where(labels == unique_labels[0], -1, 1)

    train_set, val_set, test_set = split_dataset(
        full_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_idx, val_idx, test_idx = (
        np.array(train_set.indices, dtype=int),
        np.array(val_set.indices, dtype=int),
        np.array(test_set.indices, dtype=int),
    )
    X_train, y_train = X_norm[train_idx], y_signed[train_idx]
    X_val, y_val = X_norm[val_idx], y_signed[val_idx]
    X_test, y_test = X_norm[test_idx], y_signed[test_idx]

    print(
        f"Dataset '{dataset_choice}' -> total={len(full_dataset)} | "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} | seed={args.seed}"
    )
    print(f"Labels kept={unique_labels}, samples_per_label={samples_per_label or 'all'}, image_size={args.image_size}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==========================================
# 3. QKSAN Model Definition (Source 362-388)
# ==========================================

# Ansatz: Hardware Efficient (Source 263, Eq 19 & Fig 3b)
# Parameter fitting: Tab 2 says 11 params total.
# We define a lightweight ansatz to stay close to this number.
# Structure:
# - U(theta1) on Reg1 (q0, q1)
# - U(theta3) on Reg2 (q2, q3)
# - Link(theta4)
# Params: 11 total. Let's allocate:
# - Theta1: 5 params
# - Theta2 (inverse of Theta1 usually, or separate): In paper Eq 16/17 they are distinct.
#   However, to keep params low, we might use a simple layer.
#   Let's use 'StronglyEntanglingLayers' or 'BasicEntanglerLayers' with 1 layer.
#   1 layer on 2 qubits = 2 params (rotations) or 6 params (Strongly).
#   Let's define a custom HE Ansatz: RY -> CZ -> RY
def he_ansatz(params, wires):
    # Hardware Efficient Ansatz: Ry - Entangle - Ry
    # Params shape: (Layers, Wires)
    for i, wire in enumerate(wires):
        qml.RY(params[0, i], wires=wire)
        qml.RZ(params[1, i], wires=wire) # Adding RZ based on Fig 3b
    
    # Entanglement
    qml.CZ(wires=[wires[0], wires[1]]) # Simplified entanglement
    
    # If we need more depth, repeat. For 11 total params, we need to be frugal.

# Define the QNode
@qml.qnode(dev, interface="autograd")
def qksan_circuit(data_i, data_j, params):
    # data_i: w_i, data_j: w_j
    # params: flattened parameter array
    
    # Register Definitions
    reg1 = [0, 1]
    reg2 = [2, 3]
    
    # --- Parameter Unpacking ---
    # We aim for ~11 params.
    # Let's assign:
    # theta1 (Reg1): 4 params (2 qubits * 2 gates)
    # theta2 (Reg1 inverse): 4 params
    # theta3 (Reg2): 2 params (2 qubits * 1 gate) ? 
    # theta4 (Link): 1 param
    # Total = 11.
    
    p1 = params[0:4].reshape(2, 2)
    p2 = params[4:8].reshape(2, 2)
    p3 = params[8:10].reshape(1, 2) # Simpler ansatz for value
    p4 = params[10]

    # --- Step 1 & 2: Encoding & Ansatz on Reg 1 ---
    # U_phi(w_i) on Reg 1
    qml.AmplitudeEmbedding(features=data_i, wires=reg1, normalize=True)
    
    # U(theta1)
    he_ansatz(p1, reg1)
    
    # U^dag(theta2) (Inverse Ansatz)
    qml.adjoint(he_ansatz)(p2, reg1)
    
    # U_phi^dag(w_j) on Reg 1
    qml.adjoint(qml.AmplitudeEmbedding)(features=data_j, wires=reg1, normalize=True)

    # --- Measurement M1 for DMP ---
    # In PennyLane, we can't measure and continue easily in 'default.qubit' without 'qml.cond'.
    # We simulate DMP by applying the conditional gate based on the probability of |0> state?
    # No, true DMP uses the measurement result.
    # qml.cond allows applying an operation conditioned on a measurement value.
    m1 = qml.measure(wires=0) # Measuring q0 and q1? Paper says "Ground state |0>_1"
    # Note: Reg1 has 2 qubits. "Ground state" usually means |00>.
    # For simplicity and to match Fig 4 (one measurement symbol), we measure the register.
    # But qml.measure measures a single qubit. Let's measure both.
    m1_0 = qml.measure(0)
    m1_1 = qml.measure(1)
    
    # --- Step 2 (cont): Encoding & Ansatz on Reg 2 ---
    # U_phi(w_j) on Reg 2
    qml.AmplitudeEmbedding(features=data_j, wires=reg2, normalize=True)
    
    # U(theta3)
    # he_ansatz with only 1 layer of RY (to match param count)
    for i, wire in enumerate(reg2):
        qml.RY(p3[0, i], wires=wire)

    # --- Step 3: DMP Conditional Operation ---
    # Eq 22: CU_DMP applied if Reg1 is measured as |0>.
    # Condition: m1_0 == 0 AND m1_1 == 0
    def apply_link():
        # Source 294, Eq 22: CR_Y and R_X.
        # Simplified to Controlled-Rotation for Link
        qml.RX(p4, wires=2)
        qml.RX(p4, wires=3)
        # Note: The paper mentions CR_Y spanning registers, but R_X on Reg1?
        # Eq 22 is complex. We approximate the "Control" mechanism:
        # If Reg1 is 0, rotate Reg2.

    qml.cond((m1_0 == 0) & (m1_1 == 0), apply_link)()

    # --- Step 4: Measurement M2 ---
    # Source 426: Expectation value E = <Phi|P|Phi>
    # Source 455: Measurement on q3 is regarded as predictive labels.
    # We return expectation of PauliZ on q3 (mapped to [-1, 1])
    return qml.expval(qml.PauliZ(wires=3))


# ==========================================
# 4. Optimization Loop (Source 430-441)
# ==========================================

def cost(params, X_batch, y_batch):
    # Source 29: Cost function = Sum [y_i - sgn(E)]^2
    # But sgn is non-differentiable. Usually replaced by MSE or similar surrogate in training.
    # The paper says "sgn(E)" in Eq 29 but usually for gradient based training 
    # one uses (y - E)^2 or similar. 
    # Given Eq 29 explicitly has sgn, but they use gradient descent...
    # We will use MSE on the raw expectation value to ensure differentiability, 
    # which is standard practice when the formula says sgn(E) for *classification error* # but smooth loss for *training*. Or they might mean Softsign.
    # Here we stick to MSE on Expectation: (y - E)^2
    predictions = [qksan_circuit(x, x, params) for x in X_batch] # Self-attention: x_i with x_i
    loss = np.mean((y_batch - np.stack(predictions)) ** 2)
    return loss

def accuracy(params, X_data, y_data):
    predictions = [qksan_circuit(x, x, params) for x in X_data]
    # Sign function for classification
    predicted_labels = np.sign(predictions)
    # Map 0 to 1 (if sign is 0) or handle strictly
    predicted_labels = np.where(predicted_labels >= 0, 1, -1)
    return np.mean(predicted_labels == y_data)

def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args)
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train/test splits are empty; check dataset and sampling settings.")

    params = np.random.uniform(low=-np.pi, high=np.pi, size=(11,), requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=LEARNING_RATE, momentum=GAMMA)
    print("\nStarting Training with Nesterov Momentum Optimizer...")
    print(f"Total Parameters: {len(params)}")

    for step in range(STEPS):
        batch_index = np.random.randint(0, len(X_train), (BATCH_SIZE,))
        X_batch = X_train[batch_index]
        y_batch = y_train[batch_index]
        params, loss_val = opt.step_and_cost(lambda p: cost(p, X_batch, y_batch), params)

        if step % 10 == 0 or step == STEPS - 1:
            train_acc = accuracy(params, X_batch, y_batch)
            print(f"Step {step:3d} | Cost: {loss_val:.4f} | Batch Acc: {train_acc:.2%}")

    test_acc = accuracy(params, X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.2%}")
    if len(y_val) > 0:
        val_acc = accuracy(params, X_val, y_val)
        print(f"Validation Accuracy: {val_acc:.2%}")


if __name__ == "__main__":
    main()
