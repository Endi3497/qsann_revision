# 2 * n * Denc = patch_area
# 하이퍼파라미터 설정 블록
# Denc: 인코딩 반복 수(스테이지 수)
# D   : 변분(Ansatz) 깊이
Denc = 6
D = 2
S = 256           # 패치 수 (32x32 그리드)
n = 4              # 쿼빗 수
num_layers = 1     # QSAL 레이어 수
learning_rate = 0.01
patch_size = (4, 4)  # 패치 크기
patch_side_size = 4
patch_area = 16
side_patch_number = 16
patch_num = 256
patch_shape = (16, 16)
title = '64x64_with_4x4_patch_test_bs16_patients'
left_minus = 24
upper_minus = 24
right_plus = 48
lower_plus= 48
epoch_num= 10
batch_size = 16
image_size = 64
import random




import torch
torch.set_num_threads(8)  # 원한다면 2~8 정도로 설정
torch.set_num_interop_threads(8)

random.seed(42)
torch.manual_seed(42)



# %%
import sys
sys.stdout = open(f'/home/junyeollee/.jupyter/mets/log_data/{title}.txt', 'w')
sys.stderr = sys.stdout  # 에러도 같은 파일에 저장
import warnings
from numpy import ComplexWarning  
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1
    _HAS_SGK = True
except Exception:
    StratifiedGroupKFold = None
    _HAS_SGK = False
warnings.filterwarnings("ignore", category=ComplexWarning)


# %%
import pennylane as qml
import torch
import numpy as np
import random
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import Counter

from sklearn.metrics import roc_auc_score
# %%
import torch
torch.cuda.set_device(1)  # GPU 1번 사용
device = torch.device("cpu")
from sklearn.model_selection import KFold
# %%
import functools
import inspect
import math
from collections.abc import Iterable
from typing import Callable, Dict, Union, Any, Optional
from sklearn.metrics import precision_recall_fscore_support, average_precision_score 
import torch.nn.functional as F
from pennylane import QNode
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets as tv_datasets, transforms as tv_transforms

try:
    import torch
    from torch.nn import Module

    TORCH_IMPORTED = True
except ImportError:
    # The following allows this module to be imported even if PyTorch is not installed. Users
    # will instead see an ImportError when instantiating the TorchLayer.
    from unittest.mock import Mock

    Module = Mock
    TORCH_IMPORTED = False


class TorchLayer(Module):
    def __init__(self,qnode,weights):
        if not TORCH_IMPORTED:
            raise ImportError(
                "TorchLayer requires PyTorch. PyTorch can be installed using:\n"
                "pip install torch\nAlternatively, "
                "visit https://pytorch.org/get-started/locally/ for detailed "
                "instructions."
            )
        super().__init__()

        #weight_shapes = {
        #    weight: (tuple(size) if isinstance(size, Iterable) else () if size == 1 else (size,))
        #    for weight, size in weight_shapes.items()
        #}

        # validate the QNode signature, and convert to a Torch QNode.
        # TODO: update the docstring regarding changes to restrictions when tape mode is default.
        #self._signature_validation(qnode, weight_shapes)
        self.qnode = qnode
        self.qnode.interface = "torch"

        self.q_weight = nn.Parameter(weights["weights"])  # 🔥 핵심
        self._input_arg = "inputs"

    def forward(self, inputs, loc_angles=None, texture_angles=None):  # pylint: disable=arguments-differ
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """

        # 브로드캐스트/배치 입력을 그대로 QNode에 전달
        return self._evaluate_qnode(inputs, loc_angles, texture_angles)


    def _evaluate_qnode(self, x, loc_angles=None, texture_angles=None):
        kwargs = {
            self._input_arg: x,
            "weights": self.q_weight.to(x.device),
        }
        if loc_angles is not None:
            kwargs["loc_angles"] = loc_angles.to(x.device)
        if texture_angles is not None:
            kwargs["texture_angles"] = texture_angles.to(x.device)
        res = self.qnode(**kwargs)

        # 배치 모드: QNode가 (N, d) 텐서를 반환하면 그대로 사용
        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)

        # 리스트 반환 시: 브로드캐스트(각 항목이 (N,))이면 마지막 차원으로 stack
        if isinstance(res, (list, tuple)) and len(res) > 0:
            if isinstance(res[0], torch.Tensor) and res[0].ndim >= 1:
                return torch.stack(list(res), dim=-1).type(x.dtype)
            else:
                return torch.hstack(list(res)).type(x.dtype)

        # fallback
        return torch.tensor(res, dtype=x.dtype, device=x.device)

    def __str__(self):
        detail = "<Quantum Torch Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__
    _input_arg = "inputs"

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Torch layer. Set to ``"inputs"``."""
        return self._input_arg


# (32×32) 패치 그리드의 (x,y) 정규화 좌표 (시각화 제거, 좌표만 보관)
patch_side = patch_shape[0]
pos_coords = []
for r in range(patch_side):
    for c in range(patch_side):
        pos_coords.append([r/(patch_side-1), c/(patch_side-1)])
pos_coords = torch.tensor(pos_coords)  # (S, 2)

# %%
def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

# %%
class QSAL_pennylane(torch.nn.Module):
    def __init__(self, S, n, Denc, D):
        super().__init__()
        self.seq_num = S
        self.num_pixel_qubits = n
        # Only pixel qubits: drop location and texture qubits
        self.num_q = self.num_pixel_qubits
        self.Denc = Denc
        self.D = D
        # variational 파라미터 개수: (초기 RX/RY 2개 + 각 레이어 RY 1개) * num_q
        self.d = (2 + self.D) * self.num_q
        print(f"[OptimizedQSAL __init__] num_q = {self.num_q}, d = {self.d}")

        # ▶ 공유할 Q/K/V 파라미터
        self.init_params_Q = nn.Parameter((np.pi/4) * (2 * torch.randn(self.d) - 1))
        self.init_params_K = nn.Parameter((np.pi/4) * (2 * torch.randn(self.d) - 1))
        self.init_params_V = nn.Parameter((np.pi/4) * (2 * torch.randn(self.d) - 1))


        # ▶ 양자 디바이스와 QNode 1개씩
        self.dev     = qml.device("lightning.qubit", wires=self.num_q)
        self.qnode_v = qml.QNode(self.circuit_v,  self.dev, interface="torch", diff_method="adjoint")
        self.qnode_q = qml.QNode(self.circuit_qk, self.dev, interface="torch", diff_method="adjoint")
        self.qnode_k = self.qnode_q  # still shared

        # ▶ TorchLayer도 하나씩만 생성
        self.to_Q = TorchLayer(self.qnode_q, {"weights": self.init_params_Q})
        self.to_K = TorchLayer(self.qnode_k, {"weights": self.init_params_K})
        self.to_V = TorchLayer(self.qnode_v, {"weights": self.init_params_V})

        # 나머지
        self.alpha             = None
        for p in self.parameters():
            p.data = p.data.to(torch.device("cpu"))
        self.register_buffer('pos_coords', self._create_pos_coords())
    # circuit_v, circuit_qk, forward 등은 기존과 동일하게…

    def _create_pos_coords(self):
        """위치 좌표를 미리 계산하여 버퍼에 저장"""
        patch_side = int(math.isqrt(self.seq_num)) if hasattr(math, 'isqrt') else int(np.sqrt(self.seq_num))
        pos_coords = []
        for r in range(patch_side):
            for c in range(patch_side):
                pos_coords.append([r/(patch_side-1), c/(patch_side-1)])
        return torch.tensor(pos_coords)  # (S, 2)


    def random_op(self):
    
        # 무작위 시드 고정
        set_random_seed(42)  # 원하는 시드 값 설정
        a=random.randint(0, 4)
        if a==0:
            op=qml.Identity(0)
        elif a==1:
            op=qml.PauliX(0)
        elif a==2:
            op=qml.PauliY(0)
        else:
            op=qml.PauliZ(0)

        op_elimated=qml.Identity(0)
        for i in range(1,self.num_q):
            op_elimated=op_elimated@qml.Identity(i)
        Select_wrong=True
        while Select_wrong:
            for i in range(1,self.num_q):
                a=random.randint(0, 4)
                if a==0:
                    op=op@qml.Identity(i)
                elif a==1:
                    op=op@qml.PauliX(i)
                elif a==2:
                    op=op@qml.PauliY(i)
                else:
                    op=op@qml.PauliZ(i)
            if op!=op_elimated:
                Select_wrong=False
        return op

    def circuit_v(self, inputs, weights, loc_angles=None, texture_angles=None):
        # Denc 반복 인코딩: 한 스테이지당 큐빗별 RX/RY 2각 → (2*num_q)개 소비
        # 전체 소비 입력 개수 = 2 * num_q * Denc
        s = (inputs / 255.0) * math.pi
        enc_len = 2 * self.num_pixel_qubits * self.Denc
        # 입력 길이와 불일치 시 안전 처리 (자르는 쪽이 기본)
        if s.shape[-1] > enc_len:
            s = s[..., :enc_len]
        elif s.shape[-1] < enc_len:
            pad = enc_len - s.shape[-1]
            s = F.pad(s, (0, pad), mode="constant", value=0.0)

        # 인코딩 스테이지 반복, 스테이지 사이에 링 얽힘
        for stage in range(self.Denc):
            base = 2 * self.num_pixel_qubits * stage
            for q in range(self.num_pixel_qubits):
                qml.RX(s[..., base + 2*q], q)
                qml.RY(s[..., base + 2*q + 1], q)
            if stage != self.Denc - 1:
                for q in range(self.num_pixel_qubits):
                    qml.CNOT(wires=(q, (q + 1) % self.num_pixel_qubits))
        # Ansatz
        indx = 0
        for j in range(self.num_q):
            if indx < weights.shape[0]:
                qml.RX(weights[indx], j)
            if indx + 1 < weights.shape[0]:
                qml.RY(weights[indx + 1], j)
            indx += 2
        for i in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                if indx < weights.shape[0]:
                    qml.RY(weights[indx], j)
                    indx += 1
        # Measure Pauli X, Y, Z on all qubits (4*3 = 12-dim)
        expvals = []
        for i in range(self.num_q):
            expvals.append(qml.expval(qml.PauliX(i)))
            expvals.append(qml.expval(qml.PauliY(i)))
            expvals.append(qml.expval(qml.PauliZ(i)))
        return expvals

    def circuit_qk(self, inputs, weights, loc_angles=None, texture_angles=None):
        # 동일하게 Denc 반복 인코딩 적용
        s = (inputs / 255.0) * math.pi
        enc_len = 2 * self.num_pixel_qubits * self.Denc
        if s.shape[-1] > enc_len:
            s = s[..., :enc_len]
        elif s.shape[-1] < enc_len:
            pad = enc_len - s.shape[-1]
            s = F.pad(s, (0, pad), mode="constant", value=0.0)

        for stage in range(self.Denc):
            base = 2 * self.num_pixel_qubits * stage
            for q in range(self.num_pixel_qubits):
                qml.RX(s[..., base + 2*q], q)
                qml.RY(s[..., base + 2*q + 1], q)
            if stage != self.Denc - 1:
                for q in range(self.num_pixel_qubits):
                    qml.CNOT(wires=(q, (q + 1) % self.num_pixel_qubits))
        # Ansatz
        indx = 0
        for j in range(self.num_q):
            if indx < weights.shape[0]:
                qml.RX(weights[indx], j)
            if indx + 1 < weights.shape[0]:
                qml.RY(weights[indx + 1], j)
            indx += 2
        for i in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                if indx < weights.shape[0]:
                    qml.RY(weights[indx], j)
                    indx += 1
        # Measure Pauli X, Y, Z on all qubits (4*3 = 12-dim)
        expvals = []
        for i in range(self.num_q):
            expvals.append(qml.expval(qml.PauliX(i)))
            expvals.append(qml.expval(qml.PauliY(i)))
            expvals.append(qml.expval(qml.PauliZ(i)))
        return expvals



    def forward(self, input):
        """최적화된 forward 패스 - 배치 처리 (메모리 효율 + 마이크로배칭)"""
        device = self.to_Q.q_weight.device
        input = input.to(device)
        B, S, d_in = input.shape

        # (B*S, d_in)로 평탄화
        flat_input = input.view(B * S, d_in)

        # 마이크로배칭 유틸
        def _run_mb(func, x, mb: int = 4096):
            outs = []
            for i in range(0, x.shape[0], mb):
                outs.append(func(x[i : i + mb]))
            return torch.cat(outs, dim=0)

        # Q, K, V 계산 (마이크로배칭)
        Q_flat = _run_mb(self.to_Q, flat_input)
        K_flat = _run_mb(self.to_K, flat_input)
        V_flat = _run_mb(self.to_V, flat_input)

        # 원래 형태로 복원
        Q = Q_flat.view(B, S, -1)
        K = K_flat.view(B, S, -1)
        V = V_flat.view(B, S, -1)

        # 메모리 효율적 어텐션: ||q-k||^2 = ||q||^2 + ||k||^2 - 2 q·k
        QQ = (Q * Q).sum(-1).unsqueeze(2)      # (B, S, 1)
        KK = (K * K).sum(-1).unsqueeze(1)      # (B, 1, S)
        QK = torch.bmm(Q, K.transpose(1, 2))   # (B, S, S)
        dist2 = QQ + KK - 2.0 * QK
        self.alpha = torch.exp(-dist2)
        self.alpha = self.alpha / (self.alpha.sum(dim=-1, keepdim=True) + 1e-12)

        # 어텐션 적용
        output = torch.bmm(self.alpha, V)
        return output.to(device)

class QSANN_pennylane(torch.nn.Module):
    def __init__(self,S,n,Denc,D,num_layers):
        """
        # input: input data
        # weight: trainable parameter
        # n: # of of qubits
        # d: embedding dimension which is equal to n(Denc+2)
        # Denc: the # number of layers for encoding 
        # D: the # of layers of variational layers
        # type "K": key, "Q": Query, "V": value
        """
        super().__init__()
        self.qsal_lst=[QSAL_pennylane(S,n,Denc,D) for _ in range(num_layers)]
        self.qnn=nn.Sequential(*self.qsal_lst)

    def forward(self,input):
        return self.qnn(input)

class QSANN_text_classifier(torch.nn.Module):
    def __init__(self,S,n,Denc,D,num_layers):
        """
        # input: input data
        # weight: trainable parameter
        # n: # of of qubits
        # d: embedding dimension which is equal to n(Denc+2)
        # Denc: the # number of layers for encoding 
        # D: the # of layers of variational layers
        # type "K": key, "Q": Query, "V": value
        """
        super().__init__()
        self.Qnn = QSANN_pennylane(S, n, Denc, D, num_layers)
        d_model = self.Qnn.qsal_lst[0].d
        embed_dim = 3 * n
        self.final_layer = nn.Linear(S * embed_dim, 1)
        self.final_layer = self.final_layer.float()


    def forward(self, input, return_attention=False):
        x=self.Qnn(input)
        if return_attention:
            attention = self.Qnn.qsal_lst[-1].alpha 
        x=torch.flatten(x,start_dim=1)
        # print('done2')
        output = self.final_layer(x)
        # print('done3')
        if return_attention:
            return output, attention
        return output


from PIL import Image
import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import ast
from torchvision.datasets import ImageFolder
import os
import glob
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset


import os
import ast
import random
import pandas as pd
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, ConcatDataset, DataLoader

def _extract_patient_id_from_path(path: str) -> str:
    """Extract patient ID from a filename like '12345_Tumor_1.png'.
    Assumes patient ID is the part before the first underscore.
    """
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name.split('_')[0]

def train_and_eval_with_tensors(X: torch.Tensor, y: torch.Tensor):
    """
    외부에서 준비된 64x64 단일채널 또는 3채널 이미지 텐서 X와 레이블 y(N)을 받아
    패치 분할 후 QSANN 학습/평가를 수행합니다. 전처리/시각화는 포함하지 않습니다.

    X: torch.float32 [N, 64, 64] 또는 [N, 64, 64, 3]
    y: torch.int (0/1) [N]
    """
    assert (
        (X.ndim == 3 and X.shape[1:] == (image_size, image_size)) or
        (X.ndim == 4 and X.shape[1:] == (image_size, image_size, 3))
    ), "X must be [N,64,64] or [N,64,64,3]"
    assert y.ndim == 1 and X.shape[0] == y.shape[0], "y must be [N]"

    device = torch.device("cpu")
    n_splits = 5

    # 결과 저장용 리스트
    all_metrics = []

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        torch.cuda.empty_cache()
        print(f"\n========== Fold {fold + 1} ==========")
        X_test_raw = [X[idx] for idx in test_idx] 
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # numpy 변환
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.numpy()
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.numpy()

        # 클래스 분포 확인
        unique, counts = np.unique(y_test, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(class_distribution)

        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"[Fold {fold+1}] Class distribution: {class_distribution}")

        # 패치 분할
        X_train = np.array([split_into_non_overlapping_patches(img.numpy()) for img in X_train])
        X_test = np.array([split_into_non_overlapping_patches(img.numpy()) for img in X_test])

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # 모델 초기화
        model = QSANN_text_classifier(S, n, Denc, D, num_layers).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        # 파라미터 수 출력
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")

        # 디버깅용 기능은 성능 저하를 유발하므로 비활성화
        torch.autograd.set_detect_anomaly(False)
        for epoch in tqdm(range(epoch_num), desc=f"Training Fold {fold+1}"):
            model.train()

            total_loss = 0
            train_preds = []
            train_labels = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)

                probs = torch.sigmoid(predictions).detach().cpu().numpy()
                labels = y_batch.detach().cpu().numpy()
                train_preds.extend(probs)
                train_labels.extend(labels)

            epoch_loss = total_loss / len(train_loader.dataset)
            epoch_preds = (np.array(train_preds) >= 0.5).astype(int)
            epoch_labels = np.array(train_labels)

            try:
                epoch_auroc = roc_auc_score(epoch_labels, train_preds)
            except ValueError:
                epoch_auroc = float('nan')

            epoch_acc = (epoch_preds == epoch_labels).mean()

            print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | AUROC: {epoch_auroc:.4f}")

        # 모델 저장/로딩
        model_save_path = f"/home/junyeollee/.jupyter/mets/model_results/{title}_fold{fold+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        # 평가
        y_true = y_test 
        probs = optimized_evaluation(model, X_test_tensor, device)
        probs = np.array(probs)

        threshold = 0.5
        print(f"Fold {fold+1} Using fixed threshold: {threshold}")
        y_pred = (probs >= threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, probs)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auprc = average_precision_score(y_true, probs)

        print(f"Fold {fold+1} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUROC: {auroc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        all_metrics.append({
            "Accuracy": acc,
            "AUROC": auroc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUPRC": auprc
        })

    print("\n========== Average Results across 5 folds ==========")
    for metric in ["Accuracy", "AUROC", "Precision", "Recall", "F1", "AUPRC"]:
        avg = np.mean([m[metric] for m in all_metrics])
        print(f"{metric}: {avg:.4f}")
    return all_metrics



# %%
# 2x2 크기로 겹치지 않게 패치로 나누는 함수
def split_into_non_overlapping_patches(image, patch_size=patch_size):
    patches = []
    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]].flatten()
            patches.append(patch)
    return np.array(patches)





# %%
def binary_accuracy(preds, y):
    probs = torch.sigmoid(preds)        # 1. logit → 확률로
    rounded_preds = torch.round(probs)  # 2. 확률 → 0 or 1 (클래스 예측)
    correct = (rounded_preds == y).float()
    return correct.sum() / len(y)


def optimized_evaluation(model, X_test_tensor, device, eval_batch_size: int = 32):
    """메모리 효율적인 미니배치 평가."""
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, X_test_tensor.shape[0], eval_batch_size):
            xb = X_test_tensor[i : i + eval_batch_size].to(device)
            outputs = model(xb)
            probs_all.append(torch.sigmoid(outputs).cpu())
    return torch.cat(probs_all, dim=0).numpy().flatten()



# 새 데이터셋 로더: class0/class1에서 RGB로 읽고 64x64로 맞춘 후 [N,64,64,3] 반환
def load_from_class_folders_rgb(root: str = "/home/junyeollee/.jupyter/QSANN/data/Patches(4down)",
                                per_class_limit: Optional[int] = None,
                                seed: int = 42,
                                return_groups: bool = False):
    classes = [("class0", 0), ("class1", 1)]
    Xs, ys = [], []
    groups: list[str] = []
    rng = np.random.default_rng(seed)

    for cname, label in classes:
        cdir = os.path.join(root, cname)
        if not os.path.isdir(cdir):
            continue
        # 이미지 파일 수집
        files = []
        for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            files.extend(glob.glob(os.path.join(cdir, "**", pat), recursive=True))
        if per_class_limit and len(files) > per_class_limit:
            files = list(rng.choice(files, per_class_limit, replace=False))

        for f in files:
            try:
                img = Image.open(f).convert("RGB")
                if img.size != (image_size, image_size):
                    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
                arr = np.array(img, dtype=np.float32)  # 0..255 범위 유지
                Xs.append(arr)
                ys.append(label)
                if return_groups:
                    groups.append(_extract_patient_id_from_path(f))
            except Exception:
                # 손상 파일 등은 건너뜀
                continue

    if not Xs:
        raise RuntimeError(f"No images found under {root}/class0 and {root}/class1")

    X = torch.tensor(np.stack(Xs, axis=0), dtype=torch.float32)  # [N,64,64,3]
    y = torch.tensor(ys, dtype=torch.int64)
    if return_groups:
        return X, y, groups
    return X, y


def train_and_eval_with_tensors_grouped(X: torch.Tensor, y: torch.Tensor, groups: list, n_splits: int = 5):
    """
    Group-aware CV: prevents same patient from appearing in both train and test.
    Uses StratifiedGroupKFold if available, otherwise GroupKFold.
    """
    assert (
        (X.ndim == 3 and X.shape[1:] == (image_size, image_size)) or
        (X.ndim == 4 and X.shape[1:] == (image_size, image_size, 3))
    ), "X must be [N,64,64] or [N,64,64,3]"
    assert y.ndim == 1 and X.shape[0] == y.shape[0], "y must be [N]"
    assert len(groups) == X.shape[0], "groups length must match N"

    device = torch.device("cpu")
    uniq = len(set(groups))
    if uniq < 2:
        raise ValueError(f"Need at least 2 unique patients, found {uniq}")
    eff_splits = min(n_splits, uniq)

    all_metrics = []

    if _HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=eff_splits, shuffle=True, random_state=42)
        splits = splitter.split(X, y, groups=groups)
        split_name = "StratifiedGroupKFold"
    else:
        splitter = GroupKFold(n_splits=eff_splits)
        splits = splitter.split(X, y, groups=groups)
        split_name = "GroupKFold"

    for fold, (train_idx, test_idx) in enumerate(splits):
        torch.cuda.empty_cache()
        print(f"\n========== Fold {fold + 1} ({split_name}) ==========")

        # verify no leakage
        train_pat = set(groups[i] for i in train_idx)
        test_pat = set(groups[i] for i in test_idx)
        inter = train_pat & test_pat
        if inter:
            raise RuntimeError(f"Patient leakage in fold {fold+1}: {sorted(list(inter))[:5]} ...")
        print(f"Unique patients — train: {len(train_pat)}, test: {len(test_pat)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if isinstance(y_train, torch.Tensor):
            y_train = y_train.numpy()
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.numpy()

        unique, counts = np.unique(y_test, return_counts=True)
        print(dict(zip(unique, counts)))
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"[Fold {fold+1}] Class distribution: {dict(zip(unique, counts))}")

        # patchify
        X_train = np.array([split_into_non_overlapping_patches(img.numpy()) for img in X_train])
        X_test = np.array([split_into_non_overlapping_patches(img.numpy()) for img in X_test])

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        model = QSANN_text_classifier(S, n, Denc, D, num_layers).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")

        torch.autograd.set_detect_anomaly(False)
        for epoch in tqdm(range(epoch_num), desc=f"Training Fold {fold+1}"):
            model.train()
            total_loss = 0
            train_preds, train_labels = [], []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(preds).detach().cpu().numpy()
                labels = y_batch.detach().cpu().numpy()
                train_preds.extend(probs)
                train_labels.extend(labels)

            epoch_loss = total_loss / len(train_loader.dataset)
            epoch_preds = (np.array(train_preds) >= 0.5).astype(int)
            epoch_labels = np.array(train_labels)
            try:
                epoch_auroc = roc_auc_score(epoch_labels, train_preds)
            except ValueError:
                epoch_auroc = float('nan')
            epoch_acc = (epoch_preds == epoch_labels).mean()
            print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | AUROC: {epoch_auroc:.4f}")

        # 모델 저장/로딩 (그룹 분할 버전에도 적용)
        os.makedirs("/home/junyeollee/.jupyter/mets/model_results", exist_ok=True)
        model_save_path = f"/home/junyeollee/.jupyter/mets/model_results/{title}_fold{fold+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"[Fold {fold+1}] Saved model to: {model_save_path}")
        # 필요 시 로드 검증
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        probs = optimized_evaluation(model, X_test_tensor, device)
        probs = np.array(probs)
        y_pred = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, probs)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auprc = average_precision_score(y_test, probs)

        print(f"Fold {fold+1} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUROC: {auroc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        all_metrics.append({
            "Accuracy": acc,
            "AUROC": auroc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUPRC": auprc
        })

    print("\n========== Average Results across folds ==========")
    for metric in ["Accuracy", "AUROC", "Precision", "Recall", "F1", "AUPRC"]:
        avg = np.mean([m[metric] for m in all_metrics])
        print(f"{metric}: {avg:.4f}")
    return all_metrics


if __name__ == "__main__":
    # RGB 데이터셋(class0/class1) 사용 + 환자 단위 분할 보장
    DATA_ROOT = "/home/junyeollee/.jupyter/QSANN/data/Patches(4down)"
    X, y, groups = load_from_class_folders_rgb(DATA_ROOT, return_groups=True)
    print(
        f"Loaded QSANN RGB patches: X={tuple(X.shape)}, y={tuple(y.shape)}, "
        f"class1={int((y==1).sum().item())}, class0={int((y==0).sum().item())}"
    )
    # 환자 단위 분할로 학습/평가
    # StratifiedGroupKFold(가능 시) 또는 GroupKFold로 교차검증 수행
    unique_patients = len(set(groups))
    print(f"Total unique patients: {unique_patients}")
    
    # 그룹드 버전이 없으면 기존 함수 사용 (비권장)
    try:
        _ = unique_patients  # no-op to avoid lint
        from sklearn.model_selection import GroupKFold as _gk
        # 정의된 grouped 함수가 아래 추가됨
        train_and_eval_with_tensors_grouped  # type: ignore
        train_and_eval_with_tensors_grouped(X, y, groups)
    except NameError:
        print("[WARN] grouped trainer not found; falling back to sample-level split (may leak).")
        train_and_eval_with_tensors(X, y)
