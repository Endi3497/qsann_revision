from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import math
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


def _normalize_image_size(image_size: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a length-2 sequence")
    return (int(image_size[0]), int(image_size[1]))


def _cifar10_rgb_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using standard luminance weights.
    """
    if image_array.ndim == 2:
        gray = image_array.astype(np.float32)
    else:
        gray = np.dot(image_array[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return (gray / 255.0).astype(np.float32)


def _stratified_split_indices(
    labels: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    total = len(labels)
    if total == 0:
        raise ValueError("Dataset is empty; cannot split.")
    s = train_ratio + val_ratio + test_ratio
    if not math.isclose(s, 1.0, rel_tol=1e-3):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")
    if len(set(labels)) < 2:
        raise ValueError("Stratified split requires at least two classes.")

    indices = np.arange(total)
    if val_ratio + test_ratio == 0:
        return indices.tolist(), [], []

    # First split off train vs temp (val+test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    if val_ratio == 0 or test_ratio == 0:
        # All remaining goes to val if test_ratio==0, or vice versa
        if test_ratio == 0:
            return train_idx.tolist(), temp_idx.tolist(), []
        return train_idx.tolist(), [], temp_idx.tolist()

    # Split temp into val/test with relative proportions
    test_share = test_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_share,
        stratify=temp_y,
        random_state=seed,
        shuffle=True,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def _stratified_split_indices_counts(
    labels: Sequence[int],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    total = len(labels)
    if total == 0:
        raise ValueError("Dataset is empty; cannot split.")
    if len(set(labels)) < 2:
        raise ValueError("Stratified split requires at least two classes.")
    if train_count < 0 or val_count < 0 or test_count < 0:
        raise ValueError("train_count, val_count, and test_count must be >= 0.")
    if train_count == 0:
        raise ValueError("train_count must be > 0 for count-based split.")
    total_target = train_count + val_count + test_count
    if total_target == 0:
        raise ValueError("At least one of train_count/val_count/test_count must be > 0.")
    if total_target > total:
        raise ValueError("train_count + val_count + test_count exceeds dataset size.")

    indices = np.arange(total)
    temp_count = val_count + test_count
    if temp_count == 0:
        if train_count == total:
            return indices.tolist(), [], []
        train_idx, _, _, _ = train_test_split(
            indices,
            labels,
            train_size=train_count,
            stratify=labels,
            random_state=seed,
            shuffle=True,
        )
        return train_idx.tolist(), [], []

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels,
        train_size=train_count,
        test_size=temp_count,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    if val_count == 0:
        return train_idx.tolist(), [], temp_idx.tolist()
    if test_count == 0:
        return train_idx.tolist(), temp_idx.tolist(), []

    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_y,
        train_size=val_count,
        test_size=test_count,
        stratify=temp_y,
        random_state=seed,
        shuffle=True,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise ValueError("Dataset must expose a 'labels' attribute for stratified split.")
    train_idx, val_idx, test_idx = _stratified_split_indices(labels, train_ratio, val_ratio, test_ratio, seed)
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def split_dataset_by_counts(
    dataset: Dataset,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise ValueError("Dataset must expose a 'labels' attribute for stratified split.")
    train_idx, val_idx, test_idx = _stratified_split_indices_counts(
        labels, train_count, val_count, test_count, seed
    )
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


class TensorListDataset(Dataset):
    """
    Simple dataset wrapping a list of tensors and integer labels.
    """

    def __init__(self, images: List[torch.Tensor], labels: List[int]) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        if len(images) == 0:
            raise ValueError("images list is empty")
        self.images = images
        self.labels = [int(l) for l in labels]
        self.num_channels = images[0].shape[0]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx], f"sample_{idx}"


class PCamDataset(Dataset):
    """
    PatchCamelyon (Camelyon16) patches stored as .npy under class0/class1 folders.
    Expects data shaped [H, W, C] (C can be 1, 2, or 3); scales to [0, 1].
    """

    CLASS_MAP = [("class0", 0), ("class1", 1)]

    def __init__(
        self,
        root: str | Path,
        image_size: Sequence[int] | int,
        samples_per_class: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.image_size = _normalize_image_size(image_size)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.items = self._gather_items()
        self.labels = [lbl for _, lbl in self.items]
        self.num_channels = self._infer_channels()

    def _gather_items(self) -> List[Tuple[Path, int]]:
        rng = np.random.default_rng(self.seed)
        items: List[Tuple[Path, int]] = []
        for cname, label in self.CLASS_MAP:
            cdir = self.root / cname
            if not cdir.is_dir():
                continue
            paths = sorted(cdir.rglob("*.npy"))
            if self.samples_per_class is not None and len(paths) > self.samples_per_class:
                rng.shuffle(paths)
                paths = paths[: self.samples_per_class]
            for p in paths:
                items.append((p, label))
        if not items:
            raise FileNotFoundError(f"No .npy files found under {self.root}. Expected class0/ and class1/ subfolders.")
        rng.shuffle(items)
        return items

    def _infer_channels(self) -> int:
        sample = np.load(self.items[0][0])
        if sample.ndim == 2:
            return 1
        if sample.ndim == 3:
            return sample.shape[-1]
        raise ValueError(f"Unexpected PCam sample shape {sample.shape}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.shape[2] not in (1, 2, 3):
            raise ValueError(f"Unsupported channel count {arr.shape[2]} in sample {path}")
        img = torch.from_numpy(arr).permute(2, 0, 1).float()
        if img.max() > 1.0:
            img = img / 255.0
        img = TF.resize(img, self.image_size)
        return img, label, path.stem


def _collect_standard_samples(
    datasets_list: Sequence[Dataset],
    allowed_labels: Sequence[int] | None,
    samples_per_label: int | None,
) -> Tuple[List[torch.Tensor], List[int], dict]:
    counts: Counter = Counter()
    images: List[torch.Tensor] = []
    labels: List[int] = []
    label_map = None
    if allowed_labels:
        allowed_sorted = sorted(set(int(l) for l in allowed_labels))
        label_map = {orig: idx for idx, orig in enumerate(allowed_sorted)}
        allowed_set = set(allowed_sorted)
    else:
        allowed_set = None

    for ds in datasets_list:
        for img, lbl in ds:
            if torch.is_tensor(lbl):
                lbl_val = int(lbl.view(-1)[0].item())
            elif isinstance(lbl, (list, tuple, np.ndarray)):
                lbl_val = int(np.array(lbl).reshape(-1)[0].item())
            else:
                lbl_val = int(lbl)
            if allowed_set is not None and lbl_val not in allowed_set:
                continue
            if samples_per_label is not None and counts[lbl_val] >= samples_per_label:
                continue
            mapped_label = label_map[lbl_val] if label_map is not None else lbl_val
            if torch.is_tensor(img):
                img_t = img
            else:
                img_t = transforms.ToTensor()(img)
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)
            images.append(img_t)
            labels.append(mapped_label)
            counts[lbl_val] += 1
            if allowed_set and samples_per_label is not None:
                if all(counts[l] >= samples_per_label for l in allowed_set):
                    return images, labels, label_map or {}
    return images, labels, label_map or {}


def build_standard_image_dataset(
    dataset_choice: str,
    image_size: Sequence[int] | int,
    dataset_labels: Sequence[int] | None = None,
    samples_per_label: int | None = None,
    medmnist_subset: str | None = None,
    root: str | Path = "data",
) -> TensorListDataset:
    """
    Build a tensor dataset for torchvision/MedMNIST datasets.
    """
    img_size = _normalize_image_size(image_size)
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    ds_list: List[Dataset] = []
    choice = dataset_choice.lower()

    if choice == "mnist":
        if dataset_labels is None:
            raise ValueError("dataset_labels must be provided for MNIST")
        base = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        ds_list = [base, test]
    elif choice in ("fmnist", "fashionmnist"):
        if dataset_labels is None:
            raise ValueError("dataset_labels must be provided for FMNIST")
        base = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        ds_list = [base, test]
    elif choice == "cifar10":
        cifar_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.Lambda(lambda img: _cifar10_rgb_to_grayscale(np.asarray(img))),
                transforms.ToTensor(),
            ]
        )
        base = datasets.CIFAR10(root=root, train=True, download=True, transform=cifar_transform)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=cifar_transform)
        ds_list = [base, test]
        if dataset_labels is None:
            dataset_labels = [0, 1, 2, 3, 4]  # airplane, automobile, bird, cat, deer
    elif choice == "medmnist":
        if medmnist_subset is None:
            raise ValueError("medmnist_subset is required when dataset_choice='medmnist'")
        try:
            import medmnist
            from medmnist import dataset as medmnist_dataset
        except ImportError as exc:
            raise ImportError("Please install medmnist to use dataset_choice='medmnist'") from exc
        subset = medmnist_subset.lower()
        subset_map = {
            "pathmnist": "PathMNIST",
            "dermamnist": "DermaMNIST",
            "retinamnist": "RetinaMNIST",
            "bloodmnist": "BloodMNIST",
            "organamnist": "OrganAMNIST",
        }
        if subset not in subset_map:
            raise ValueError(f"Unsupported medmnist_subset '{medmnist_subset}'")
        cls_name = subset_map[subset]
        dataset_cls = getattr(medmnist_dataset, cls_name)
        ds_list = [
            dataset_cls(split="train", transform=transform, download=True, root=root),
            dataset_cls(split="val", transform=transform, download=True, root=root),
            dataset_cls(split="test", transform=transform, download=True, root=root),
        ]
        images, labels, label_map = _collect_standard_samples(ds_list, dataset_labels, samples_per_label)
        ds = TensorListDataset(images, labels)
        ds.label_map = label_map
        return ds
    else:
        raise ValueError(f"Unsupported dataset_choice '{dataset_choice}'")

    images, labels, label_map = _collect_standard_samples(ds_list, dataset_labels, samples_per_label)
    ds = TensorListDataset(images, labels)
    ds.label_map = label_map
    return ds


def _default_pcam_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "camelyon" / "3channel"


def build_pcam_dataset(
    image_size: Sequence[int] | int,
    samples_per_class: int | None = None,
    root: str | Path | None = None,
    seed: int = 42,
) -> PCamDataset:
    data_root = Path(root) if root is not None else _default_pcam_root()
    return PCamDataset(root=data_root, image_size=image_size, samples_per_class=samples_per_class, seed=seed)


def _pcam_split_dirs(root: Path) -> Tuple[Path, Path, Path | None] | None:
    train_dir = root / "train"
    test_dir = root / "test"
    val_dir = root / "val"
    has_train = train_dir.is_dir()
    has_test = test_dir.is_dir()
    has_val = val_dir.is_dir()
    if has_train and has_test:
        return train_dir, test_dir, val_dir if has_val else None
    if has_train or has_test or has_val:
        raise FileNotFoundError(
            f"PCam root {root} contains partial split directories; expected both train/ and test/ if split exists."
        )
    return None


def build_standard_loaders(
    dataset_choice: str,
    image_size: Sequence[int] | int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_count: int | None = None,
    val_count: int | None = None,
    test_count: int | None = None,
    seed: int = 42,
    dataset_labels: Sequence[int] | None = None,
    samples_per_label: int | None = None,
    medmnist_subset: str | None = None,
    balance_sampler: bool = False,
    pcam_root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> Tuple[Tuple[DataLoader, Dataset], Tuple[DataLoader, Dataset], Tuple[DataLoader, Dataset]]:
    """
    Construct DataLoaders for torchvision/MedMNIST datasets with label filtering and per-label limits,
    or for PCam/Camelyon16 numpy patches stored under class0/class1 folders.
    Set data_root to override the torchvision/MedMNIST root directory.
    If pcam_root contains train/test (and optional val) directories, those splits are used as-is.
    """
    choice = dataset_choice.lower()
    use_counts = any(c is not None for c in (train_count, val_count, test_count))
    if use_counts and not all(c is not None for c in (train_count, val_count, test_count)):
        raise ValueError("train_count, val_count, and test_count must all be set when using count-based split.")
    if choice == "pcam":
        pcam_root_path = Path(pcam_root) if pcam_root is not None else _default_pcam_root()
        split_dirs = _pcam_split_dirs(pcam_root_path)
        if split_dirs is not None:
            train_dir, test_dir, val_dir = split_dirs
            train_dataset = build_pcam_dataset(
                image_size=image_size,
                samples_per_class=samples_per_label,
                root=train_dir,
                seed=seed,
            )
            test_dataset = build_pcam_dataset(
                image_size=image_size,
                samples_per_class=samples_per_label,
                root=test_dir,
                seed=seed,
            )
            val_dataset = (
                build_pcam_dataset(
                    image_size=image_size,
                    samples_per_class=samples_per_label,
                    root=val_dir,
                    seed=seed,
                )
                if val_dir is not None
                else None
            )
            if use_counts:
                if val_dataset is not None:
                    train_set = split_dataset_by_counts(train_dataset, train_count, 0, 0, seed)[0]
                    val_set = split_dataset_by_counts(val_dataset, val_count, 0, 0, seed)[0]
                elif val_count > 0:
                    train_set, val_set, _ = split_dataset_by_counts(train_dataset, train_count, val_count, 0, seed)
                else:
                    train_set = split_dataset_by_counts(train_dataset, train_count, 0, 0, seed)[0]
                    val_set = Subset(train_dataset, [])
                test_set = split_dataset_by_counts(test_dataset, test_count, 0, 0, seed)[0]
            else:
                train_set = Subset(train_dataset, range(len(train_dataset)))
                test_set = Subset(test_dataset, range(len(test_dataset)))
                if val_dataset is not None:
                    val_set = Subset(val_dataset, range(len(val_dataset)))
                else:
                    val_set = Subset(train_dataset, [])
        else:
            full_dataset = build_pcam_dataset(
                image_size=image_size,
                samples_per_class=samples_per_label,
                root=pcam_root_path,
                seed=seed,
            )
            if use_counts:
                train_set, val_set, test_set = split_dataset_by_counts(
                    full_dataset, train_count, val_count, test_count, seed
                )
            else:
                train_set, val_set, test_set = split_dataset(full_dataset, train_ratio, val_ratio, test_ratio, seed)
    else:
        root = data_root if data_root is not None else "data"
        full_dataset = build_standard_image_dataset(
            dataset_choice=dataset_choice,
            image_size=image_size,
            dataset_labels=dataset_labels,
            samples_per_label=samples_per_label,
            medmnist_subset=medmnist_subset,
            root=root,
        )
        if use_counts:
            train_set, val_set, test_set = split_dataset_by_counts(
                full_dataset, train_count, val_count, test_count, seed
            )
        else:
            train_set, val_set, test_set = split_dataset(full_dataset, train_ratio, val_ratio, test_ratio, seed)

    train_sampler = None
    if balance_sampler:
        base_dataset = train_set.dataset if hasattr(train_set, "dataset") else train_set
        labels = [base_dataset.labels[i] for i in train_set.indices]
        total = len(labels)
        counts = Counter(labels)
        class_weights = {cls: total / (2 * max(1, cnt)) for cls, cnt in counts.items()}
        weights = torch.as_tensor([class_weights[lbl] for lbl in labels], dtype=torch.double)
        train_sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=not balance_sampler,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return (train_loader, train_set), (val_loader, val_set), (test_loader, test_set)
