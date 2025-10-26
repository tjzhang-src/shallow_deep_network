"""
data.py
- Standardized data loaders for CIFAR10, CIFAR100, and Tiny-ImageNet.
- Tiny-ImageNet supports Parquet-backed listings (path or bytes) and preserves the original API
  (aug_train_loader/train_loader/test_loader and corresponding datasets).
"""

import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class AddTrigger(object):
    def __init__(self, square_size: int = 5, square_loc: Tuple[int, int] = (26, 26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data: Image.Image) -> Image.Image:
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data


class CIFAR10:
    def __init__(self, batch_size: int = 128, add_trigger: bool = False):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        self.normalized = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.aug_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.normalized)
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.normalized)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

        if add_trigger:
            self.trigger_transform = transforms.Compose([AddTrigger(), transforms.ToTensor(), normalize])
            self.trigger_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.trigger_transform)
            self.trigger_test_loader = DataLoader(self.trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=4)


class CIFAR100:
    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        self.augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
        self.normalized = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.aug_trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=self.normalized)
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=self.normalized)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):  # type: ignore[override]
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class ParquetImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, path_col: str = 'path', label_col: str = 'label', transform=None, return_path: bool = False, class_to_idx: Optional[dict] = None):
        self.df = df.reset_index(drop=True)
        if path_col not in self.df.columns:
            raise ValueError(f"Missing column '{path_col}' in dataframe")
        if label_col not in self.df.columns:
            raise ValueError(f"Missing column '{label_col}' in dataframe")
        self.paths = self.df[path_col].tolist()
        self.labels_raw = self.df[label_col].tolist()
        # Establish shared mapping if provided; otherwise infer and expose it
        if class_to_idx is None:
            uniq = sorted(set(self.labels_raw))
            self.class_to_idx = {c: i for i, c in enumerate(uniq)}
            self.classes = uniq
        else:
            self.class_to_idx = class_to_idx
            # Keep a stable classes list aligned with mapping order
            self.classes = list(class_to_idx.keys())
        # Map all labels using mapping (robust for str/int)
        def _map_label(v, mapping):
            if v in mapping:
                return mapping[v]
            sv = str(v)
            if sv in mapping:
                return mapping[sv]
            try:
                iv = int(v)
                if iv in mapping:
                    return mapping[iv]
            except Exception:
                pass
            raise KeyError(f'Label value {v!r} not found in class_to_idx')
        self.labels = [int(_map_label(x, self.class_to_idx)) for x in self.labels_raw]
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, label, path
        return img, label


class ParquetImageBytesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_col: str = 'image', label_col: str = 'label', transform=None, return_raw: bool = False, class_to_idx: Optional[dict] = None):
        self.df = df.reset_index(drop=True)
        if image_col not in self.df.columns:
            raise ValueError(f"Missing image bytes column '{image_col}' in dataframe")
        if label_col not in self.df.columns:
            raise ValueError(f"Missing label column '{label_col}' in dataframe")
        self.images = self.df[image_col].tolist()
        self.labels_raw = self.df[label_col].tolist()
        if class_to_idx is None:
            uniq = sorted(set(self.labels_raw))
            self.class_to_idx = {c: i for i, c in enumerate(uniq)}
            self.classes = uniq
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())
        def _map_label(v, mapping):
            if v in mapping:
                return mapping[v]
            sv = str(v)
            if sv in mapping:
                return mapping[sv]
            try:
                iv = int(v)
                if iv in mapping:
                    return mapping[iv]
            except Exception:
                pass
            raise KeyError(f'Label value {v!r} not found in class_to_idx')
        self.labels = [int(_map_label(x, self.class_to_idx)) for x in self.labels_raw]
        self.transform = transform
        self.return_raw = return_raw

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import io
        import base64
        raw = self.images[idx]
        label = self.labels[idx]

        def to_pil(obj):
            # 1) Direct bytes-like
            if isinstance(obj, (bytes, bytearray, memoryview)):
                return Image.open(io.BytesIO(obj)).convert('RGB')
            # 2) Dict structures (e.g., {"bytes": ..., "path": ...})
            if isinstance(obj, dict):
                # Prefer explicit bytes-like content
                for k in ('bytes', 'image', 'data', 'content', 'buffer'):
                    if k in obj:
                        try:
                            return to_pil(obj[k])
                        except Exception:
                            pass
                # Fallback to a path field if present and exists
                for k in ('path', 'filepath', 'file'):
                    if k in obj and isinstance(obj[k], str) and os.path.exists(obj[k]):
                        return Image.open(obj[k]).convert('RGB')
                # If dict had no usable payload
                raise ValueError('Unsupported dict payload for image column')
            # 3) String: path or base64
            if isinstance(obj, str):
                # Path-like string
                if os.path.exists(obj):
                    return Image.open(obj).convert('RGB')
                # Try base64 decode
                try:
                    decoded = base64.b64decode(obj, validate=True)
                    return Image.open(io.BytesIO(decoded)).convert('RGB')
                except Exception:
                    # Not base64; cannot interpret
                    raise ValueError('String image payload is neither a valid path nor base64-encoded bytes')
            # 4) NumPy arrays
            try:
                import numpy as np  # type: ignore
                if isinstance(obj, np.ndarray):
                    arr = obj
                    if arr.dtype != np.uint8:
                        if arr.dtype.kind == 'f':
                            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
                        else:
                            arr = arr.astype(np.uint8)
                    return Image.fromarray(arr)
            except Exception:
                pass
            # Unknown type
            raise ValueError(f'Unsupported image payload type: {type(obj)}')

        img = to_pil(raw)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_raw:
            return img, label, raw
        return img, label


class TinyImagenet:
    def __init__(self, batch_size: int = 128):
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64

        train_parquet = '/home/tangjie.zhang/dataset/tinyimagenet/train.parquet'
        valid_parquet = '/home/tangjie.zhang/dataset/tinyimagenet/valid.parquet'

        df_train = pd.read_parquet(train_parquet)
        df_valid = pd.read_parquet(valid_parquet)

        def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        path_col = find_col(df_train, ['path', 'filepath', 'image_path', 'file_path', 'img_path', 'filename', 'file', 'name', 'image_file'])
        label_col = find_col(df_train, ['label', 'labels', 'target', 'targets', 'y', 'class_id', 'class', 'wnid', 'synset'])
        bytes_col = None if path_col else find_col(df_train, ['image', 'bytes', 'data', 'content', 'image_bytes'])
        if label_col is None:
            raise ValueError('Parquet must have a label column (e.g., label/target/class_id)')

        self.num_train = len(df_train)
        self.num_test = len(df_valid)
        classes_sorted = sorted(pd.unique(df_train[label_col]).tolist())
        self.num_classes = len(classes_sorted) if len(classes_sorted) > 0 else 200

        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
        self.augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            normalize,
        ])
        self.normalized = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        # Build a shared class mapping from training labels to ensure consistency with validation
        train_labels = df_train[label_col].tolist()
        uniq_train = sorted(set(train_labels))
        class_to_idx = {c: i for i, c in enumerate(uniq_train)}

        if path_col:
            self.aug_trainset = ParquetImagePathDataset(df_train, path_col, label_col, transform=self.augmented, class_to_idx=class_to_idx)
            self.trainset = ParquetImagePathDataset(df_train, path_col, label_col, transform=self.normalized, class_to_idx=class_to_idx)
            self.testset = ParquetImagePathDataset(df_valid, path_col, label_col, transform=self.normalized, class_to_idx=class_to_idx)
            self.testset_paths = ParquetImagePathDataset(df_valid, path_col, label_col, transform=self.normalized, return_path=True, class_to_idx=class_to_idx)
        elif bytes_col:
            self.aug_trainset = ParquetImageBytesDataset(df_train, bytes_col, label_col, transform=self.augmented, class_to_idx=class_to_idx)
            self.trainset = ParquetImageBytesDataset(df_train, bytes_col, label_col, transform=self.normalized, class_to_idx=class_to_idx)
            self.testset = ParquetImageBytesDataset(df_valid, bytes_col, label_col, transform=self.normalized, class_to_idx=class_to_idx)
            self.testset_paths = ParquetImageBytesDataset(df_valid, bytes_col, label_col, transform=self.normalized, return_raw=True, class_to_idx=class_to_idx)
        else:
            raise ValueError('Parquet must have an image path column (e.g., path) or an image bytes column (e.g., image)')

        self.aug_train_loader = DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        self.train_loader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        self.test_loader = DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=8)


def get_mean_and_std(dataset):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def create_val_folder():
    path = os.path.join('data/tiny-imagenet-200', 'val/images')
    filename = os.path.join('data/tiny-imagenet-200', 'val/val_annotations.txt')
    with open(filename, 'r') as fp:
        data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    for img, folder in val_img_dict.items():
        newpath = os.path.join(path, folder)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_w_preds(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count