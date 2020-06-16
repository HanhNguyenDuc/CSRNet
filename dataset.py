import os
import random
import torch
import numpy as np 
from torch.utils.data import Dataset
from PIL import Image
from image import load_data
from typing import Callable, Optional

class ListDataset(Dataset):
    def __init__(
        self,
        root,
        shape: Optional[tuple]=None,
        shuffle: Optional[bool]=False,
        transform: Optional[Callable]=None,
        train: Optional[bool]=None,
        seen: Optional[int]=0,
        batch_size: Optional[int]=1,
        num_workers: Optional[int]=4
    ) -> None:
        if train:
            root = root
        random.shuffle(root)
        
        self.n_sample = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]
        img, target = load_data(img_path, self.train)

        if self.transform is not None:
            img = self.transform(img)
        return img, target