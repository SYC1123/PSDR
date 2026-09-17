import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.csv_utils import label_frame_to_int


class ISICDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        super().__init__()
        file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.images = file["image"].values
        label_matrix = label_frame_to_int(file.iloc[:, 1:])
        self.labels = np.argmax(label_matrix.values, 1)
        self.transform = transform
        self.n_class = len(np.unique(self.labels))
        self.class_names = file.columns[1:]

        print(
            "Total # images:{}, labels:{}, number of classes={}".format(
                len(self.images), len(self.labels), self.n_class
            )
        )

    def __getitem__(self, index):
        try:
            image_name = os.path.join(self.root_dir, self.images[index] + ".jpg")
            image = Image.open(image_name).convert("RGB")
        except FileNotFoundError:
            image_name = os.path.join(self.root_dir, self.images[index] + ".JPG")
            image = Image.open(image_name).convert("RGB")
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.images)
