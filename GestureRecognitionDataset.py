import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class GestureRecognitionDataset(Dataset):
    def __init__(self, csv_file, directory, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        imagePath = os.path.join(self.directory, self.annotations.iloc[index, 0])
        image = io.imread(imagePath)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label



