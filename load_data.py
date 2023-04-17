import torch
import glob
import pandas as pd
from PIL import Image


class MyDataClass(torch.utils.data.Dataset):
    def __init__(self, root_path, mode, labels_to_idx):
        self.root_path = root_path
        self.mode = mode
        self.paths = [f for f in glob.glob(root_path + mode + "/" + '*')]
        self.df = pd.read_csv(self.root_path + f"{mode}.csv")
        self.labels_to_idx = labels_to_idx

    def __getitem__(self, idx):
        image_name = self.paths[idx][len(self.root_path) + len(self.mode) + 1:]
        image = Image.open(f'{self.root_path}' + f'{self.mode}' + "/" + f'{image_name}').convert('RGB')
        label = self.df[self.df["image_id"] == image_name]
        if self.mode == "test":
            return image, image_name
        return image, torch.tensor(self.labels_to_idx[label.values.tolist()[0][1]], dtype=torch.int64)

    def __len__(self):
        return len(self.paths)


class AugData(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx][0]
        item = self.transform(item)
        return item, self.data[idx][1]


def get_all_labels(path_to_data):
    df = pd.read_csv(path_to_data)
    return df["label"].unique()
