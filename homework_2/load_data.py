import torch
import cv2
import pandas as pd
import numpy as np
import glob

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class RecognitionDataset(Dataset):
    """Class for training image-to-text mapping using CTC-Loss."""

    def __init__(self, config, alphabet, transforms=None):
        """Constructor for class.
        
        Args:
            - config: List of items, each of which is a dict with keys "file" & "text".
            - alphabet: String of chars required for predicting.
            - transforms: Transformation for items, should accept and return dict with keys "image", "seq", "seq_len" & "text".
        """
        super(RecognitionDataset, self).__init__()
        self.config = config
        self.alphabet = alphabet
        self.image_names, self.texts = self._parse_root_()
        self.transforms = transforms

    def _parse_root_(self):
        image_names, texts = [], []
        for item in self.config:
            image_name = item["file"]
            text = item['text']
            texts.append(text)
            image_names.append(image_name)
        return image_names, texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """Returns dict with keys "image", "seq", "seq_len" & "text".
        Image is a numpy array, float32, [0, 1].
        Seq is list of integers.
        Seq_len is an integer.
        Text is a string.
        """
        image = cv2.imread(self.image_names[item]).astype(np.float32) / 255.
        text = self.texts[item]
        seq = self.text_to_seq(text)
        seq_len = len(seq)
        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)
        if self.transforms is not None:
            output = self.transforms(output)
        return output

    def text_to_seq(self, text):
        """Encode text to sequence of integers.
        
        Args:
            - String of text.
            
        Returns:
            List of integers where each number is index of corresponding characted in alphabet + 1.
        """
        seq = [self.alphabet.find(c) + 1 for c in text]

        return seq

    
class TestDataset(Dataset):

    def __init__(self, root_path, alphabet, size=(320, 64)):
        super(TestDataset, self).__init__()
        self.root_path = root_path
        self.paths = [f for f in glob.glob(root_path + "test/test/" + '*')]
        self.alphabet = alphabet
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        image = cv2.imread(self.paths[item]).astype(np.float32) / 255.

        interpolation = cv2.INTER_AREA if self.size[0] < image.shape[1] else cv2.INTER_LINEAR
        image = cv2.resize(image, self.size, interpolation=interpolation)
        output = image
            
        image_name = self.paths[item][len(self.root_path) + 2 * len("test") + 2:]
        return output, image_name
    
    
class Resize(object):

    def __init__(self, size=(320, 100)):
        self.size = size

    def __call__(self, item):
        """Apply resizing.
        
        Args: 
            - item: Dict with keys "image", "seq", "seq_len", "text".
        
        Returns: 
            Dict with image resized to self.size.
        """
        
        interpolation = cv2.INTER_AREA if self.size[0] < item["image"].shape[1] else cv2.INTER_LINEAR
        item["image"] = cv2.resize(item["image"], self.size, interpolation=interpolation)

        return item
