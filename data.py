from __future__ import print_function, division
import torch
from skimage import io
from torch.utils.data import Dataset


def split(word):
    return [char for char in word]


class IAMDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open('readme.txt', 'w+') as lineasciifile:
            lineasciifile.write("")
            self.labels = []
            self.images = []
            for line in lineasciifile:
                line = line.split(" ")

                if line[0] != "#":
                    words = line[8].replace(" ", "")
                    words = words.replace("|", " ")
                    filename = line[0] + ".png"
                    self.labels.append(words)
                    self.images.append(filename)

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
        - An integer representing the length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Gets an item from the dataset.

        Args:
        - idx: An integer representing the index of the item to get.

        Returns:
        - A dictionary containing the label and image for the specified index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread("data/images/" + self.images[idx])
        label = []
        for c in split(self.labels[idx]):
            label.append(ord(c) + 1)

        item = {'label': label, 'image': image}
        return item
