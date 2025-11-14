import os
import numpy as np
from tensorflow.keras.utils import Sequence
from PIL import Image

# This will load images and masks from given directories or lists of file paths
# It automatically preprocesses them (resizing, normalization) and yields batches for training
class GirafeDataset(Sequence):
    def __init__(self, image_input, mask_input, batch_size=16, image_size=(256, 256), shuffle=True):
        if isinstance(image_input, list) and isinstance(mask_input, list):
            self.image_paths = image_input
            self.mask_paths = mask_input
        else:
            self.image_paths = sorted([os.path.join(image_input, fname) for fname in os.listdir(image_input)])
            self.mask_paths = sorted([os.path.join(mask_input, fname) for fname in os.listdir(mask_input)])

        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        idxs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = []
        batch_masks = []
        for i in idxs:
            img = Image.open(self.image_paths[i]).resize(self.image_size)
            mask = Image.open(self.mask_paths[i]).resize(self.image_size)
            batch_images.append(np.array(img.convert("L")) / 255.0)
            batch_masks.append(np.array(mask.convert("L")) / 255.0)
        return np.expand_dims(np.array(batch_images), -1), np.expand_dims(np.array(batch_masks), -1)
