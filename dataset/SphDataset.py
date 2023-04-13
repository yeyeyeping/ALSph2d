import torch
import os
import numpy as np
from os.path import join, exists
import SimpleITK as sitk
from torch.utils.data import Dataset, SubsetRandomSampler
from PIL import Image


class SubsetSampler(SubsetRandomSampler):
    def __iter__(self):
        return iter(self.indices)


class Dataset3d(Dataset):
    def __init__(self, folder) -> None:
        super().__init__()
        assert exists(folder), folder
        self.base_folder = folder
        self.nii_data = os.listdir(join(folder, "image"))

    def __len__(self):
        return len(self.nii_data)

    def __getitem__(self, index):
        name = self.nii_data[index]
        image_path, mask_path = join(self.base_folder, "image", name), join(self.base_folder, "label",
                                                                            name)
        img_obj, mask_obj = sitk.GetArrayFromImage(sitk.ReadImage(image_path)), sitk.GetArrayFromImage(
            sitk.ReadImage(mask_path))
        #强度标准化
        img_obj = np.asarray((img_obj - img_obj.mean()) / img_obj.std(), dtype=np.float32)
        #slice标准化
        img_obj = (img_obj - img_obj.mean(axis=(1,2), keepdims=True)) / img_obj.std(axis=(1,2), keepdims=True)
        mask_obj = np.asarray(mask_obj, dtype=np.uint8)
        return torch.tensor(img_obj, dtype=torch.float32).unsqueeze(1), torch.tensor(mask_obj > 0.5,
                                                                                     dtype=torch.long).unsqueeze(1)


class Dataset2d(Dataset):
    def __init__(self, datafolder: str, transform=None) -> None:
        assert os.path.exists(datafolder), datafolder
        super().__init__()
        self.data_folder = datafolder
        self.transforms = transform
        assert exists(datafolder)
        assert exists((join(datafolder, "image")))
        self.jpg_data = os.listdir(os.path.join(datafolder, "image"))

    def __len__(self):
        return len(self.jpg_data)

    def __getitem__(self, index):
        data_name = label_name = self.jpg_data[index]
        data_path, label_path = join(self.data_folder, "image", data_name), join(self.data_folder, "label", label_name)
        data, label = Image.open(data_path).convert("L"), Image.open(label_path).convert("L")
        data, label = np.asarray(data, dtype=np.float32), np.asarray(label, dtype=np.uint8)
        data = (data - data.mean()) / data.std()
        if self.transforms is not None:
            transformed = self.transforms(image=data, mask=label)
            data, label = transformed["image"], transformed["mask"]

        return torch.tensor(data, dtype=torch.float32).unsqueeze(0), torch.tensor(label == 255,
                                                                                  dtype=torch.long).unsqueeze(0)
