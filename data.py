import torch
import glob
import os
import cv2
import numpy as np

class SeepDataset(torch.utils.data.Dataset):
    def __init__(self, img_path: str, mask_path: str):
        self.img_path = img_path
        self.mask_path = mask_path
        self.data_list = self.get_data_list(img_path, mask_path)

    def get_data_list(self, img_path: str, mask_path: str):
        data_list = []
        for filename in glob.iglob(img_path + "*tif"):
            img_id = os.path.basename(filename)
            data_list.append(img_id)
        return data_list

    def __getitem__(self, index):
        img_id = self.data_list[index]

        img = cv2.imread(self.img_path + img_id, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)

        mask = cv2.imread(self.mask_path + img_id, cv2.IMREAD_UNCHANGED)
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return img, mask, img_id

    def __len__(self):
        return len(self.data_list)
