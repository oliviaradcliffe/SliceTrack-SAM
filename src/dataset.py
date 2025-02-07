from torch.utils.data import Dataset
import os
import cv2
import torch
from PIL import Image
import numpy as np

class MicroUSDataset(Dataset):
    def __init__(self, root, split="train", img_folder="imgs", useFloatPrompt=False, usePrevMask=False, useMultiImage=False, transform=None):
        """
        Initialize the dataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split, e.g., 'train', 'val', or 'test'.
            img_folder (str): Folder containing the images.
            prevMask (bool): Whether to use the previous mask as input.
            transform (Optional[Callable]): Transform to be applied on a sample.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.img_folder = img_folder
        self.usePrevMask = usePrevMask
        self.useFloatPrompt = useFloatPrompt
        self.useMultiImage = useMultiImage
        self.img_dir = os.path.join(self.root, f"{split}_png", self.img_folder)
        self.mask_dir = os.path.join(self.root, f"{split}_png", "gts")
        self.img_filenames = sorted(os.listdir(self.img_dir))

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.img_filenames)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "val" or self.split == "test":
            img_name = self.img_filenames[idx]
            img_path = os.path.join(self.img_dir, img_name)

            slice_num = int(img_name.split("_")[-1].split(".")[0])
            base_name = "_".join(img_name.split("_")[:-1])
            total_slices = len([name for name in self.img_filenames if name.startswith(base_name)]) - 1
            
            if self.useMultiImage:
                if slice_num == 0:
                    prev_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    prev_img_path = img_path.replace(f"_{slice_num:02d}.png", f"_{slice_num-1:02d}.png")
                    prev_img = cv2.imread(prev_img_path, cv2.IMREAD_GRAYSCALE) 

                if slice_num == total_slices:
                    next_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    next_img_path = img_path.replace(f"_{slice_num:02d}.png", f"_{slice_num+1:02d}.png")
                    next_img = cv2.imread(next_img_path, cv2.IMREAD_GRAYSCALE)
                
                cur_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = np.stack((prev_img, cur_img, next_img), axis=-1)

            else:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            H, W = image.shape[0], image.shape[1]
 
            if self.img_folder == "imgs_croppad":
                gt_path = os.path.join(self.mask_dir, img_name.replace("img", "gt").replace("_croppad.png", ".png"))
            else:
                gt_path = os.path.join(self.mask_dir, img_name.replace("img", "gt"))
            gt = Image.open(gt_path)
            gt_array = np.array(gt)
            mask = np.where(gt_array == 255, 1, 0) 

            if self.usePrevMask:
                if "00" in gt_path:
                    prev_mask = np.zeros((256, 256), dtype=np.float32)

                else:
                    prev_mask_path = gt_path.replace(f"_{slice_num:02d}.png", f"_{slice_num-1:02d}.png")
                    prev_mask = Image.open(prev_mask_path)
                    prev_mask = np.array(prev_mask)
                    prev_mask = np.where(prev_mask == 255, 1, 0)

            else:
                prev_mask = np.zeros((256, 256), dtype=np.float32)
            
            if self.useFloatPrompt:
                # Calculate slice position
                slice_position = slice_num/(total_slices+1)
                slice_position =  torch.as_tensor(slice_position).to(dtype=torch.float32)

            if self.transform:
                if self.usePrevMask:
                    image, mask, prev_mask = self.transform(image, mask, prev_mask)
                else:
                    image, mask = self.transform(image, mask)

            if self.usePrevMask:
                if self.useFloatPrompt:
                    return image, mask, img_name, prev_mask, slice_position, H, W
                return image, mask, img_name, prev_mask, H, W
            else:
                if self.useFloatPrompt:
                    return image, mask, img_name, slice_position, H, W
                return image, mask, img_name, H, W