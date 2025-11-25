from torch.utils.data import Dataset
import os
import cv2
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.models as models


class MicroUSDataset(Dataset):
    def __init__(self, root, split="train", img_folder="imgs", useFloatPrompt=False, 
                usePrevMask=False, useMultiImage=False, useAGBCE=False,
                use_video_embedding=False, transform=None):
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
        self.agbce = useAGBCE
        self.non_expert_list = os.path.join(self.root, f"{split}_png", "sts")
        # self.use_video_embedding = use_video_embedding

        # -------------------------------------------------------------
        # NEW: Build a frozen video embedding CNN (ResNet18 by default)
        # -------------------------------------------------------------
        # if self.use_video_embedding:
        #     resnet = models.resnet18(weights=None)
        #     resnet.fc = torch.nn.Identity()          # output = 512-dim
        #     self.video_encoder = resnet.eval()        # freeze
        #     for p in self.video_encoder.parameters():
        #         p.requires_grad = False

        #     self.preprocess = T.Compose([
        #         T.ToTensor(),
        #         T.Resize((224, 224)),
        #     ])

        #     self.video_embedding_cache = {}   # stores precomputed embeddings
        # # -------------------------------------------------------------

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.img_filenames)
    
    # def compute_video_embedding(self, base_name):
    #     """
    #     Computes a single embedding for all frames in the sequence (base_name_*).
    #     Uses a frozen ResNet18 over all frames → averaged feature.
    #     """
    #     frame_paths = sorted(
    #         [f for f in self.img_filenames if f.startswith(base_name)]
    #     )

    #     feats = []

    #     for fname in frame_paths:
    #         img_path = os.path.join(self.img_dir, fname)
    #         img = cv2.imread(img_path)
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #         img = self.preprocess(Image.fromarray(img))   # (3,224,224)
    #         with torch.no_grad():
    #             feat = self.video_encoder(img.unsqueeze(0))  # (1,512)
    #         feats.append(feat)

    #     feats = torch.stack(feats, dim=0).mean(dim=0)  # → (1,512)
    #     return feats.squeeze(0)  # (512,)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "val" or self.split == "test":
            img_name = self.img_filenames[idx]
            img_path = os.path.join(self.img_dir, img_name)

            slice_num = int(img_name.split("_")[-1].split(".")[0])
            base_name = "_".join(img_name.split("_")[:-1])
            total_slices = len([name for name in self.img_filenames if name.startswith(base_name)]) - 1
            
            # -------------------------------------------------------------
            # NEW: Video embedding (cached!)
            # -------------------------------------------------------------
            # if self.use_video_embedding:
            #     if base_name not in self.video_embedding_cache:
            #         self.video_embedding_cache[base_name] = self.compute_video_embedding(base_name)
            #     video_embedding = self.video_embedding_cache[base_name]  # (512,)
            # else:
            #     video_embedding = None
            # -------------------------------------------------------------

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

            if self.agbce:
                # print("Loading non-expert mask")
                non_expert_path = os.path.join(self.non_expert_list, img_name.replace("img", "st"))
                non_expert_mask = Image.open(non_expert_path)
                non_expert_mask = np.array(non_expert_mask)
                non_expert_mask = np.where(non_expert_mask == 255, 1, 0)

            if self.transform:
                if self.usePrevMask:
                    if self.agbce:
                        image, mask, prev_mask, non_expert_mask = self.transform(image, mask, prev_mask, non_expert_mask)
                    else:
                        image, mask, prev_mask = self.transform(image, mask, prev_mask)
                else:
                    if self.agbce:
                        image, mask, non_expert_mask = self.transform(image, mask, non_expert_mask=non_expert_mask)
                    else:
                        image, mask = self.transform(image, mask)

            # -------------------------------------------------------------
            # Return includes video embedding
            # -------------------------------------------------------------
            # if self.use_video_embedding:
            #     # Always include as torch tensor
            #     video_embedding = video_embedding.clone().detach()

            if self.usePrevMask:
                items = [image, mask, img_name, prev_mask]
            else:
                items = [image, mask, img_name]
            if self.useFloatPrompt:
                items.append(slice_position)
            if self.agbce:
                items.append(non_expert_mask)
                
            # if self.use_video_embedding:
            #     items.append(video_embedding)

            items.extend([H, W])

            return tuple(items)