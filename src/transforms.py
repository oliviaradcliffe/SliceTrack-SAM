from torchvision.transforms import v2 as T
from torchvision.tv_tensors import Image, Mask
import random
import torch
import cv2

class Transform():
    def __init__(self, output_size=1024, mask_size=256, crop_pad=False, augment=False, h_flip_prob=0):
        self.output_size = output_size
        self.mask_size = mask_size
        self.crop_pad = crop_pad
        self.augment = augment
        self.h_flip_prob = h_flip_prob

    def __call__(self, image, mask, prevMask=None):
        transform = T.Resize((self.output_size, self.output_size))
        input_image_torch = torch.as_tensor(image).permute(2, 0, 1)
        input_image_torch = transform(input_image_torch)
        transformed_image = input_image_torch 
        transformed_image = transformed_image.to(dtype=torch.float32)

        transformed_mask = Mask(mask)
        transformed_mask = transform(transformed_mask).to(dtype=torch.float32)
        
        if prevMask is not None:
            transformed_prevMask = Mask(prevMask)
            transformed_prevMask = T.Resize((self.output_size, self.output_size))(transformed_prevMask).to(dtype=torch.float32)

        if self.crop_pad:
            # crop top and bottom
            top = random.randint(0, 128)
            bottom = random.randint(0, 128)

            transformed_image = transformed_image[:, top:self.output_size-bottom, :]

            # add black padding
            transformed_image = cv2.copyMakeBorder(transformed_image.permute(1,2,0).cpu().numpy(), top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
            transformed_image = torch.tensor(transformed_image).permute(2,0,1) 

        if self.augment:
            if prevMask is not None:
                transformed_image, transformed_mask, transformed_prevMask = T.RandomHorizontalFlip(p=self.h_flip_prob)(transformed_image, transformed_mask, transformed_prevMask)
                transformed_image, transformed_mask, transformed_prevMask = T.RandomResizedCrop(self.output_size, scale=(0.7,1))(transformed_image, transformed_mask, transformed_prevMask)

            else:
                transformed_image, transformed_mask = T.RandomHorizontalFlip(p=self.h_flip_prob)(transformed_image, transformed_mask)
                transformed_image, transformed_mask = T.RandomResizedCrop(self.output_size, scale=(0.8,1))(transformed_image, transformed_mask)

            gamma = self._sample_gamma()
            transformed_image = transformed_image**gamma
    
        # resize mask to original size
        transformed_mask = T.Resize((image.shape[0], image.shape[1]))(transformed_mask)

        # normalize
        transformed_image = (transformed_image - transformed_image.min()) / (transformed_image.max() - transformed_image.min())
           
        if prevMask is not None:
            return transformed_image, transformed_mask, transformed_prevMask
        else:
            return transformed_image, transformed_mask

    def _sample_gamma(self):
        
        gamma = 2**random.uniform(-1,1)
        return gamma
