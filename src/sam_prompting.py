import typing as tp
from warnings import warn
import torch
from torch import nn
import numpy as np
from src.sam_wrappers import build_medsam, build_sam 

class SliceTrackSam(nn.Module):
    BACKBONE_OPTIONS = [
        "sam",
        "medsam",
    ]

    def __init__(
        self,
        floating_point_prompts: list[str] = [],
        use_prev_mask_prompt: bool = False,
        sam_backbone: tp.Literal[
            "sam", "medsam",
        ] = "medsam",
        freeze_mask_decoder: bool = False,
        freeze_image_encoder: bool = False,
        prompt_embedding_dim=256,
        random_prev_mask_prob = 0.5,
        image_size = 1024, 
        auto_resize_image_to_native: bool = False
    ):
        super().__init__()
        self.floating_point_prompts = floating_point_prompts
        self.use_prev_mask_prompt = use_prev_mask_prompt
        self.random_prev_mask_prob = random_prev_mask_prob
        self.prompt_embedding_dim = prompt_embedding_dim
        self.auto_resize_image_to_native = auto_resize_image_to_native


        # BUILD BACKBONE
        if sam_backbone == "medsam":
            self.medsam_model = build_medsam()
            self.image_size_for_features = image_size
        elif sam_backbone == "sam":
            self.medsam_model = build_sam()
            self.image_size_for_features = image_size
        

        if freeze_image_encoder:
            print("Freezing image encoder")
            for param in self.medsam_model.image_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            print("Freezing mask decoder")
            for param in self.medsam_model.mask_decoder.parameters():
                param.requires_grad = False

        # ====================================================
        # BUILD PROMPT MODULES
        # ==================================================

        # floating point prompts
        self.floating_point_prompt_modules = torch.nn.ModuleDict()
        for prompt in self.floating_point_prompts:
            self.floating_point_prompt_modules[prompt] = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, prompt_embedding_dim),
            )

    def forward(
        self,
        image=None,
        prev_mask=None,
        return_prompt_embeddings=False,
        **prompts,
    ):

        B, C, H, W = image.shape

        if (H != self.image_size_for_features or W != self.image_size_for_features) and self.auto_resize_image_to_native:
            warn(f"Detected image of resolution {H, W} which is different from native image encoder resolution {self.image_size_for_features, self.image_size_for_features}. Resizing...")
            image_resized_for_features = torch.nn.functional.interpolate(
                image, size=(self.image_size_for_features, self.image_size_for_features), mode="bicubic"
            )
        else:
            image_resized_for_features = image

        image_feats = self.medsam_model.image_encoder(image_resized_for_features)
       

        if self.use_prev_mask_prompt:
            if np.random.rand() > self.random_prev_mask_prob:
                rand_prev_masks = None
            else:
                rand_prev_masks = prev_mask
        else:
            rand_prev_masks = None
        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder.forward(
            None, None, rand_prev_masks
        )
        if (dense_embedding.shape[-2] != image_feats.shape[-2]) or (dense_embedding.shape[-1] != image_feats.shape[-1]):
            dense_embedding = torch.nn.functional.interpolate(
                dense_embedding, size=image_feats.shape[-2:],
            )

        if sparse_embedding.size(0) == 1 and B > 1:
            sparse_embedding = sparse_embedding.expand(B, -1, -1)

        for prompt_name, prompt_value in prompts.items():
            if prompt_name in self.floating_point_prompts:
                prompt_embedding = self.floating_point_prompt_modules[prompt_name](
                    prompt_value
                )
            else:
                raise ValueError(f"Unknown prompt: {prompt_name}")

            prompt_embedding = prompt_embedding[:, None, :]

            sparse_embedding = torch.cat([sparse_embedding, prompt_embedding], dim=1)

      
        pe = self.medsam_model.prompt_encoder.get_dense_pe()
        if (pe.shape[-2] != image_feats.shape[-2]) or (pe.shape[-1] != image_feats.shape[-1]):
            pe = torch.nn.functional.interpolate(
                pe, size=image_feats.shape[-2:],
            )

        mask_logits = self.medsam_model.mask_decoder.forward(
            image_feats,
            pe,
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
        )[0]

        if return_prompt_embeddings:
            return mask_logits, sparse_embedding, dense_embedding
        else:
            return mask_logits


    def get_params(self):

        img_encoder_params = self.medsam_model.image_encoder.parameters()
        prompt_encoder_params = self.medsam_model.prompt_encoder.parameters()
        decoder_params = self.medsam_model.mask_decoder.parameters()

        return img_encoder_params, prompt_encoder_params, decoder_params