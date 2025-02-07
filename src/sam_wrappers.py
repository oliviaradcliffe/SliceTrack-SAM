"""
Implements wrappers and registry for Segment Anything Model (SAM) models.
"""

import os
from torch import nn

import sys 
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "vendor"
    )
)

from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling.image_encoder import ImageEncoderViT


CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR") or './chkpt'  # top level checkpoint directory
if CHECKPOINT_DIR is None:
    raise ValueError(
        """Environment variable CHECKPOINT_DIR must be set. It should be a directory with sam and medsam checkpoints."""
    )


def build_sam(add_interpolated_pos_embedding=True):
    """Builds the sam-vit-b model."""
    checkpoint = os.path.join(CHECKPOINT_DIR, "sam_vit_l_0b3195.pth")
    model = sam_model_registry["vit_l"](checkpoint=checkpoint)
    if add_interpolated_pos_embedding:
        wrap_with_interpolated_pos_embedding_(model)
    return model


def build_medsam(add_interpolated_pos_embedding=True):
    """
    Builds the MedSAM model by building the SAM model and loading the medsam checkpoint.
    """
    checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b.pth")
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth") # for CPU
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    if add_interpolated_pos_embedding:
        wrap_with_interpolated_pos_embedding_(model)
    return model


def interpolate_pos_encoding(x, pos_embed):
    npatch_in_h = x.shape[1]
    npatch_in_w = x.shape[2]

    patch_pos_embed = pos_embed

    npatch_native_h = patch_pos_embed.shape[1]
    npatch_native_w = patch_pos_embed.shape[2]

    if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
        return pos_embed

    w0 = npatch_in_w
    h0 = npatch_in_h
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.permute(0, 3, 1, 2),
        scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
    return patch_pos_embed


def forward_return_features(image_encoder: ImageEncoderViT, x, return_hiddens=False): 
    # "Return hiddens" feature added

    x = image_encoder.patch_embed(x)
    if image_encoder.pos_embed is not None:
        x = x + interpolate_pos_encoding(x, image_encoder.pos_embed)

    hiddens = []
    for blk in image_encoder.blocks:
        x = blk(x)
        if return_hiddens:
            hiddens.append(x)

    x = image_encoder.neck(x.permute(0, 3, 1, 2))

    return (x, hiddens) if return_hiddens else x


def wrap_with_interpolated_pos_embedding_(sam_model): 
    type(sam_model.image_encoder).forward = forward_return_features
