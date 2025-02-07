# testing
from segment_anything import sam_model_registry
from skimage import io
import os
join = os.path.join
import torch
import numpy as np
from torchvision.transforms import v2 as T
import numpy as np
from medpy import metric
import cv2
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.dataset import MicroUSDataset
from src.transforms import Transform
from src.utils import DiceMetric

from src.sam_prompting import SliceTrackSam

import argparse

import scipy.ndimage as ndimage

parser = argparse.ArgumentParser()
parser.add_argument(
    "-root",
    type=str,
    default="./data",
    help="path to dataset",
)
parser.add_argument(
    "-chkpt",
    type=str,
    default='./work_dir/sam_finetuned_256.pth',
    help="path to SAM checkpoint",
)
parser.add_argument(
    "-backbone",
    type=str,
    default="medsam",
    help="backbone",
)
parser.add_argument(
    "-save_path",
    type=str,
    default="./work_dir/sam_finetuned_256/pred",
    help="path to save the model",
)
parser.add_argument(
    "-img_folder",
    type=str,
    default="imgs",
    help="folder name with test images. default: 'imgs'",
)
parser.add_argument(
    "-img_size",
    type=int,
    default=1024,
    help="image size for training",
)
parser.add_argument(
    "-mask_size",
    type=int,
    default=256,
    help="mask size for training",
)
parser.add_argument(
    "-useFloatPrompt",
    type=bool,
    default=False,
    help="use floating point prompts",
)
parser.add_argument(
    "-useMultiImage",
    type=bool,
    default=False,
    help="use multiple images as input",
)
parser.add_argument(
    "-penalty",
    type=bool,
    default=True,
    help="use penalty for hd95",
)



args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"using checkpoint: {args.chkpt}")
    print(f"using img_folder: {args.img_folder}")
    print(f"using device: {device}")
    print(f"using root: {args.root}")
    print(f"using save path: {args.save_path}")
    print(f"using image size: {args.img_size}")
    print(f"using mask size: {args.mask_size}")
    print(f"using floating point prompts: {args.useFloatPrompt}")
    print(f"using multiple images: {args.useMultiImage}")
    print(f"using penalty for hd95: {args.penalty}")
    print("loading model...")

    # instantiate the model
    finetuned_sam_model = SliceTrackSam(
        floating_point_prompts=["position"],
        sam_backbone=args.backbone,
        image_size=args.img_size,
    )
    print(finetuned_sam_model.load_state_dict(torch.load(args.chkpt)))
    finetuned_sam_model.to(device)

    test_dataset = MicroUSDataset(args.root, split="test", useFloatPrompt=args.useFloatPrompt, useMultiImage=args.useMultiImage, img_folder=args.img_folder, transform=Transform(output_size=args.img_size, mask_size=args.mask_size))
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    dice = DiceMetric(num_classes=2)
    hd95_scores = []
    spacing = 0.033586

    # test the model
    finetuned_sam_model.eval()
    for test_iter, batch in enumerate(test_dataloader):
        if args.useFloatPrompt:
            images, gts, img_names, slice_positions, H, W = batch
            slice_positions = slice_positions.unsqueeze(1).to(device)
        else:
            images, gts, img_names, H, W = batch
        
        images = images.to(device)
        gts = gts.to(device)

        with torch.no_grad():
            # run the model
            if args.useFloatPrompt:
                heatmap_logits = finetuned_sam_model(
                    images,
                    position=slice_positions
                )
            else:
                heatmap_logits = finetuned_sam_model(images)
        
            pred = F.interpolate(
                heatmap_logits,
                size=(H[0], W[0]),
                mode="bilinear",
                align_corners=False,
            )

            sam_seg_int = (pred > 0.5).squeeze()

            # dice calculation
            dice.update(sam_seg_int, gts)


            batch_hd95 = []
            for i in range(sam_seg_int.shape[0]):
                pred = np.array(sam_seg_int[i,:,:].cpu().numpy())
                gt = np.array(gts[i,:,:].cpu().numpy())

                pred = pred.astype(np.uint8)
                gt = gt.astype(np.uint8)

                pred = cv2.resize(src=pred, dsize=(1372, 962), interpolation = cv2.INTER_NEAREST)
                gt = cv2.resize(src=gt, dsize=(1372, 962), interpolation = cv2.INTER_NEAREST)
                    
                if sam_seg_int[i,:,:].sum().item() > 0 and gts[i,:,:].sum().item()> 0:
                    hd95 = metric.binary.hd95(pred, gt)
                    batch_hd95.append(hd95*spacing)
                else:
                    if args.penalty == True:
                        if sam_seg_int[i,:,:].sum().item() == 0 and gts[i,:,:].sum().item() == 0:
                            batch_hd95.append(0)
                        elif sam_seg_int[i,:,:].sum().item() == 0 and gts[i,:,:].sum().item() > 0:
                            # find center of mass of gt
                            gt_com = np.array(ndimage.measurements.center_of_mass(gt))
                            # create blank segmentation with center of mass
                            center_pred = np.zeros_like(gt)
                            # set center of mass to 1
                            center_pred[int(gt_com[0]), int(gt_com[1])] = 1
                            hd95 = metric.binary.hd95(gt, center_pred)*spacing
                            batch_hd95.append(hd95)
                        elif sam_seg_int[i,:,:].sum().item() > 0 and gts[i,:,:].sum().item() == 0:
                            # find center of mass of prediction
                            pred_com = np.array(ndimage.measurements.center_of_mass(pred))
                            # create blank segmentation with center of mass
                            center_gt = np.zeros_like(pred)
                            # set center of mass to 1
                            center_gt[int(pred_com[0]), int(pred_com[1])] = 1
                            hd95 = metric.binary.hd95(pred, center_gt)*spacing
                            batch_hd95.append(hd95)
                            
                    else:
                        batch_hd95.append(-1)

            hd95_scores.extend(batch_hd95)


        # check if save path exists
        print(args.save_path)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        # save prediction as png
        sam_seg_int = np.where(sam_seg_int.cpu().numpy() == 1, 255, 0).astype(np.uint8)
        if len(img_names) > 1:
            for i in range(len(img_names)):
                print(f"{img_names[i]} Dice score: ", dice._dice_scores[-1][i], "Hausdorff Distance: ", batch_hd95[i])
                io.imsave(join(args.save_path, f"pred_{img_names[i]}"), sam_seg_int[i])
        else:
            print(f"{img_names[0]} Dice score: ", dice._dice_scores[-1][0], "Hausdorff Distance: ", batch_hd95[0])
            io.imsave(join(args.save_path, f"pred_{img_names[0]}"), sam_seg_int)

    avg_dice_score = dice.compute().item()
    for hd in hd95_scores:
        if hd == -1:
            hd95_scores.remove(hd)
    avg_hd95 = np.mean(hd95_scores)

    print(f"Average Dice Score: {avg_dice_score:.4f}, Average Hausdorff Distance: {avg_hd95:.4f}")


if __name__ == "__main__":
    main()
