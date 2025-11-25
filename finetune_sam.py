import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from skimage import io
from statistics import mean
from tqdm import tqdm
from torch.utils.data import DataLoader
join = os.path.join
from torchvision.transforms import v2 as T
import torch.nn.functional as F
from src.sam_prompting import SliceTrackSam
from src.dataset import MicroUSDataset
from src.transforms import Transform
from src.utils import DiceMetric, Hausdorff, attention_BCE_loss 
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

import wandb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-root",
    type=str,
    default="./data",
    help="path to dataset",
)
parser.add_argument(
    "-device",
    type=str,
    default="cuda",
    help="device to use",
)
parser.add_argument(
    "-backbone",
    type=str,
    default="medsam",
    help="backbone model. medsam or sam",
)
parser.add_argument(
    "-save_path",
    type=str,
    default="./work_dir/finetune_MedSAM",
    help="path to save the model",
)
parser.add_argument(
    "-saved_model_name",
    type=str,
    default="sam_finetuned",
    help="name of the new trained model",
)
parser.add_argument(
    "-img_size",
    type=int,
    default=1024,
    help="image size",
)
parser.add_argument(
    "-mask_size",
    type=int,
    default=256,
    help="mask size",
)
parser.add_argument(
    "-crop_pad",
    type=bool,
    default=False,
    help="Augment with cropping top and bottom with padding. Default is False",
)
parser.add_argument(
    "-augment",
    type=bool,
    default=False,
    help="augment data. Default is False",
)
parser.add_argument(
    "-h_flip_prob",
    type=float,
    default=0.5,
    help="horizontal flip probability. Default is 0.5",
)
parser.add_argument(
    "--use_prev_mask",
    action="store_true",
    help="use previous mask prompt. Default is False",
)
parser.add_argument(
    "-random_prev_mask_prob",
    type=float,
    default=0.5,
    help="random previous mask probability. Default is 0.5",
)
parser.add_argument(
    "--use_float_prompt",
    action="store_true",
    help="use floating point prompts",
)
parser.add_argument(
    "--useMultiImage",
    action="store_true",
    help="use multiple images as input",
)
parser.add_argument(
    "-optimizer",
    type=str,
    default="adam",
    help="optimizer. Default is adam",
)
parser.add_argument(
    "-loss_fn",
    type=str,
    default="bce",
    help="loss function. Default is bce. Options: mse, bce",
)
parser.add_argument(
    "-batch_size",
    type=int,
    default=12,
    help="batch size",
)
parser.add_argument(
    "--freeze_img_encoder",
    action="store_true",
    help="freeze image encoder.",
)
parser.add_argument(
    "--freeze_mask_decoder",
    action="store_true", 
    help="freeze mask decoder.",
)
parser.add_argument(
    "-num_epochs",
    type=int,
    default=20,
    help="number of epochs",
)
parser.add_argument(
    '--use_scheduler',
    action='store_true',
    help='Use learning rate scheduler'
)
parser.add_argument(
    "-lr_scheduler",
    type=str,
    default="lambda",
    help="learning rate scheduler. Default is lambda. Options: step, plateau, cosine, cosine_warm",
)
parser.add_argument(
    "-decoder_lr",
    type=float,
    default=1e-3,
    help="learning rate",
)
parser.add_argument(
    "-encoder_lr",
    type=float,
    default=1e-5,
    help="learning rate",
)
parser.add_argument(
    "-wd",
    type=float,
    default=1e-4,
    help="weight decay",
)
parser.add_argument(
    "-patience",
    type=int,
    default=5,
    help="patience for early stopping",
)
args = parser.parse_args()

# parser = argparse.ArgumentParser()


def main():
    # VARIABLES -------------------------------------
    
    if args.loss_fn == "bce":
        loss_fn = torch.nn.BCELoss()
    elif args.loss_fn == "bcel":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss_fn == "mse":
        loss_fn = torch.nn.MSELoss()
    elif args.loss_fn == "agbce":
        loss_fn = "agbce"  # custom loss handled in prep_and_forward


    print("arguments: ")
    print("\t root: ", args.root)
    print("\t model_save_path: ", args.save_path)
    print("\t saved_model_name: ", args.saved_model_name)
    print("\t backbone: ", args.backbone)
    print("\t device: ", args.device)
    print("\t crop_pad: ", args.crop_pad)
    print("\t augment: ", args.augment)
    print("\t h_flip_prob: ", args.h_flip_prob)
    print("\t use prev_mask: ", args.use_prev_mask)
    print("\t random_prev_mask_prob: ", args.random_prev_mask_prob)
    print("\t use_float_prompt: ", args.use_float_prompt)
    print("\t useMultiImage: ", args.useMultiImage)
    print("\t batch_size: ", args.batch_size)
    print("\t num_epochs: ", args.num_epochs)
    print("\t encoder learning rate: ", args.encoder_lr)
    print("\t decoder learning rate: ", args.decoder_lr)
    print("\t optimizer: ", args.optimizer)
    print("\t use_scheduler: ", args.use_scheduler)
    print("\t lr_scheduler: ", args.lr_scheduler)
    print("\t weight decay: ", args.wd)
    print("\t loss function: ", loss_fn)

    wandb.init(
        # set the wandb project where this run will be logged
        project="SPIE2025",

        # track hyperparameters and run metadata
        config={
        "encoder learning_rate": args.encoder_lr,
        "decoder learning_rate": args.decoder_lr,
        "use_scheduler": args.use_scheduler,
        "lr_scheduler": args.lr_scheduler,
        "optimizer": args.optimizer,
        "weight_decay": args.wd,
        "backbone": args.backbone,
        "saved model name": args.saved_model_name,
        "loss_function": loss_fn,
        "use prevMask": args.use_prev_mask,
        "random_prev_mask_prob": args.random_prev_mask_prob,
        "use_float_prompt": args.use_float_prompt,
        "useMultiImage": args.useMultiImage,
        "crop_pad": args.crop_pad,
        "augment": args.augment,
        "h_flip_prob": args.h_flip_prob,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        }
    )

    # LOAD MODEL -------------------------------------

    medsam_model = SliceTrackSam(
        floating_point_prompts=["position"],
        use_prev_mask_prompt=args.use_prev_mask,
        sam_backbone=args.backbone,
        random_prev_mask_prob = args.random_prev_mask_prob,
        freeze_mask_decoder=args.freeze_mask_decoder,
        freeze_image_encoder=args.freeze_img_encoder,
        image_size=args.img_size,
    )
    medsam_model.to(device=args.device)

    img_encoder_params, prompt_encoder_params, mask_decoder_params = medsam_model.get_params()

    print("...Loaded model")
    # print(sam_model)

    # LOAD DATASET -------------------------------------
    if args.loss_fn == "agbce":
        tr_dataset = MicroUSDataset(args.root, split="train", useFloatPrompt=args.use_float_prompt, usePrevMask=args.use_prev_mask, useMultiImage=args.useMultiImage, useAGBCE=True, transform=Transform(output_size=args.img_size, mask_size=args.mask_size, augment=args.augment, h_flip_prob=args.h_flip_prob, crop_pad=args.crop_pad))
        val_dataset = MicroUSDataset(args.root, split="val", useFloatPrompt=args.use_float_prompt, useMultiImage=args.useMultiImage, useAGBCE=True, transform=Transform(output_size=args.img_size, mask_size=args.mask_size, augment=False, h_flip_prob=0))
    else:
        tr_dataset = MicroUSDataset(args.root, split="train", useFloatPrompt=args.use_float_prompt, usePrevMask=args.use_prev_mask, useMultiImage=args.useMultiImage, transform=Transform(output_size=args.img_size, mask_size=args.mask_size, augment=args.augment, h_flip_prob=args.h_flip_prob, crop_pad=args.crop_pad))
        val_dataset = MicroUSDataset(args.root, split="val", useFloatPrompt=args.use_float_prompt, useMultiImage=args.useMultiImage, transform=Transform(output_size=args.img_size, mask_size=args.mask_size, augment=False, h_flip_prob=0))

    print("tr_dataset (num images): ", len(tr_dataset))
    print("val_dataset (num images): ", len(val_dataset))

    # Create a DataLoader
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    print("tr_dataloader: ", len(tr_dataloader))
    print("val_dataloader: ", len(val_dataloader))

    print("...Loaded dataset for training")

    # FINETUNING -----------------------------------
    # Set up the optimizer, hyperparameter tuning will improve performance her
    params = []
    params.append({"params": prompt_encoder_params, "lr": args.encoder_lr})

    if not args.freeze_img_encoder:
            params.append({"params": img_encoder_params, "lr": args.encoder_lr})
    if not args.freeze_mask_decoder:
            params.append({"params": mask_decoder_params, "lr": args.decoder_lr})

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, weight_decay=args.wd)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, weight_decay=args.wd)
    
    if args.use_scheduler:
        if args.lr_scheduler == "lambda":
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)
        elif args.lr_scheduler == "step":
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        elif args.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        elif args.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
        elif args.lr_scheduler == "cosine_warm":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=-1)

    best_dice = 0
    best_val_loss = 1e10
    no_improvement = 0

    print("...Training")

    for epoch in range(args.num_epochs):
        torch.cuda.empty_cache()

        print(f"Epoch {epoch}")

        # train
        train_loss, train_dice, train_hd95 = run_train_epoch(tr_dataloader, medsam_model, optimizer, loss_fn, desc="train")

        # validate
        val_loss, val_dice, val_hd95 = run_eval_epoch(epoch, val_dataloader, medsam_model, loss_fn, desc="val")

        # Step the scheduler
        if args.use_scheduler:
            scheduler.step()
        
        lrs = []
        for i,param in enumerate(optimizer.param_groups):
            lrs.append(param["lr"])
        #     wandb.log({f"learning_rate {i}": param["lr"]})
             

        # make saved model directory if not exists
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        torch.save(medsam_model.state_dict(), join(args.save_path, f'{args.saved_model_name}.pth'))
        # save the best model

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(medsam_model.state_dict(), join(args.save_path, f"{args.saved_model_name}_best_val_dice.pth"))

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Learning rate(s) {lrs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val HD95: {val_hd95:.4f}")
        wandb.log({"train_loss": train_loss, "train_dice": train_dice, "train_hd95": train_hd95, "Learning rate(s)": lrs, "val_loss": val_loss, "val_dice": val_dice, "val_hd95": val_hd95})

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                print("Early stopping!")
                break


    wandb.finish()
    

def prep_and_forward(batch, model, loss_fn, dice, hd95, desc="train"):

        # batch = batch.copy()

        # extract relevant data and move to gpu
        if args.use_prev_mask and desc == "train":
            if args.use_float_prompt:
                if args.loss_fn == "agbce":
                    images, gts, img_names, prev_masks, slice_positions, non_expert_labels, H, W = batch
                    prev_masks = prev_masks.unsqueeze(1).to(args.device)
                    slice_positions = slice_positions.unsqueeze(1).to(args.device)
                    non_expert_labels = non_expert_labels.to(args.device)
                else:
                    images, gts, img_names, prev_masks, slice_positions, H, W = batch
                    prev_masks = prev_masks.unsqueeze(1).to(args.device)
                    slice_positions = slice_positions.unsqueeze(1).to(args.device)
            else:
                if args.loss_fn == "agbce":
                    images, gts, img_names, prev_masks, non_expert_labels, H, W = batch
                    prev_masks = prev_masks.unsqueeze(1).to(args.device)
                    non_expert_labels = non_expert_labels.to(args.device)
                else:
                    images, gts, img_names, prev_masks, H, W = batch
                    prev_masks = prev_masks.unsqueeze(1).to(args.device)
        else:
            if args.use_float_prompt:
                if args.loss_fn == "agbce":
                    images, gts, img_names, slice_positions, non_expert_labels, H, W = batch
                    slice_positions = slice_positions.unsqueeze(1).to(args.device)
                    non_expert_labels = non_expert_labels.to(args.device)
                else:
                    images, gts, img_names, slice_positions, H, W = batch
                    slice_positions = slice_positions.unsqueeze(1).to(args.device)
            else:
                if args.loss_fn == "agbce":
                    images, gts, img_names, non_expert_labels, H, W = batch
                    non_expert_labels = non_expert_labels.to(args.device)
                else:
                    images, gts, img_names, H, W = batch
        
        images = images.to(args.device)
        gts = gts.to(args.device)

        # run the model
        if args.use_prev_mask and desc == "train":
            if args.use_float_prompt:
                heatmap_logits = model(
                    images,
                    prev_mask=prev_masks,
                    position=slice_positions
                )
            else:
                heatmap_logits = model(
                    images,
                    prev_mask=prev_masks
                    # position=None
                )
        else:
            if args.use_float_prompt:
                heatmap_logits = model(
                    images,
                    prev_mask=None,
                    position=slice_positions
                )
            else:
                heatmap_logits = model(
                    images,
                    prev_mask=None
                    # position=None
                )

        if torch.any(torch.isnan(heatmap_logits)):
            print("NaNs in heatmap logits")

        pred = F.interpolate(
            heatmap_logits,
            size=(H[0], W[0]),
            mode="bilinear",
            align_corners=False,
        )

        # sigmoid activation
        pred = torch.sigmoid(pred)

        # dice calculation
        sam_seg_int = (pred > 0.5).squeeze()

        dice.update(sam_seg_int, gts)

        # hd95 calculation
        pred_binary = torch.where(pred >= 0.5, 1, 0).squeeze()
        gt_binary = torch.where(gts >= 0.5, 1, 0)
        hd95.update(pred_binary, gt_binary)

        # agbce loss
        if args.loss_fn == "agbce":
            hard_weight=4
            loss = attention_BCE_loss(hard_weight, gts, pred.squeeze(1), non_expert_labels, ks=5)
  
        else:
            # loss calculation
            loss = loss_fn(
                pred.squeeze(1), gts,
            )

        return (
            loss,
            pred,
            dice.compute().item(),
            hd95.compute(),
            batch,
        )

def run_train_epoch(loader, model, optimizer, loss_fn, desc="train"):
        # setup epoch
        model.train()
        losses = []
        avg_dice = []
        avg_hd95 = []
        dice_scores = DiceMetric(num_classes=2)
        hd95_scores = Hausdorff()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            (
                loss,
                heatmap_logits,
                dice,
                hd95,
                batch,
            ) = prep_and_forward(batch, model, loss_fn, dice_scores, hd95_scores, desc=desc)

            # backward pass
            loss.backward()

            # optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # log metrics
            losses.append(loss.item())
            avg_dice.append(dice)
            avg_hd95.append(hd95)

        epoch_loss = mean(losses)

        # compute and log metrics
        return epoch_loss, mean(avg_dice), mean(avg_hd95)

@torch.no_grad()
def run_eval_epoch(epoch, loader, model, loss_fn, desc="eval"):
    model.eval()
    losses = []
    avg_dice = []
    avg_hd95 = []
    dice_scores = DiceMetric(num_classes=2)
    hd95_scores = Hausdorff()

    for val_iter, batch in enumerate(tqdm(loader, desc=desc)):
        (
            loss,
            pred,
            dice,
            hd95,
            batch,
        ) = prep_and_forward(batch, model, loss_fn, dice_scores, hd95_scores, desc=desc)
        
        if val_iter % 10 == 0:
            # save prediction as png
            sam_seg_int = (pred > 0.5).squeeze()
            img = T.Resize((sam_seg_int[0].shape[0], sam_seg_int[0].shape[1]))(batch[0][0].cpu())
            val_img = wandb.Image(
                img,
                masks={
                    "predictions": {"mask_data": sam_seg_int[0].cpu().numpy(), "class_labels": {0:"background", 1:"prostate"}},
                    "ground_truth": {"mask_data": batch[1][0].cpu().numpy()},
                },
            )
            wandb.log({f"val_{epoch}_{val_iter}_pred_slice{batch[2][0]}": val_img})
            sam_seg_int = np.where(sam_seg_int[0].cpu().numpy() == 1, 255, 0).astype(np.uint8)
            print(f"{val_iter} {batch[3][0]} Val batch Dice score: ", np.round(dice, 2), " Val batch Loss: ", np.round(loss.item(), 2))
            # io.imsave(join(args.save_path , f"val_{epoch}/val_{val_iter}_pred_{batch[3][-0]}"), sam_seg_int, check_contrast=False, format="png")


        losses.append(loss.item())
        avg_dice.append(dice)
        avg_hd95.append(hd95)

    epoch_loss = mean(losses)
    return epoch_loss, mean(avg_dice),  mean(avg_hd95)

if __name__ == "__main__":
    main()
