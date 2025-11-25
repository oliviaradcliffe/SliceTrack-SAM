import torch
import numpy as np
import cv2
import SimpleITK as sitk
from medpy import metric

class DiceMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._dice_scores = []
 
    def update(self, y_pred, y_true):
        """
        Adds the predictions and targets to the metric.
 
        Args:
            y_pred: torch.Tensor, shape (B, H, W). The predicted labels.
            y_true: torch.Tensor, shape (B, H, W). The target labels.
        """
        if len(y_pred.shape) == 2:
            B = 1
        else:
            B = y_pred.shape[0]

        dice_score_for_batch = torch.zeros(B, self.num_classes)
        for class_index in range(self.num_classes):
            y_true_for_class = (y_true == class_index).float()
            y_pred_for_class = (y_pred == class_index).float()

            dice_score_for_class = 2 * (y_true_for_class * y_pred_for_class).sum(dim=(-2, -1)) / (y_true_for_class.sum(dim=(-2, -1)) + y_pred_for_class.sum(dim=(-2, -1)))
            
            # replace nan values (when pred + gt = 0) with 1
            if torch.isnan(dice_score_for_class).any():
                dice_score_for_class[torch.isnan(dice_score_for_class)] = 1
    
            dice_score_for_batch[:, class_index] = dice_score_for_class
 
        self._dice_scores.append(dice_score_for_batch)
 
    def compute(self):
        dice_scores = torch.cat(self._dice_scores, dim=0)
        # aggregate across batches and classes
        return dice_scores.mean()
    
def Average(lst): 
    return sum(lst) / len(lst) 

class Hausdorff:
    def __init__(self):
        self.hd95_scores = []

    def update(self, y_pred, y_true):
        spacing =  0.033586
        B = y_pred.shape[0]
        num = 0
        hd95 = 0
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()


        for i in range(B):
            pred_sum = y_pred[i,:,:].sum()
            gt_sum = y_true[i,:,:].sum()
            if pred_sum >0 and gt_sum>0:
                num+=1
                hd95 += metric.binary.hd95(y_pred[i,:,:], y_true[i,:,:])
        if num>0:
            hd95 = (hd95*spacing)/num
            self.hd95_scores.append(hd95)


    def compute(self):
        if not self.hd95_scores:
            return 0
        return Average(self.hd95_scores)


# Annotation-guided binary cross entropy loss (AG-BCE) - from MICROSEGNET paper
def attention_BCE_loss(h_W, y_true, y_pred, y_std, ks = 5):

    print( "Using Attention BCE Loss")
    
    number_of_pixels = y_true.shape[0]*y_true.shape[1]*y_true.shape[2]

    y_true_np = y_true.cpu().detach().numpy()
    y_std_np = y_std.cpu().detach().numpy()

    hard = cv2.bitwise_xor(y_true_np, y_std_np)
    hard = hard.astype(np.uint8)
    
    # Apply dilation operation to hard regions
    kernel_size = ks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for i in range(hard.shape[0]):
        hard[i] = cv2.dilate(hard[i], kernel)
    hard = hard.astype(np.float32)

    easy = abs(hard-1)
    hard = torch.tensor(hard).cuda()
    easy = torch.tensor(easy).cuda()

    epsilon = 0.000001
    beta = 0.5

    loss = -beta*torch.mul(y_true,torch.log(y_pred + epsilon)) - (1.0 - beta)*torch.mul(1.0-y_true,torch.log(1.0 - y_pred + epsilon))
    hard_loss = torch.sum(torch.mul(loss,hard))
    easy_loss = torch.sum(torch.mul(loss,easy))

    LOSS = ((1/(1+h_W))*easy_loss + (h_W/(1+h_W))*hard_loss)/(number_of_pixels)

    return LOSS