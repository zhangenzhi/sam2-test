import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_score(inputs, targets, smooth=1):    
    
    # print(inputs.shape, targets.shape)
    #flatten label and prediction tensors
    pred = torch.flatten(inputs[:,1,:,:])
    true = torch.flatten(targets[:,1,:,:])
    
    intersection = (pred * true).sum()
    coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)   

    return coeff  

def dice_score_plot(inputs, targets, smooth=1):     
    #flatten label and prediction tensors
    pred = inputs[...,0].flatten()
    true = targets[...,0].flatten()
    
    intersection = (pred * true).sum()
    coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)   
    return coeff 

# Define the Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        #flatten label and prediction tensors
        pred = torch.flatten(inputs[:,:,:,:])
        true = torch.flatten(targets[:,:,:,:])
        intersection = (pred * true).sum()
        dice_loss = 1 - (2.*intersection + self.smooth)/(pred.sum() + true.sum() + self.smooth)  
        
        return dice_loss

class DiceCLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-6):
        super(DiceCLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        
    def forward(self, preds, labels, exclude_background=True):
        
        preds = F.softmax(preds, dim=1)  # Apply softmax to the predictions if they are logits
    
        # If excluding background, ignore the first class
        start_class = 1 if exclude_background else 0
        dice_loss = 0
        
        for cls in range(start_class, preds.shape[1]):
            pred_cls = preds[:, cls, :, :]  # Probability map for class `cls`
            true_cls = labels[:, cls, :, :]  # Ground truth for class `cls`
            
            intersection = (pred_cls * true_cls).sum(dim=(1, 2))  # Compute intersection (B)
            union = pred_cls.sum(dim=(1, 2)) + true_cls.sum(dim=(1, 2))  # Compute union (B)
            
            dice_coeff = (2 * intersection + self.eps) / (union + self.eps)  # Compute Dice coefficient for class `cls`
            dice_loss += 1 - dice_coeff  # Dice loss for class `cls` is 1 - Dice coefficient
        dice_loss = dice_loss/(self.num_classes-1)
        return dice_loss.mean()  # Average Dice loss across all classes and batch


class DiceCELoss(nn.Module):
    def __init__(self, num_classes, eps=1e-6, weight=None, lambda_dice=0.5, lambda_ce=0.5):
        super(DiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)  # Optional: weight for class imbalance
        self.lambda_dice = lambda_dice  # Weighting for Dice loss
        self.lambda_ce = lambda_ce  # Weighting for CE loss

    def forward(self, preds, labels, exclude_background=True):
        
        # Apply softmax to the predictions if they are logits
        preds_soft = F.softmax(preds, dim=1)

        # Cross-Entropy loss (using raw logits, as PyTorch's CE function expects logits)
        ce_loss = self.ce_loss(preds, labels.to(torch.float32))

        # One-hot encode the labels if necessary
        if labels.ndimension() == 3:  # If labels are not one-hot encoded (shape [B, H, W])
            labels_onehot = F.one_hot(labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        else:
            labels_onehot = labels.float()  # Assumes labels are already one-hot encoded

        # If excluding background, ignore the first class
        start_class = 1 if exclude_background else 0
        dice_loss = 0

        # Iterate through classes
        for cls in range(start_class, preds_soft.shape[1]):
            pred_cls = preds_soft[:, cls, :, :]  # Probability map for class `cls`
            true_cls = labels_onehot[:, cls, :, :]  # Ground truth for class `cls`
            
            intersection = (pred_cls * true_cls).sum(dim=(1, 2))  # Compute intersection
            union = pred_cls.sum(dim=(1, 2)) + true_cls.sum(dim=(1, 2))  # Compute union
            
            dice_coeff = (2 * intersection + self.eps) / (union + self.eps)  # Compute Dice coefficient for class `cls`
            dice_loss += (1 - dice_coeff)  # Dice loss for class `cls` is 1 - Dice coefficient

        # Normalize Dice loss
        dice_loss /= (self.num_classes - 1) if exclude_background else self.num_classes
        
        # Final combined Dice and Cross-Entropy loss
        total_loss = self.lambda_dice * dice_loss.mean() + self.lambda_ce * ce_loss
        return total_loss

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target, smooth=1e-6):
        # Apply sigmoid to the prediction (if it has not been applied already)
        pred = torch.sigmoid(pred)
        
        # Flatten the tensors
        pred = pred.view(pred.size(0), pred.size(1), -1)  # (batch_size, num_classes, H*W)
        target = target.view(target.size(0), target.size(1), -1)  # (batch_size, num_classes, H*W)
        
        iou_per_class = []
        for i in range(pred.size(1)):  # Loop through each class/channel
            # Intersection and Union for each class
            intersection = (pred[:, i] * target[:, i]).sum(dim=1)
            union = (pred[:, i] + target[:, i]).sum(dim=1) - intersection
            
            # IoU calculation with a smoothing factor
            iou = (intersection + smooth) / (union + smooth)
            iou_per_class.append(iou)
        
        # Mean IoU across all classes
        mean_iou = torch.mean(torch.stack(iou_per_class, dim=1), dim=1)
        
        # IoU loss is 1 - mean IoU
        return 1 - mean_iou.mean()

class CE_IoULoss(nn.Module):
    def __init__(self, num_classes=4, weight_ce=1.0, weight_iou=1.0):
        super(CE_IoULoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.iou_loss = IoULoss()
        self.weight_ce = weight_ce
        self.weight_iou = weight_iou

    def forward(self, pred, target):
        # Cross-Entropy Loss (pred needs to be raw logits for CE Loss)
        ce_loss = self.ce_loss(pred, target.argmax(dim=1))  # Use target.argmax(dim=1) for multi-class CE

        # IoU Loss (convert one-hot target for IoU calculation)
        iou_loss = self.iou_loss(pred, target)

        # Combine the losses
        total_loss = self.weight_ce * ce_loss + self.weight_iou * iou_loss
        return total_loss


class DiceBLoss(nn.Module):
    def __init__(self, weight=0.5, num_class=2,  smooth=1, act=True):
        super(DiceBLoss, self).__init__()
        self.weight = weight
        self.num_class = num_class
        self.smooth = smooth
        self.act = act

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if self.act:
            inputs = F.sigmoid(inputs)       
        
        # pred = torch.flatten(inputs)
        # true = torch.flatten(targets)
        
        # #flatten label and prediction tensors
        pred = torch.flatten(inputs[:,1:,:,:])
        true = torch.flatten(targets[:,1:,:,:])
        
        intersection = (pred * true).sum()
        coeff = (2.*intersection + self.smooth)/(pred.sum() + true.sum() + self.smooth) 
                                   
        dice_loss = 1 - (2.*intersection + self.smooth)/(pred.sum() + true.sum() + self.smooth)  
        BCE = F.binary_cross_entropy(pred, true, reduction='mean')
        dice_bce = self.weight*BCE + (1-self.weight)*dice_loss
        # dice_bce = dice_loss 
        
        return dice_bce
    