import os
import sys
sys.path.append("./")

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from model.sam2 import SAM2
from dataset.kits import KITS2D
from utiliz.losses import DiceLoss


import logging
def dice_score(inputs, targets, smooth=1):    
    
    # print(inputs.shape, targets.shape)
    #flatten label and prediction tensors
    pred = torch.flatten(inputs[:,:,:,:])
    true = torch.flatten(targets[:,:,:,:])
    
    intersection = (pred * true).sum()
    coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)   

    return coeff  

# Configure logging
def log(args):
    os.makedirs(args.savefile, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.savefile, "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
 
def main(args):
    log(args=args)
    best_val_score = 0.0
    global_step = 0
    
    # Create an instance of the U-Net model and other necessary components
    model = SAM2(image_shape=(args.resolution,  args.resolution),
            output_dim=1, 
            pretrain=args.pretrain
            )
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    # Move the model to GPU
    model.to(device)
    if args.reload:
        if os.path.exists(os.path.join(args.savefile, "best_score_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.savefile, "best_score_model.pth")))

    # Split the dataset into train, validation, and test sets
    dataset = KITS2D(args.data_dir)
    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    test_size = val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, dataset_size))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = val_loader
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, test_size))

    # Training loop
    num_epochs = args.epoch
    train_losses = []
    val_losses = []
    output_dir = args.savefile  # Change this to the desired directory
    os.makedirs(output_dir, exist_ok=True)
    
    import time
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_score = 0.0
        start_time = time.time()
        for batch in train_loader:
            step_time = time.time()
            images, masks = batch
            images, masks = images.to(device), masks.to(device)  # Move data to GPU
            
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)
            score = dice_score(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_score += score.item()
            global_step +=1
            print(f"step {global_step}, step loss:{loss.item()}, step score:{score.item()}, cost:{time.time()-step_time}")
            
        end_time = time.time()
        epoch_train_loss /= len(train_loader)
        epoch_train_score /= len(train_loader)
        train_losses.append(epoch_train_loss)
        # scheduler.step()
        print("epoch cost:{}, sec/img:{}, train score:{}".format(end_time-start_time,(end_time-start_time)/train_size, epoch_train_score))


        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_score = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)  # Move data to GPU
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                
                loss = criterion(outputs, masks)
                score = dice_score(outputs, masks)
                
                epoch_val_loss += loss.item()
                epoch_val_score += score.item()

        epoch_val_loss /= len(val_loader)
        epoch_val_score /= len(val_loader)
        val_losses.append(epoch_val_loss)
                
        if epoch_val_score > best_val_score:
            best_val_score = epoch_val_score
            torch.save(model.state_dict(), os.path.join(args.savefile, "best_score_model.pth"))
            logging.info(f"Model save with dice score {best_val_score} at epoch {epoch}")
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Score: {epoch_val_score:.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Score: {epoch_val_score:.4f}.")

    # Save train and validation losses
    train_losses_path = os.path.join(output_dir, 'train_losses.pth')
    val_losses_path = os.path.join(output_dir, 'val_losses.pth')
    torch.save(train_losses, train_losses_path)
    torch.save(val_losses, val_losses_path)

    # Test the model
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)  # Move data to GPU
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            
            loss = criterion(outputs, masks)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default="paip", help='name of the dataset.')
    parser.add_argument('--data_dir', default="./dataset/paip/output_images_and_masks", 
                        help='base path of dataset.')
    parser.add_argument('--resolution', default=512, type=int,
                        help='resolution of img.')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='patch size.')
    parser.add_argument('--pretrain', default="sam2-t", type=str,
                        help='Use SAM2 pretrained weigths.')
    parser.add_argument('--reload', default=True, type=bool,
                        help='continue weigths training.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./sam2",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    main(args)
