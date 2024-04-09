import os
import time
from glob import glob
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
from data import SpleenDataset, SpleenDataset_Recon
from model_transunet import TransUnet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(model, seg_loader, recon_loader, optimizer, segmentation_loss_fn, reconstruction_weight, device):
    epoch_segmentation_loss = 0.0
    epoch_reconstruction_loss = 0.0
    model.train()

    for (x_seg, y_seg), (x_recon, y_recon) in zip(seg_loader, recon_loader):
        x_seg, y_seg = x_seg.to(device, dtype=torch.float32), y_seg.to(device, dtype=torch.float32)
        x_recon, y_recon = x_recon.to(device, dtype=torch.float32), y_recon.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        # Segmentation Task
        segmentation_output, _ = model(x_seg)
        segmentation_loss = segmentation_loss_fn(segmentation_output, y_seg)
        # Reconstruction Task
        _, reconstruction_output = model(x_seg)
        reconstruction_loss = nn.MSELoss()(reconstruction_output, x_seg)

        # Overall loss is a weighted sum of segmentation and reconstruction losses
        loss = segmentation_loss + (reconstruction_weight * reconstruction_loss)

        loss.backward()
        optimizer.step()

        epoch_segmentation_loss += segmentation_loss.item()
        epoch_reconstruction_loss += reconstruction_loss.item()

    epoch_segmentation_loss = epoch_segmentation_loss / len(seg_loader)
    epoch_reconstruction_loss = epoch_reconstruction_loss / len(recon_loader)
    return epoch_segmentation_loss, epoch_reconstruction_loss

def evaluate(model, seg_loader, recon_loader, segmentation_loss_fn, reconstruction_weight, device):
    epoch_segmentation_loss = 0.0
    epoch_reconstruction_loss = 0.0

    model.eval()
    with torch.no_grad():
        for (x_seg, y_seg), (x_recon, y_recon) in zip(seg_loader, recon_loader):
            x_seg, y_seg = x_seg.to(device, dtype=torch.float32), y_seg.to(device, dtype=torch.float32)
            x_recon, y_recon = x_recon.to(device, dtype=torch.float32), y_recon.to(device, dtype=torch.float32)

            # Segmentation Task
            segmentation_output, reconstruction_output = model(x_seg)
            segmentation_loss = segmentation_loss_fn(segmentation_output, y_seg)

            # Reconstruction Task
            reconstruction_loss = F.mse_loss(reconstruction_output, x_recon)

            # Overall loss is a weighted sum of segmentation and reconstruction losses
            loss = segmentation_loss + (reconstruction_weight * reconstruction_loss)

            epoch_segmentation_loss += segmentation_loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()

        epoch_segmentation_loss = epoch_segmentation_loss / len(seg_loader)
        epoch_reconstruction_loss = epoch_reconstruction_loss / len(recon_loader)
    return epoch_segmentation_loss, epoch_reconstruction_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("Model_Weights_MultiTask")

    """ Load dataset """
    train_x_Seg = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_TransUNet\train_data\image\*'))
    train_y_Seg = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_TransUNet\train_data\mask\*'))

    valid_x_Seg = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_TransUNet\test_data\image\*'))
    valid_y_Seg = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_TransUNet\test_data\mask\*'))

    train_x_Recon = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_TransUNet\train_data\image\*'))
    train_y_Recon = train_x_Recon

    valid_x_Recon = sorted(glob(r'D:\Risab\Medical_Experiments\Spleen_TransUNet\test_data\image\*'))
    valid_y_Recon = valid_x_Recon

    data_str_Seg = f"Dataset Size for Segmentation:\nTrain: {len(train_x_Seg)} - Valid: {len(valid_x_Seg)}\n"
    data_str_Recon = f"Dataset Size for Reconstruction:\nTrain: {len(train_x_Recon)} - Valid: {len(valid_x_Recon)}\n"

    print(data_str_Seg)
    print(data_str_Recon)

    """ Hyperparameters """
    H = 256
    W = 256
    size = (H, W)
    batch_size = 8
    num_epochs = 200
    lr = 1e-4
    checkpoint_path = r'D:\Risab\Medical_Experiments\Spleen_TransUNet\Model_Weights_MultiTask\MultiTask_checkpoint_v1.pth'

    """ Dataset and loader """
    train_dataset_Seg = SpleenDataset(train_x_Seg, train_y_Seg)
    valid_dataset_Seg = SpleenDataset(valid_x_Seg, valid_y_Seg)

    train_dataset_Recon = SpleenDataset_Recon(train_x_Recon, train_y_Recon)
    valid_dataset_Recon = SpleenDataset_Recon(valid_x_Recon, valid_y_Recon)

    train_loader_Seg = DataLoader(
        dataset=train_dataset_Seg,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_loader_Seg = DataLoader(
        dataset=valid_dataset_Seg,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    train_loader_Recon = DataLoader(
        dataset=train_dataset_Recon,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_loader_Recon = DataLoader(
        dataset=valid_dataset_Recon,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransUnet(in_channels=3, img_dim=256, vit_blocks=8, vit_dim_linear_mhsa_block=2048, classes=3)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)
    segmentation_loss_fn = DiceBCELoss()
    reconstruction_weight = 0.2  # You can adjust the weight for reconstruction loss

    """ Training the model """
    best_valid_loss = float("inf")

    # Lists to store the loss values
    train_segmentation_losses = []
    train_reconstruction_losses = []
    valid_segmentation_losses = []
    valid_reconstruction_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()

        train_segmentation_loss, train_reconstruction_loss = train(
            model, train_loader_Seg, train_loader_Recon,
            optimizer, segmentation_loss_fn, reconstruction_weight, device
        )
        valid_segmentation_loss, valid_reconstruction_loss = evaluate(
            model, valid_loader_Seg, valid_loader_Recon,
            segmentation_loss_fn, reconstruction_weight, device
        )

        """ Saving the model """
        if valid_segmentation_loss + valid_reconstruction_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_segmentation_loss + valid_reconstruction_loss:2.4f}. Saving checkpoints..."
            print(data_str)

            best_valid_loss = valid_segmentation_loss + valid_reconstruction_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Segmentation Loss: {train_segmentation_loss:.3f}\n'
        data_str += f'\tTrain Reconstruction Loss: {train_reconstruction_loss:.3f}\n'
        data_str += f'\t Val. Segmentation Loss: {valid_segmentation_loss:.3f}\n'
        data_str += f'\t Val. Reconstruction Loss: {valid_reconstruction_loss:.3f}\n'
        print(data_str)
        # Append the loss values to the lists
        train_segmentation_losses.append(train_segmentation_loss)
        train_reconstruction_losses.append(train_reconstruction_loss)
        valid_segmentation_losses.append(valid_segmentation_loss)
        valid_reconstruction_losses.append(valid_reconstruction_loss)

        # Plot the training curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+2), train_segmentation_losses, label='Train Segmentation Loss')
        plt.plot(range(1, epoch+2), train_reconstruction_losses, label='Train Reconstruction Loss')
        plt.plot(range(1, epoch+2), valid_segmentation_losses, label='Valid Segmentation Loss')
        plt.plot(range(1, epoch+2), valid_reconstruction_losses, label='Valid Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.savefig('Training_Curve_TransUNet.png')
