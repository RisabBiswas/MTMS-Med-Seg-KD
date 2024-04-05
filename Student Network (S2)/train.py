import os
import time
from glob import glob
import torch.optim as optim 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import SpleenDataset
from model import build_small_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
import matplotlib.pyplot as plt

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("Model_File")

    """ Load dataset """
    train_x = sorted(glob(r'D:\Risab\Thesis_Experiment_v2\Spleen_Seg_Student_v2\train_data\image\*'))
    train_y = sorted(glob(r'D:\Risab\Thesis_Experiment_v2\Spleen_Seg_Student_v2\train_data\mask\*'))

    valid_x = sorted(glob(r'D:\Risab\Thesis_Experiment_v2\Spleen_Seg_Student_v2\test_data\image\*'))
    valid_y = sorted(glob(r'D:\Risab\Thesis_Experiment_v2\Spleen_Seg_Student_v2\test_data\mask\*'))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 256
    W = 256
    size = (H, W)
    batch_size = 16
    num_epochs = 120
    lr = 1e-4
    checkpoint_path = r'D:\Risab\Thesis_Experiment_v2\Spleen_Seg_Student_v2\Model_File\checkpoint.pth'

    """ Dataset and loader """
    train_dataset = SpleenDataset(train_x, train_y)
    valid_dataset = SpleenDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_small_unet()
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),lr=1.5e-4, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.05, amsgrad=False)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    train_segmentation_losses = []
    valid_segmentation_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)

        train_segmentation_losses.append(train_loss)
        valid_segmentation_losses.append(valid_loss)

        # Plot the training curves
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+2), train_segmentation_losses, label='Train Segmentation Loss')
        plt.plot(range(1, epoch+2), valid_segmentation_losses, label='Valid Segmentation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.savefig('Training_Curves.png')