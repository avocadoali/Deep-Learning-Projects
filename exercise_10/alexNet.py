import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import wandb

from torchvision.io import read_image
from torchvision.models import AlexNet, AlexNet_Weights
from exercise_code.data.segmentation_dataset import SegmentationData, label_img_to_rgb
from exercise_code.data.download_utils import download_dataset
from exercise_code.util import visualizer, save_model
from exercise_code.util.Util import checkSize, checkParams, test
from exercise_code.networks.segmentation_nn_alexnet import SegmentationNN, DummySegmentationModel
from exercise_code.tests import test_seg_nn
from tqdm import tqdm
from torch.utils.data import DataLoader


#set up default cuda device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
device = 'cuda:0'

download_url = 'https://i2dl.vc.in.tum.de/static/data/segmentation_data.zip'
i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
data_root = os.path.join(i2dl_exercises_path, 'datasets','segmentation')

download_dataset(
    url=download_url,
    data_dir=data_root,
    dataset_zip_name='segmentation_data.zip',
    force_download=False,
)

def train_model(epochs: int, train_loader: DataLoader, val_loader: DataLoader, model: torch.nn.Module, loss_func: torch.nn.Module, optimizer: torch.optim.Optimizer) -> torch.nn.Module:
    model = model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        training_loss = 0.0


        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images)
            print('')
            print(preds.shape)
            # loss = loss_func(preds, targets)
            # loss.backward()
            # optimizer.step()
            # training_loss += loss.item()

    #     torch.cuda.empty_cache()

    #    # Validation phase
    #     model.eval()
    #     validation_loss = 0.0
    #     with torch.no_grad():
    #         for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
    #             images, targets = batch
    #             images, targets = images.to(device), targets.to(device)
    #             preds = model(images)
    #             loss = loss_func(preds, targets)
    #             validation_loss += loss.item()
            

    #     # Calculate average losses
    #     train_loss = training_loss / len(train_loader)
    #     val_loss = validation_loss / len(val_loader)
    #     print(f'EPOCH [{epoch + 1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    #     # Log the metrics to wandb
    #     wandb.log({
    #         'epoch': epoch + 1,
    #         'train_loss': train_loss,
    #         'val_loss': val_loss,
    #     })

    print('done')
    return model

# Hyperparameters
hparams = {
    'epochs': 40,
    'batch_size': 15,
    'learning_rate': 5e-4,
    'padding' : 1, 
    'stride_down' : 1,
    'stride_up' : 2
}

# # Initialize wandb
# wandb.init(
#     project="my-awesome-project",
#     config=hparams
# )
# wandb.config.learning_rate = hparams['learning_rate']
# wandb.config.batch_size = hparams['batch_size']
# wandb.config.epochs= hparams['epochs']

# Initialize model, optimizer, loss function, and data loaders


# Step 1: Initialize model with the best available weights
alexNet = AlexNet(pe)

model = SegmentationNN(hp = hparams, alexNet=alexNet)
optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])

# Datasets
train_data = SegmentationData(image_paths_file=f'{data_root}/segmentation_data/train.txt')
val_data = SegmentationData(image_paths_file=f'{data_root}/segmentation_data/val.txt')
test_data = SegmentationData(image_paths_file=f'{data_root}/segmentation_data/test.txt')

train_loader = DataLoader(train_data, batch_size=hparams['batch_size'], shuffle=True)
val_loader = DataLoader(val_data, batch_size=hparams['batch_size'], shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,shuffle=False,num_workers=0)

# Loss function 
loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

# Train the model
model = train_model(hparams['epochs'], train_loader, val_loader, model, loss_func, optimizer)


os.makedirs('models', exist_ok=True)
save_model(model, "segmentation_nn.model")





