import sys 
sys.path.append('..')
sys.path.append('../src')
sys.path.append('/Users/tom/Development/carbot_ai_model/src')
sys.path.append('/Users/tom/Development/carbot_ai_model/Depth-Anything-V2')

from waypoint_dataset import WaypointDataset
from model_context import ModelContextV2
from model_v2 import ModelV2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from train import train_model, view_evaluation_sample, view_val_train_loss


## Setup optimizer and loss function
train_iters = 25
learning_rate = 1e-3
warmup_steps = 2000
batch_size = 16
train_split = 0.90

data_folder = 'outside_office_v2'

model_context = ModelContextV2(waypoint_num=6, num_prev_observations=0, levels=[16, 32, 64], height=280, width=280, embedding_dim=128, hidden_dim=256, attention_blocks=2) 

transform = transforms.Compose([
    transforms.Resize((model_context.height, model_context.width)),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
    transforms.Normalize(mean=model_context.mean, std=model_context.std)])

## 5fps so a frame every second as context.
frame_skips = 5
waypoint_skips = 2
dataset = WaypointDataset(data_folder=data_folder, frame_skips = frame_skips, waypoint_skips=waypoint_skips, model_context=model_context, transform=transform)


train_idx, val_idx = train_test_split(list(range(len(dataset))), train_size=train_split, random_state=42)

  # Create data loaders
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

train_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=train_sampler, num_workers=0)
val_loader = DataLoader(dataset, batch_size=batch_size, 
                        sampler=val_sampler, num_workers=0)

model = ModelV2(model_context=model_context, kernel_size=3).to(model_context.device)



def warmup_lr(step):
    return min(step, warmup_steps) / warmup_steps

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
criterion = nn.MSELoss()

dummy_input = torch.randn(2, model_context.num_prev_observations + 1, model_context.channels, model_context.height, model_context.width).to(model_context.device)
with torch.no_grad():
    dummy_output = model(dummy_input)
print("Dummy output shape:", dummy_output.shape)


avg_losses, avg_val_losses = train_model(model = model,
                                         train_loader = train_loader,
                                         val_loader = val_loader,
                                         criterion = criterion,
                                         optimizer = optimizer,
                                         scheduler = scheduler,
                                         train_iters = train_iters,
                                         save_model_path = data_folder,
                                         model_context = model_context, 
                                         custom_loss = True)

view_val_train_loss(avg_losses, avg_val_losses)
