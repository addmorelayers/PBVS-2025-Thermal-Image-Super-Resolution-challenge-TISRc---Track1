import pdb
import torch
import torch.nn as nn
from thermal_sr.utils.config import Config
from thermal_sr.modules.dataset import SRDataset
from thermal_sr.modules.simple_model import SimpleModel

config = Config("data")
    
# Create datasets
train_dataset = SRDataset(config, split='train')
valid_dataset = SRDataset(config, split='valid')
# test_dataset = SRDataset(config, split='test')

# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=8,
    shuffle=True,
    num_workers=4
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=4
)


# Example usage
for lr, hr in train_loader:
    print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")
    break

 

# Training loop
def train_step(model, optimizer, lr, hr, epoch, batch_idx):
   optimizer.zero_grad()
   print(f"\nProcessing batch {batch_idx} in epoch {epoch}")
   
   sr = model(lr)
   loss = nn.L1Loss()(sr, hr)
   
   loss.backward()
   optimizer.step()
   
   print(f"Batch {batch_idx} Loss: {loss.item():.4f}")
   return loss.item()

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("\nModel created with Adam optimizer, lr=1e-4")

# Dummy data for example
lr = torch.randn(8, 1, 56, 80)
hr = torch.randn(8, 1, 448, 640)

print("\nStarting training...")
loss = train_step(model, optimizer, lr, hr, epoch=1, batch_idx=0)








