import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        
        # Upsampling network
        self.upsampling = nn.Sequential(
            # First upscale (x2)
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            
            # Second upscale (x4)
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            
            # Third upscale (x8)
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            
            # Final conv to get single channel
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        # Extract features
        feat = self.features(x)
        # Upscale
        out = self.upsampling(feat)
        return out

