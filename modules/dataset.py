from utils.config import Config
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class SRDataset(Dataset):
    def __init__(self, config: Config, split='train'):
        self.config = config
        self.split = split
        
        # Set base directory based on split
        if split == 'train':
            base_dir = config.train_dir
            lr_folder = config.train_features_folder
        elif split == 'valid':
            base_dir = config.valid_dir
            lr_folder = config.valid_features_folder
        else:  # test
            base_dir = config.test_dir
            lr_folder = config.test_features_folder

        # Get image paths
        self.lr_path = os.path.join(base_dir, lr_folder)
        self.lr_images = sorted(os.listdir(self.lr_path))
        
        # Get GT paths if not test
        if split != 'test':
            self.gt_path = os.path.join(base_dir, config.gt_folder)
            self.gt_images = sorted(os.listdir(self.gt_path))

        # Basic transforms
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        # Load LR image
        lr_img = Image.open(os.path.join(self.lr_path, self.lr_images[idx])).convert('L')
        lr_tensor = self.transform(lr_img).squeeze(0)

        if self.split != 'test':
            # Load GT image
            gt_img = Image.open(os.path.join(self.gt_path, self.gt_images[idx])).convert('L')
            gt_tensor = self.transform(gt_img).squeeze(0)
            return lr_tensor, gt_tensor
        
        return lr_tensor

