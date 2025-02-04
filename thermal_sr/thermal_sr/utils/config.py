from dataclasses import dataclass
import os

@dataclass
class Config:
    dataset_base: str
    gt_folder: str = "GT"
    train_features_folder: str = "LR_x8"
    valid_features_folder: str = "LR_x8" 
    test_features_folder: str = "sisr_x8"

    def __post_init__(self):
        # Set main directories
        self.train_dir = os.path.join(self.dataset_base, "train")
        self.test_dir = os.path.join(self.dataset_base, "test")
        self.valid_dir = os.path.join(self.dataset_base, "val")

        # Check directories exist
        self.train_gt_dir = os.path.join(self.train_dir, self.gt_folder)
        self.train_lr_dir = os.path.join(self.train_dir, self.train_features_folder)
        self.valid_gt_dir = os.path.join(self.valid_dir, self.gt_folder)
        self.valid_lr_dir = os.path.join(self.valid_dir, self.valid_features_folder)
        self.test_lr_dir = os.path.join(self.test_dir, self.test_features_folder)

        paths = [
            self.train_gt_dir, 
            self.train_lr_dir, 
            self.valid_gt_dir, 
            self.valid_lr_dir, 
            self.test_lr_dir
        ]
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Not found: {p}")



