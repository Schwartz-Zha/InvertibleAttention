import torch
from PIL import Image
import os


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.join(root_dir, 'img_align_celeba')
        self.transform = transform
        self.img_files = [os.path.join(self.root_dir, f) for f
                          in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = Image.open(self.img_files[idx])
        if self.transform is not None:
            data = self.transform(images)
        return data