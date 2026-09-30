from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
class CelebADataset(Dataset):

    def __init__(self,root,transform=None):

        self.root = root
        self.paths = sorted(Path(root).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    
    def __getitem__(self,idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


