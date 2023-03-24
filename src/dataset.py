from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from .transforms import Compose

__all__ = ['Dataset']

	
class Dataset(torch.utils.data.Dataset):

    def __init__(self, name: str, path: str, transforms: list=None, debug: bool=False):
        super().__init__()

        self.name = name
        self.text_file = path
        self.files = self._filenames()
        
        transforms = transforms or [lambda x: x]
        self.transforms = Compose([*transforms])

        if debug:
            self.files = self.files[:10]

    def __len__(self):
        return len(self.files)
        
    def _transform(self, images):
        images = [transforms.ToTensor()(img) for img in images]
        return self.transforms(*images)        

    def _filenames(self):
        with open(self.text_file, 'r') as text_file:
            files = [f.strip().split(';') for f in text_file.readlines()]
        return files
    
    def __getitem__(self, idx):
        images = [Image.open(f) for f in self.files[idx]]
        return self._transform(images)