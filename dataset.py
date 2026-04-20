import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SDXLDataset(Dataset):
    def __init__(self, image_dir, size=1024):
        self.image_dir = image_dir
        self.size = size

        self.pairs = []
        valid_exts = [".png", ".jpg", ".jpeg"]

        for file in os.listdir(image_dir):
            name, ext = os.path.splitext(file)

            if ext.lower() in valid_exts:
                txt_path = os.path.join(image_dir, name + ".txt")
                img_path = os.path.join(image_dir, file)

                if os.path.exists(txt_path):
                    self.pairs.append((img_path, txt_path))
                else:
                    print(f"[WARNING] Missing caption for {file}, skipping.")

        if len(self.pairs) == 0:
            raise ValueError("No valid image-text pairs found!")

        print(f"Loaded {len(self.pairs)} image-caption pairs.")

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, txt_path = self.pairs[idx]


        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)


        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        return {
            "pixel_values": image,
            "caption": caption,
            "original_size": (self.size, self.size),
            "crop_coords": (0, 0),
            "target_size": (self.size, self.size)
        }
        