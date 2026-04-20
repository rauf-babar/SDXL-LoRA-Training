import torch

ckpt = torch.load("lora_sdxl.pth", map_location="cpu")

print(type(ckpt))

if isinstance(ckpt, dict):
    print("Top-level keys:")
    for k in list(ckpt.keys())[:30]:
        print(k)