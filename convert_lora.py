import torch
from safetensors.torch import save_file

ckpt = torch.load("lora_sdxl.pth", map_location="cpu")

lora_state = {}

for k, v in ckpt.items():

    if "lora_A" in k or "lora_B" in k:
        new_key = k


        new_key = new_key.replace(".original", "")


        new_key = "lora_unet_" + new_key.replace(".", "_")


        new_key = new_key.replace("_lora_A_weight", ".lora_down.weight")
        new_key = new_key.replace("_lora_B_weight", ".lora_up.weight")

        lora_state[new_key] = v

save_file(lora_state, "lora_diffusers.safetensors")

print("Converted to Diffusers LoRA")