import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

from lora import inject_lora, get_lora_params
from dataset import SDXLDataset
from utils import encode_prompt, get_time_ids

# ---------------- CONFIG ----------------
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_DIR = "./images"

BATCH_SIZE = 1
LR = 5e-6
EPOCHS = 2
RANK = 8

DEVICE = "cuda"



pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
).to(DEVICE)

unet = pipe.unet
vae = pipe.vae

vae.to(DEVICE, dtype=torch.float32)
text_encoder1 = pipe.text_encoder
text_encoder2 = pipe.text_encoder_2
tokenizer1 = pipe.tokenizer
tokenizer2 = pipe.tokenizer_2

vae.eval() 

inject_lora(unet, rank=RANK)
unet.to(DEVICE, dtype=torch.bfloat16)


unet.enable_gradient_checkpointing()


for p in unet.parameters():
    p.requires_grad = False

for m in unet.modules():
    if hasattr(m, "lora_A"):
        for p in m.parameters():
            p.requires_grad = True


optimizer = torch.optim.AdamW(get_lora_params(unet), lr=LR, eps=1e-6)


noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")


dataset = SDXLDataset(IMAGE_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):


        pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float32)


        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()

        latents = latents * 0.13025
        latents = latents.to(dtype=torch.bfloat16)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=DEVICE
        )

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


        embeddings, pooled_embeddings = encode_prompt(
            batch,
            tokenizer1,
            tokenizer2,
            text_encoder1,
            text_encoder2,
            DEVICE
        )


        time_ids = get_time_ids(batch).to(DEVICE)


        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=embeddings,
            added_cond_kwargs={
                "time_ids": time_ids,
                "text_embeds": pooled_embeddings
            }
        ).sample


        loss = F.mse_loss(noise_pred.float(), noise.float())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(get_lora_params(unet), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss: {loss.item()}")


torch.save(unet.state_dict(), "lora_sdxl.pth")