# SDXL LoRA Training Pipeline

A lightweight, custom-built pipeline for training Low-Rank Adaptations (LoRA) on the Stable Diffusion XL (SDXL) base model using PyTorch and Hugging Face Diffusers. 

This repository provides an end-to-end workflow, from injecting LoRA layers into the UNet to training with custom image-caption pairs and converting the final weights into a standard Diffusers-compatible `.safetensors` format.

---

## Main Project: Furniture-AI
This training pipeline is the underlying engine utilized for **[Furnish-AI](https://github.com/rauf-babar/Furnish-AI)**. 

Furniture-AI is a dedicated product built to generate high-quality, realistic furniture settings and interior design concepts using customized SDXL models. If you are interested in seeing how these LoRA models are deployed in a real-world application, please check out the main repository!

---

## Repository Overview

Here is a breakdown of the core files in this repository and their functionalities:

* **`train.py`**: The primary entry point for training. It loads the SDXL pipeline, injects the custom LoRA layers into the UNet, freezes non-LoRA parameters, and runs the training loop over your dataset.
* **`dataset.py`**: Contains the `SDXLDataset` class. It loads image files (`.png`, `.jpg`, `.jpeg`) and their corresponding `.txt` captions from a specified directory, applying resizing (default 1024x1024) and normalization.
* **`lora.py`**: The core LoRA implementation. It defines the `LoRALinear` module and includes utility functions to recursively inject these layers into the UNet and extract trainable parameters for the optimizer.
* **`utils.py`**: Handles SDXL-specific conditioning. It includes `encode_prompt` to process text through SDXL's dual text encoders, and `get_time_ids` to format image size and crop coordinates required by the model.
* **`convert_lora.py`**: A post-training utility script. It takes the custom `lora_sdxl.pth` state dictionary and converts the keys into the standard Hugging Face Diffusers `.safetensors` format so the LoRA can be easily loaded in standard inference pipelines.
* **`check.py`**: A simple debugging script to load and inspect the top-level keys of the saved `.pth` checkpoint.
* **`requirements.txt`**: Lists all the necessary Python dependencies required to run the pipeline (e.g., `torch`, `diffusers`, `transformers`, `accelerate`).

---

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/rauf-babar/SDXL-LoRA-Training.git](https://github.com/rauf-babar/SDXL-LoRA-Training.git)
cd SDXL-LoRA-Training
pip install -r requirements.txt

```

### 2. Dataset Preparation

Create an `images` folder in the root directory. Place your training images alongside their corresponding text files containing the captions. They must share the same filename.

```text
images/
  ├── chair_01.jpg
  ├── chair_01.txt
  ├── table_02.png
  └── table_02.txt

```

### 3. Training

Adjust the configuration parameters (like `BATCH_SIZE`, `LR`, `EPOCHS`, and `RANK`) directly inside `train.py` if needed. Then, start the training process:

```bash
python train.py

```

This will output a `lora_sdxl.pth` file containing your trained weights.

### 4. Conversion for Inference

To use your newly trained LoRA in standard UI tools (like ComfyUI, Automatic1111) or a standard Diffusers pipeline, convert the custom `.pth` file to `.safetensors`:

```bash
python convert_lora.py

```

This generates `lora_diffusers.safetensors`, which is ready for deployment.

