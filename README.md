# 🧨 Stable Diffusion Implementations

This repository contains a comprehensive implementation and exploration of **Stable Diffusion models**, covering both theoretical foundations and production-ready tools. It includes:

- A from-scratch implementation of the **DDPM** (Denoising Diffusion Probabilistic Model)
- Hands-on usage of Hugging Face 🤗 **Diffusers** and **Transformers** libraries
- Integration with the **Stable Diffusion Web UI** for fast visual experimentation

---

## 📂 Project Structure

```
├── DDPM/ddpm_from_scratch.ipynb    # Pure PyTorch implementation of DDPM
├── Deep/deep_dive2.ipynb           # AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
├── stable-diffusion-webui/         # Stable Diffusion Web UI (AUTOMATIC1111 fork or similar)
└── README.md                       # You are here
```

---

## 🌪️ 1. ddpm_from_scratch.ipynb

This notebook implements the original **Denoising Diffusion Probabilistic Model (DDPM)** from scratch in PyTorch.

### 🔍 Highlights:
- Implements the forward (noise addition) and reverse (denoising) diffusion processes
- Uses sinusoidal positional embeddings for timestep encoding
- UNet-like architecture for epsilon prediction
- Generates 28x28 or 64x64 images (e.g., MNIST, CIFAR-10) from pure noise

### 📚 References:
- [Ho et al., 2020](https://arxiv.org/abs/2006.11239): *Denoising Diffusion Probabilistic Models*
- [DDPM Illustrated](https://jalammar.github.io/illustrated-diffusion/)

---

## 🧨 2. Diffusers Models

Using the Hugging Face 🤗 `diffusers` library to explore modular Stable Diffusion components:

### ✅ Components Used:
- `AutoencoderKL`: Compresses and reconstructs images into/from latent space
- `UNet2DConditionModel`: Core denoising model, conditioned on text embeddings
- `LMSDiscreteScheduler`: Scheduler for the noise prediction steps

### 🔧 What This Covers:
- Loading pre-trained models (e.g., `CompVis/stable-diffusion-v1-4`)
- Conditioning generation with text prompts
- Latent space denoising pipeline
- Latent-to-image decoding using VAE

### 🧪 Sample Usage:
```python
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
```

---

## 🧠 3. Transformers: CLIPTextModel, CLIPTokenizer, logging

Used Hugging Face 🤗 `transformers` library to tokenize and embed text prompts for conditioning.

### ✨ Components:
- `CLIPTokenizer`: Converts text prompts to token IDs
- `CLIPTextModel`: Converts tokens into embeddings
- `logging`: Used for tracking inference and debugging

### 🔧 Example:
```python
from transformers import CLIPTextModel, CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

inputs = tokenizer("A futuristic city at sunset", return_tensors="pt")
embeddings = text_encoder(**inputs).last_hidden_state
```

---

## 🖼️ 4. Stable Diffusion Web UI (AUTOMATIC1111)

Integrated with the popular Web UI for interactive visual experimentation and batch image generation.

### 💻 Features:
- Inpainting, outpainting, and text-to-image
- Batch generation and prompt matrix
- Extensions for controlnet, LoRA, etc.

### 🚀 Setup (Basic):
```bash
git clone https://github.com/SamAdebisi/STABLE_DIFFUSION.git
python Stable_Diffusion_WebUI_A1111.ipynb
```

### 🌐 Access UI:
Visit `http://localhost:7860` in your browser.

---

## 📊 Comparison Overview

| Component               | Library       | Role                                      |
|------------------------|---------------|-------------------------------------------|
| `DDPM_from_scratch`    | PyTorch       | Theory and core diffusion implementation  |
| `AutoencoderKL`        | 🤗 Diffusers   | Latent image compression & reconstruction |
| `UNet2DConditionModel` | 🤗 Diffusers   | Noise prediction and conditioning         |
| `CLIPTextModel`        | 🤗 Transformers| Text embedding for guidance               |
| `stable-diffusion-webui`| UI Interface | Fast prototyping and visual testing       |

---

## 🧠 Learning Objectives

By using this repository, you’ll:
- Understand the foundations of DDPMs
- Learn how latent diffusion differs from pixel-space diffusion
- Use Hugging Face APIs to build text-to-image generation pipelines
- Interface with production-level Stable Diffusion Web UIs

---

## 📚 References

- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [CLIP by OpenAI](https://github.com/openai/CLIP)
- [AUTOMATIC1111 Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

## 🛠️ Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

MIT License. See `LICENSE` file for details.
