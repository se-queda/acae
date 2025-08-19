# 🧠 ACAE — Adversarial Contrastive Auto-Encoder for Time-Series Anomaly Detection

This is an **unofficial, fully modular implementation** of the paper:

> **Adversarial Contrastive Auto-Encoder for Self-Supervised Time-Series Anomaly Detection**  
> by Yujia Zheng et al., Tencent AI Lab, 2023  
> [arXiv:2306.00983](https://arxiv.org/abs/2306.00983)

---

## 🚀 Features

- TensorFlow 2.x implementation with `@tf.function` and XLA acceleration  
- encoder, decoder, discriminator, masking  
- End-to-end training on the full [SMD Dataset](https://github.com/NetManAIOps/OmniAnomaly/tree/master/OmniAnomaly/datasets/SMD)
- Batch training across all 28 machines
- AUC-ROC evaluation + model checkpointing

---

## 📦 Folder Structure
```
acae/
├── checkpoints
├── src/
│ ├── models.py
│ ├── losses.py
│ ├── masking.py
│ ├── utils.py
│ └── trainer.py
├── data/
│ └── smd/ # Place your downloaded SMD dataset here
├── checkpoints/
│ └── *.h5 # Saved model weights per machine
├── config.yaml
├── requirements.txt
├── train.py
├── .gitignore
└── README.md
```
---
## 🧪 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. Download the SMD dataset:
   Place the 28 .txt files inside data/smd/.
3. Train on a single machine: python train.py --config config.yaml
4. Train on all 28 machines (batch mode): python train.py --config config.yaml --all
---
## ⚙️ Configuration

Edit config.yaml to change:
```
latent_dim: 256
batch_size: 128
lr: 0.0001
mask_rates: [0.05, 0.15, 0.3, 0.5]
lambda_d: 1.0
lambda_e: 1.0
recon_weight: 1.0
epochs: 10
```
---
## 📄 Citation

Zheng, Yujia, et al.  
**Adversarial Contrastive Auto-Encoder for Self-Supervised Time-Series Anomaly Detection**.  
*arXiv preprint arXiv:2306.00983*, 2023.  
🔗 [View on arXiv](https://arxiv.org/abs/2306.00983)
