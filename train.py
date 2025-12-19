import os
import yaml
import argparse
from src.utils import load_smd_windows, build_tf_datasets
from src.models import build_projection_head, build_encoder, build_decoder, build_discriminator
from src.trainer import ACAETrainer
from src.evaluator import evaluate_model

def main(args):
# 1. Load Config
with open(args.config, 'r') as f:
config = yaml.safe_load(f)

# 2. Load Packed Data
# train_packed shape: (N, 5, window, features)
train_packed, test_w, test_labels, mean, scale = load_smd_windows(
data_root=config['data_root'],
machine_id=args.machine,
window=config['window_size'],
train_stride=config['train_stride']
)

# 3. Build Datasets
# train_ds/val_ds yield (B, 5, window, features)
# test_ds yields (B, window, features)
train_ds, val_ds, test_ds = build_tf_datasets(
train_packed,
test_w,
val_split=config['val_split'],
batch_size=config['batch_size']
)

# 4. Initialize Models
# Note: input_shape for models is (window_size, num_features)
input_shape = (config['window_size'], train_packed.shape[-1])

proj_head = build_projection_head(input_shape)
encoder = build_encoder(latent_dim=config['latent_dim'])
decoder = build_decoder(input_shape, latent_dim=config['latent_dim'])
discriminator = build_discriminator(latent_dim=config['latent_dim'])

# 5. Initialize Trainer
trainer = ACAETrainer(
projection_head=proj_head,
encoder=encoder,
decoder=decoder,
discriminator=discriminator,
config=config
)

# 6. Train using Physics-Augmented Views
trainer.fit(train_ds, val_ds, epochs=config['epochs'])

# 7. Evaluate on standard Test Windows
# get_reconstruction_errors expects the standard test_ds (unpacked)
scores, y_true = trainer.get_reconstruction_errors(test_ds, test_labels)

metrics = evaluate_model(scores, y_true)
print(f"\n📊 Final Results for {args.machine}:")
print(metrics)

if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--machine", type=str, default="1-1")
main(parser.parse_args())

