# ACAE Model Architecture

```mermaid
flowchart TD
 subgraph subGraph0["Data Pipeline"]
        Input["Time Series Batch (128 x 64 x 38)"]
  end
 subgraph subGraph1["ACAE Model"]
    direction LR
        Encoder(("Encoder"))
        Decoder(("Decoder"))
        Discriminator(("Discriminator"))
  end
 subgraph subGraph2["Positive Samples"]
        GMV(("generate_masked_views"))
        PosLatents(("Positive Latents"))
        MF1(("mix_features"))
        LatentZ(("Z"))
        PosSamples(("Positive Samples"))
  end
 subgraph subGraph3["Negative Samples"]
        Shuffle["tf.random.shuffle(z)"]
        MF2(("mix_features"))
        NegSamples(("Negative Samples"))
  end
 subgraph subGraph4["Loss Calculation"]
    direction LR
        DiscOut["Discriminator Output<br>(label, proportion)"]
        DL(("discriminator_loss"))
        EL(("encoder_loss"))
        ReconLoss(("Reconstruction Loss"))
        Reconstructed["Reconstructed Batch<br>(128 x 64 x 38)"]
        TL(("Total Loss"))
  end
 subgraph subGraph5["Training Step"]
        subGraph2
        subGraph3
        subGraph4
  end
    Input --> Encoder & GMV
    Encoder -- Latent Vector z (128 x 256) --> LatentZ
    GMV -- Masked Views --> Encoder
    Encoder -- Positive Latents --> PosLatents
    LatentZ --> MF1 & Shuffle & MF2 & EL & Decoder
    PosLatents --> MF1
    MF1 --> PosSamples
    Shuffle --> MF2
    MF2 --> NegSamples
    PosSamples --> Discriminator & Discriminator
    NegSamples --> Discriminator & Discriminator
    Discriminator --> DiscOut
    DiscOut --> DL & EL
    Decoder --> Reconstructed
    Reconstructed --> ReconLoss
    ReconLoss --> TL
    DL --> TL
    EL --> TL

    style Input fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Encoder fill:#fff2cc,stroke:#333,stroke-width:2px
    style Decoder fill:#fff2cc,stroke:#333,stroke-width:2px
    style Discriminator fill:#fff2cc,stroke:#333,stroke-width:2px
    style LatentZ fill:#ffe6cc,stroke:#333,stroke-width:2px
    style PosSamples fill:#e1f3fe,stroke:#333,stroke-width:2px
    style NegSamples fill:#f8cecc,stroke:#333,stroke-width:2px
    style DiscOut fill:#f9f9f9,stroke:#333,stroke-width:2px
    style ReconLoss fill:#d4e8d4,stroke:#333,stroke-width:2px
    style Reconstructed fill:#d4e8d4,stroke:#333,stroke-width:2px
    style TL fill:#cce6ff,stroke:#333,stroke-width:2px



```

## Model Components

### Input Layer
- Time series data with 64 timesteps and 38 features
- Input shape: (64, 38)

### Data Split
- Original sample used as anchor
- Generates positive samples through masking
- Creates negative samples through augmentation/shuffling

### Masking Layer (Positive Sample Generation)
- Applies different masking rates for contrastive learning:
  - 5% masking
  - 15% masking
  - 30% masking
  - 50% masking

### Encoders
- Three parallel encoding paths:
  1. Anchor path: Processes original unmasked input
  2. Positive path: Processes masked versions of input
  3. Negative path: Processes negative samples
- Each encoder consists of:
  - Conv1D blocks with residual connections
  - Global pooling layer
  - Projects to 256-dimensional latent space

### Contrastive Learning
- Computes similarities between anchor-positive and anchor-negative pairs
- Maximizes agreement between anchor and positive samples
- Minimizes agreement between anchor and negative samples

### Decoder
- Dense layer with reshape operation
- Upsampling blocks
- Reconstructs original input dimensions (64, 38)
- Trained with reconstruction loss

### Discriminator
- Dense layers (128 → 64)
- Output layer for adversarial training
- Provides adversarial loss signal

### Loss Components
- Contrastive Loss: From triplet comparisons
- Reconstruction Loss: From decoder output
- Adversarial Loss: From discriminator
- Combined into total loss for training
