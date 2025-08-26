# ACAE Model Architecture

```mermaid
flowchart TD
    subgraph Input["Input Layer"]
        I[Time Series Data<br/>64 x 38] --> Split
    end

    subgraph Split["Data Split"]
        S1[Original Sample] --> Masking
        S1 --> Anchor
        S1 --> NegGen[Negative Sample<br/>Generation]
    end

    subgraph Masking["Masking Layer"]
        M1[Mask 5%] --> Pos1
        M2[Mask 15%] --> Pos2
        M3[Mask 30%] --> Pos3
        M4[Mask 50%] --> Pos4
    end

    subgraph PosPath["Positive Path"]
        Pos1 & Pos2 & Pos3 & Pos4 --> PE[Positive Encoder]
        PE --> PL[Positive Latent<br/>256-dim]
    end

    subgraph AnchorPath["Anchor Path"]
        Anchor --> AE[Anchor Encoder]
        AE --> AL[Anchor Latent<br/>256-dim]
    end

    subgraph NegPath["Negative Path"]
        NegGen --> NE[Negative Encoder]
        NE --> NL[Negative Latent<br/>256-dim]
    end

    subgraph ContrastiveLearning["Contrastive Learning"]
        PL & AL & NL --> CL[Contrastive Loss]
    end

    subgraph Decoder["Decoder"]
        AL --> D1[Dense + Reshape]
        D1 --> D2[Upsampling Block 1]
        D2 --> D3[Upsampling Block 2]
        D3 --> O[Output<br/>64 x 38]
    end

    subgraph Discriminator["Discriminator"]
        AL --> DIS1[Dense 128]
        DIS1 --> DIS2[Dense 64]
        DIS2 --> DIS3[Output Layer]
    end

    O --> RecLoss[Reconstruction Loss]
    DIS3 --> AdvLoss[Adversarial Loss]
    CL --> TotalLoss[Total Loss]
    RecLoss --> TotalLoss
    AdvLoss --> TotalLoss

    style Input fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Split fill:#fff2cc,stroke:#333,stroke-width:2px
    style Masking fill:#d5e8d4,stroke:#333,stroke-width:2px
    style PosPath fill:#e1f3fe,stroke:#333,stroke-width:2px
    style AnchorPath fill:#e1f3fe,stroke:#333,stroke-width:2px
    style NegPath fill:#e1f3fe,stroke:#333,stroke-width:2px
    style ContrastiveLearning fill:#ffe6cc,stroke:#333,stroke-width:2px
    style Decoder fill:#e1f3fe,stroke:#333,stroke-width:2px
    style Discriminator fill:#f8cecc,stroke:#333,stroke-width:2px
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