# ACAE Model Architecture

```mermaid
flowchart TD
    subgraph Input["Input Layer"]
        I[Time Series Data<br/>64 x 38] --> E
    end

    subgraph E["Encoder"]
        E1[Conv1D Block 1] --> E2[Conv1D Block 2]
        E2 --> E3[Global Pooling]
        E3 --> L[Latent Space<br/>256-dim]
    end

    subgraph D["Decoder"]
        L --> D1[Dense + Reshape]
        D1 --> D2[Upsampling Block 1]
        D2 --> D3[Upsampling Block 2]
        D3 --> O[Output<br/>64 x 38]
    end

    subgraph DIS["Discriminator"]
        L --> DIS1[Dense 128]
        DIS1 --> DIS2[Dense 64]
        DIS2 --> DIS3[Output Layer]
    end

    subgraph Masking["Masking Layer"]
        M1[Mask 5%]
        M2[Mask 15%]
        M3[Mask 30%]
        M4[Mask 50%]
    end

    I --> Masking
    Masking --> E
    L --> D
    O --> Loss
    DIS3 --> Loss

    style Input fill:#f9f9f9,stroke:#333,stroke-width:2px
    style E fill:#e1f3fe,stroke:#333,stroke-width:2px
    style D fill:#e1f3fe,stroke:#333,stroke-width:2px
    style DIS fill:#f8cecc,stroke:#333,stroke-width:2px
    style Masking fill:#d5e8d4,stroke:#333,stroke-width:2px
```

## Model Components

### Input Layer
- Time series data with 64 timesteps and 38 features
- Input shape: (64, 38)

### Masking Layer
- Applies different masking rates for contrastive learning:
  - 5% masking
  - 15% masking
  - 30% masking
  - 50% masking

### Encoder
- Conv1D blocks with residual connections
- Global pooling layer
- Projects to 256-dimensional latent space

### Decoder
- Dense layer with reshape operation
- Upsampling blocks
- Reconstructs original input dimensions (64, 38)

### Discriminator
- Dense layers (128 → 64)
- Output layer for adversarial training
