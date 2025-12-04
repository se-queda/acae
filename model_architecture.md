flowchart TD
 subgraph subGraph0
    Input
    Projection["Projection Layer<br>(128 x 64 x 256)"]
 end
 subgraph subGraph1["ACAE Model"]
    direction LR
        Encoder(("Encoder"))
        Decoder(("Decoder"))
        Discriminator(("Discriminator"))
 end
 subgraph subGraph2
        GMV(("generate_masked_views"))
        PosLatents(("Positive Latents"))
        MF1(("mix_features"))
        LatentZ(("Z (Anchor)"))
        PosSamples(("Positive Composites"))
 end
 subgraph subGraph3
        Shuffle["tf.random.shuffle(z)"]
        MF2(("mix_features"))
        NegSamples(("Negative Composites"))
 end
 subgraph subGraph4["Loss Calculation"]
    direction LR
        DiscOut
        DL(("discriminator_loss"))
        EL(("encoder_loss"))
        ReconLoss(("Reconstruction Loss"))
        Reconstructed
        TL(("Total Loss"))
 end
 subgraph subGraph5
        subGraph2
        subGraph3
        subGraph4
 end
    Input --> Projection
    Projection --> Encoder & GMV
    Encoder -- Latent Vector z (128 x 256) --> LatentZ
    GMV -- Masked Views --> Encoder
    Encoder -- Positive Latents --> PosLatents
    LatentZ --> MF1 & Shuffle & MF2 & EL & Decoder
    
    %% Critical Update: Discriminator needs the Anchor Z to compare against composites
    LatentZ --> Discriminator
    
    PosLatents --> MF1
    MF1 --> PosSamples
    Shuffle -- Negative Latents --> MF2
    MF2 --> NegSamples
    
    PosSamples --> Discriminator
    NegSamples --> Discriminator
    
    Discriminator --> DiscOut
    DiscOut --> DL & EL
    Decoder --> Reconstructed
    Reconstructed --> ReconLoss
    ReconLoss --> TL
    DL --> TL
    EL --> TL

    style Input fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Projection fill:#d0e0e3,stroke:#333,stroke-width:2px
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
    style LatentZ fill:#ffe6cc,stroke:#333,stroke-width:2px
    style PosSamples fill:#e1f3fe,stroke:#333,stroke-width:2px
    style NegSamples fill:#f8cecc,stroke:#333,stroke-width:2px
    style DiscOut fill:#f9f9f9,stroke:#333,stroke-width:2px
    style ReconLoss fill:#d4e8d4,stroke:#333,stroke-width:2px
    style Reconstructed fill:#d4e8d4,stroke:#333,stroke-width:2px
    style TL fill:#cce6ff,stroke:#333,stroke-width:2px