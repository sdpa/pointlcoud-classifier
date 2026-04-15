import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes=40, embed_dim=128, num_heads=4, num_layers=3):
        super(TransformerClassifier, self).__init__()
        
        # 1. Point Embedding: x,y,z mapped to a higher dimension
        self.input_embed = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        # 2. Transformer Encoder Blocks (Self-Attention)
        # Using PyTorch's native TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=256, 
            dropout=0.1, 
            activation='relu',
            batch_first=True # Input is (B, N, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x is (B, 3, N)
        # Embed points
        feat = self.input_embed(x) # (B, embed_dim, N)
        
        # Prepare for transformer (B, N, embed_dim)
        feat = feat.permute(0, 2, 1) 
        
        # Since transformer blocks are permutation equivariant on the set elements (if we ignore positional encodings or if PEs are the features themselves),
        # we can just pass the features into the self attention blocks.
        # Note: We rely on the initial feature extraction to encode the local interactions, 
        # but pure self-attention here will model global interactions between all point embeddings.
        attn_out = self.transformer_encoder(feat) # (B, N, embed_dim)
        
        # Global Pooling (Max)
        # Permute back to (B, embed_dim, N) for max pooling over N points
        attn_out = attn_out.permute(0, 2, 1)
        global_feat = torch.max(attn_out, dim=2, keepdim=False)[0] # (B, embed_dim)
        
        # Classify
        logits = self.classifier(global_feat) # (B, num_classes)
        return F.log_softmax(logits, dim=1)
