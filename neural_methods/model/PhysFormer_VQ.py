"""PhysFormer with Vector Quantization applied to encoder features."""

import torch
from torch import nn
from torch.nn import functional as F

from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp, as_tuple

class VectorQuantizer(nn.Module):
    """Basic vector quantization module."""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs: torch.Tensor):
        """Quantize `inputs` which should be shape (B, C, T, H, W)."""
        b, c, t, h, w = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)
        quantized = torch.matmul(encodings, self.embedding.weight).view(b, t, h, w, c)
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, loss, perplexity


class PhysFormerEncoder(ViT_ST_ST_Compact3_TDC_gra_sharp):
    """Encoder from PhysFormer returning feature map before the final conv."""
    def forward(self, x: torch.Tensor, gra_sharp: float):
        b, c, t, fh, fw = x.shape
        x = self.Stem0(x)
        x = self.Stem1(x)
        x = self.Stem2(x)
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x, _ = self.transformer1(x, gra_sharp)
        x, _ = self.transformer2(x, gra_sharp)
        x, _ = self.transformer3(x, gra_sharp)
        x = x.transpose(1, 2).view(b, self.dim, t // 4, 4, 4)
        x = self.upsample(x)
        x = self.upsample2(x)
        return x


class PhysFormerVQ(nn.Module):
    """PhysFormer model with vector quantization on encoder features."""
    def __init__(self, *, num_embeddings: int = 512, commitment_cost: float = 0.25, **kwargs):
        super().__init__()
        self.encoder = PhysFormerEncoder(**kwargs)
        self.dim = self.encoder.dim // 2
        self.vq = VectorQuantizer(num_embeddings, self.dim, commitment_cost)
        self.conv_last = nn.Conv1d(self.dim, 1, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, gra_sharp: float):
        feats = self.encoder(x, gra_sharp)
        quantized, vq_loss, perplexity = self.vq(feats)
        out = torch.mean(quantized, 3)
        out = torch.mean(out, 3)
        out = self.conv_last(out)
        rppg = out.squeeze(1)
        return rppg, vq_loss, perplexity
