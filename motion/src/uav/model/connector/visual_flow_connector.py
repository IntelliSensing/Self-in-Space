"""
VisualFlowConnector: Fuse ViT-encoded optical flow features with visual embeddings.

Pipeline:
    MOFNet → flow → 3ch pseudo-image → Qwen ViT (frozen) → flow_embeds [N, lang_dim]
    → LayerNorm → MLP → direct add with visual_embeds

Follows the same pattern as Spatial-MLLM's MLPAddConnector:
    Norm → MLP projection → additive fusion.
No attention ops — pure LayerNorm + Linear, fully bf16-safe.
"""

import torch
import torch.nn as nn


class VisualFlowConnector(nn.Module):
    """Fuse ViT-encoded flow features with visual embeddings.

    Architecture (following Spatial-MLLM's MLPAddConnector):
        1. LayerNorm on flow_embeds
        2. MLP: Linear → GELU → Linear (lang_dim → lang_dim)
        3. Additive fusion: visual_embeds + projected_flow
    """

    def __init__(self, lang_dim: int, **kwargs):
        super().__init__()
        self.lang_dim = lang_dim
        self.ln = nn.LayerNorm(lang_dim)
        self.mlp = nn.Sequential(
            nn.Linear(lang_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, lang_dim),
        )

    def forward(
        self,
        visual_embeds: torch.Tensor,
        flow_embeds: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_embeds: [N, lang_dim] visual embeddings from ViT.
            flow_embeds:   [N, lang_dim] flow embeddings from ViT (frozen).
            grid_thw:      [num_videos, 3] grid layout (unused, kept for API compat).
        Returns:
            [N, lang_dim] fused embeddings.
        """
        flow_embeds = self.ln(flow_embeds)
        flow_embeds = self.mlp(flow_embeds)
        return visual_embeds + flow_embeds

    def print_trainable_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        is_trainable = any(p.requires_grad for p in self.parameters())
        print(f"VisualFlowConnector Trainable: {is_trainable} ({trainable:,}/{total:,} params)")
