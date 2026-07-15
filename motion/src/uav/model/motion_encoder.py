from types import SimpleNamespace
from typing import List

import deepspeed
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations import is_deepspeed_zero3_enabled

from src.uav.external.videoflow.core.Networks.MOFNetStack.network import MOFNet
from src.uav.external.videoflow.core.utils.utils import coords_grid

try:
    import alt_cuda_corr
    from src.uav.external.videoflow.core.Networks.MOFNetStack.corr import AlternateCorrBlock as CorrBlock
except ImportError:
    from src.uav.external.videoflow.core.Networks.MOFNetStack.corr import CorrBlock


class VideoFlowMotionEncoderConfig(PretrainedConfig):
    model_type = "videoflow_motion_encoder"
    base_config_key = "motion_config"

    def __init__(
        self,
        motion_dim=128,
        down_ratio=8,
        decoder_depth=12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.motion_dim = motion_dim
        self.down_ratio = down_ratio
        self.decoder_depth = decoder_depth


def _make_mofnet_config(decoder_depth=12):
    """Create config namespace for MOFNet with sintel defaults."""
    return SimpleNamespace(
        feat_dim=256,
        down_ratio=8,
        fnet='twins',
        cnet='twins',
        pretrain=False,
        corr_fn='default',
        Tfusion='stack',
        context_3D=False,
        corr_levels=4,
        corr_radius=4,
        decoder_depth=decoder_depth,
        mixed_precision=False,
        cost_heads_num=1,
    )


class VideoFlowMotionEncoderPreTrainedModel(PreTrainedModel):
    config_class = VideoFlowMotionEncoderConfig
    base_model_prefix = "motion_encoder"

    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        mofnet_cfg = _make_mofnet_config(decoder_depth=config.decoder_depth)
        self.mofnet = MOFNet(mofnet_cfg)
        self.mofnet.eval()

        self._decoder_depth = config.decoder_depth
        self._down_ratio = config.down_ratio
        self._hidden_dim = mofnet_cfg.feat_dim // 2  # 128

    def _init_weights(self, module):
        pass

    def load_pretrained_weights(self, ckpt_path: str):
        if is_deepspeed_zero3_enabled():
            self._load_pretrained_weights_zero3(ckpt_path)
        else:
            self._load_pretrained_weights(ckpt_path)

    def _load_pretrained_weights_zero3(self, ckpt_path: str):
        with deepspeed.zero.GatheredParameters(list(self.mofnet.parameters()), modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                self._load_pretrained_weights(ckpt_path)

    def _load_pretrained_weights(self, ckpt_path: str):
        print(f"Loading VideoFlow MOFNet weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Checkpoint saved with DataParallel: keys have "module." prefix
        mofnet_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                mofnet_state_dict[k[len("module."):]] = v

        if not mofnet_state_dict:
            raise ValueError(
                f"No keys with prefix 'module.' found in checkpoint {ckpt_path}. "
                f"Available key prefixes: {set(k.split('.')[0] for k in state_dict.keys())}"
            )

        missing_keys, unexpected_keys = self.mofnet.load_state_dict(mofnet_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys when loading MOFNet: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading MOFNet: {unexpected_keys}")
        print(f"Successfully loaded {len(mofnet_state_dict)} MOFNet parameters.")

    @torch.no_grad()
    def forward(self, video_frames: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        """
        Run full MOFNet forward pass and extract optical flow.

        Args:
            video_frames: List of [T_i, 3, H_i, W_i] tensors, float in [0, 255].
                          Each video must have T >= 3 frames.

        Returns:
            List of [T_i, 4, H/down_ratio, W/down_ratio] optical flow tensors.
            4 channels = forward_flow_x, forward_flow_y, backward_flow_x, backward_flow_y.
        """
        target_h, target_w = 320, 480  # fixed resolution for motion encoder
        out_dtype = video_frames[0].dtype  # remember caller dtype (bf16) for output cast
        results = []

        # Run entire motion encoder in float32 to avoid mixed-precision issues
        # (VideoFlow internals create float32 coords/grids that conflict with bf16 weights).
        # DeepSpeed may convert weights to bf16; ensure float32 before use.
        with torch.amp.autocast("cuda", enabled=False):
            self.mofnet.float()
            for video in video_frames:
                video = video.float()  # cast input to float32
                T, C, H, W = video.shape
                assert T >= 3, f"MOFNet requires at least 3 frames, got {T}"

                # Resize to fixed resolution if not already pre-resized by data processor
                if H != target_h or W != target_w:
                    video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)

                flow = self._extract_optical_flow(video.unsqueeze(0))

                # Pad boundary frames by replicating the nearest flow field.
                flow = torch.cat([flow[:1], flow, flow[-1:]], dim=0)
                results.append(flow.to(out_dtype))

        return results

    def _extract_optical_flow(self, images):
        """
        Run MOFNet components step by step and return bidirectional optical flow.

        Args:
            images: [1, N, 3, H, W] tensor, float in [0, 255].

        Returns:
            [N-2, 4, H/dr, W/dr] forward/backward optical-flow tensor.
        """
        mofnet = self.mofnet
        B, N, _, H, W = images.shape
        dr = self._down_ratio
        hdim = self._hidden_dim

        # Normalize to [-1, 1]
        images_norm = 2.0 * (images / 255.0) - 1.0

        # 1. Feature extraction (per-frame)
        fmaps = mofnet.fnet(images_norm.reshape(B * N, 3, H, W))
        fmaps = fmaps.reshape(B, N, -1, H // dr, W // dr)

        # 2. Build correlation volumes between adjacent frames
        middle = fmaps[:, 1:N-1].reshape(B * (N - 2), -1, H // dr, W // dr)
        forward_ref = fmaps[:, 2:N].reshape(B * (N - 2), -1, H // dr, W // dr)
        backward_ref = fmaps[:, 0:N-2].reshape(B * (N - 2), -1, H // dr, W // dr)

        forward_corr_fn = CorrBlock(
            middle, forward_ref,
            num_levels=mofnet.cfg.corr_levels,
            radius=mofnet.cfg.corr_radius,
        )
        backward_corr_fn = CorrBlock(
            middle, backward_ref,
            num_levels=mofnet.cfg.corr_levels,
            radius=mofnet.cfg.corr_radius,
        )

        # 3. Context encoder (for middle frames)
        cnet = mofnet.cnet(images_norm[:, 1:N-1].reshape(B * (N - 2), 3, H, W))
        if mofnet.cfg.context_3D:
            cnet = cnet.reshape(B, N - 2, -1, H // dr, W // dr).permute(0, 2, 1, 3, 4)
            cnet = mofnet.context_3D(cnet) + cnet
            cnet = cnet.permute(0, 2, 1, 3, 4).reshape(B * (N - 2), -1, H // dr, W // dr)

        net, inp = torch.split(cnet, [hdim, hdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        attention = mofnet.att(inp)

        # 4. Initialize flow coordinates
        bs = B * (N - 2)
        coords0 = coords_grid(bs, H // dr, W // dr).to(images.device)
        forward_coords0 = coords0.clone()
        forward_coords1 = coords0.clone()
        backward_coords0 = coords0.clone()
        backward_coords1 = coords0.clone()

        # 5. Iterative refinement (the core motion estimation loop)
        motion_hidden_state = None
        for _ in range(self._decoder_depth):
            forward_coords1 = forward_coords1.detach()
            backward_coords1 = backward_coords1.detach()

            forward_corr = forward_corr_fn(forward_coords1)
            backward_corr = backward_corr_fn(backward_coords1)

            forward_flow = forward_coords1 - forward_coords0
            backward_flow = backward_coords1 - backward_coords0

            net, motion_hidden_state, up_mask, delta_flow = mofnet.update_block(
                net, motion_hidden_state, inp,
                forward_corr, backward_corr,
                forward_flow, backward_flow,
                forward_coords0, attention, bs=B,
            )

            forward_coords1 = forward_coords1 + delta_flow[:, 0:2]
            backward_coords1 = backward_coords1 + delta_flow[:, 2:4]

        # 6. Extract final optical flow (forward + backward, 4 channels)
        final_forward_flow = forward_coords1 - forward_coords0   # [B*(N-2), 2, H/dr, W/dr]
        final_backward_flow = backward_coords1 - backward_coords0  # [B*(N-2), 2, H/dr, W/dr]
        flow = torch.cat([final_forward_flow, final_backward_flow], dim=1)  # [B*(N-2), 4, H/dr, W/dr]
        return flow.reshape(B, N - 2, 4, H // dr, W // dr).squeeze(0)

    def print_trainable_parameters(self) -> None:
        is_trainable = any(param.requires_grad for param in self.parameters())
        print(f"Motion Encoder Trainable: {is_trainable}")
