# coding: utf-8

"""
Warping field estimator(W) defined in the paper, which generates a warping field using the implicit
keypoint representations x_s and x_d, and employs this flow field to warp the source feature volume f_s.
"""

from torch import nn
import torch.nn.functional as F
import cv2 # for saving co-ordinates as EXR files
import numpy as np # for saving co-ordinates as EXR files
import os # for saving co-ordinates as EXR files
from .util import SameBlock2d
from .dense_motion import DenseMotionNetwork


class WarpingNetwork(nn.Module):
    def __init__(
        self,
        num_kp,
        block_expansion,
        max_features,
        num_down_blocks,
        reshape_channel,
        estimate_occlusion_map=False,
        dense_motion_params=None,
        **kwargs
    ):
        super(WarpingNetwork, self).__init__()

        self.upscale = kwargs.get('upscale', 1)
        self.flag_use_occlusion_map = kwargs.get('flag_use_occlusion_map', True)

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp,
                feature_channel=reshape_channel,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params
            )
        else:
            self.dense_motion_network = None

        self.third = SameBlock2d(max_features, block_expansion * (2 ** num_down_blocks), kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=block_expansion * (2 ** num_down_blocks), out_channels=block_expansion * (2 ** num_down_blocks), kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map
        
        # for exr file saving
        self.msc_frame_counter = 0
        self.msc_descriptor = ""
        
    def deform_input(self, inp, deformation):
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, feature_3d, kp_driving, kp_source):
        if self.dense_motion_network is not None:
            # Feature warper, Transforming feature representation according to deformation and occlusion
            dense_motion = self.dense_motion_network(
                feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source
            )
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
            else:
                occlusion_map = None

            deformation = dense_motion['deformation']  # Bx16x64x64x3
            out = self.deform_input(feature_3d, deformation)  # Bx32x16x64x64

            bs, c, d, h, w = out.shape  # Bx32x16x64x64
            out = out.view(bs, c * d, h, w)  # -> Bx512x64x64
            out = self.third(out)  # -> Bx256x64x64
            out = self.fourth(out)  # -> Bx256x64x64

            if self.flag_use_occlusion_map and (occlusion_map is not None):
                out = out * occlusion_map

        ret_dct = {
            'occlusion_map': occlusion_map,
            'deformation': deformation,
            'out': out,
        }

        # --- MSc THESIS INJECTION START ---
        import os
        import cv2
        import numpy as np

        # 1. EXPORT THE ST MAP (Flattening 3D to 2D)
        grid_numpy = deformation.detach().cpu().numpy() # Shape: (1, 16, 64, 64, 3)
        
        # Extract Batch 0, Depth Layer 8 (Center), All H, All W, Only X and Y coords
        st_2d_slice = grid_numpy[0, 8, :, :, 0:2] 
        
        # Shift PyTorch math (-1 to 1) to VFX math (0 to 1)
        st_map_data = (st_2d_slice + 1.0) / 2.0
        
        # Extract X and Y arrays
        x_coords = st_map_data[:, :, 0:1]
        y_coords = st_map_data[:, :, 1:2]
        blank_channel = np.zeros((64, 64, 1), dtype=np.float32)
        # Stack in OpenCV's expected BGR order so it saves as RGB
        bgr_out = np.concatenate((blank_channel, y_coords, x_coords), axis=-1)
        # Save the file
        st_save_path = f"output/tmp/metadata/st_map_{self.msc_descriptor}-{self.msc_frame_counter:04d}.exr"
        os.makedirs(os.path.dirname(st_save_path), exist_ok=True)
        cv2.imwrite(st_save_path, bgr_out, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])


        # 2. EXPORT THE OCCLUSION ALPHA MASK
        if occlusion_map is not None:
            occ_numpy = occlusion_map.detach().cpu().numpy() # Shape: (1, 1, 64, 64)
            occ_2d = occ_numpy[0, 0, :, :] # Strip Batch and Channel dimensions
            
            # The AI occlusion map is a multiplier from 0.0 to 1.0. 
            # Convert to standard 8-bit grayscale for easy compositing.
            occ_png = (occ_2d * 255.0).clip(0, 255).astype(np.uint8)
            
            mask_save_path = f"output/tmp/metadata/mask_{self.msc_descriptor}-{self.msc_frame_counter:04d}.png"
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
            cv2.imwrite(mask_save_path, occ_png)
        # --- MSc THESIS INJECTION END ---
        
        return ret_dct
