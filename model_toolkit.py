"""Rebalance-Pack model toolkit.

"""

import math
import re

import torch
import torch.nn.functional as F

import folder_paths
import comfy.utils
import comfy.lora
import comfy.lora_convert
import comfy.sd

from .conditioning_rebalance import merge_conditioning_mode


# SDXL block groups, in the same order as ModelMergeSDXL.
# Each entry is (display_name, key_prefix) where key_prefix is matched against
# the LoRA key map (after stripping the leading "diffusion_model.").
SDXL_BLOCKS = [
    ("base",           None),                 # time_embed. + label_emb. + anything not matched
    ("input_blocks.0", "input_blocks.0"),
    ("input_blocks.1", "input_blocks.1"),
    ("input_blocks.2", "input_blocks.2"),
    ("input_blocks.3", "input_blocks.3"),
    ("input_blocks.4", "input_blocks.4"),
    ("input_blocks.5", "input_blocks.5"),
    ("input_blocks.6", "input_blocks.6"),
    ("input_blocks.7", "input_blocks.7"),
    ("input_blocks.8", "input_blocks.8"),
    ("middle_block.0", "middle_block.0"),
    ("middle_block.1", "middle_block.1"),
    ("middle_block.2", "middle_block.2"),
    ("output_blocks.0", "output_blocks.0"),
    ("output_blocks.1", "output_blocks.1"),
    ("output_blocks.2", "output_blocks.2"),
    ("output_blocks.3", "output_blocks.3"),
    ("output_blocks.4", "output_blocks.4"),
    ("output_blocks.5", "output_blocks.5"),
    ("output_blocks.6", "output_blocks.6"),
    ("output_blocks.7", "output_blocks.7"),
    ("output_blocks.8", "output_blocks.8"),
    ("out",            "out."),
    ("out",            None),                 # 24th slot: catch-all for any remaining keys
]

# Default vector: full strength on every block. Spaces split the vector into
# the SDXL segments: base | input_blocks.0-8 | middle_block.0-2 |
# output_blocks.0-8 | out | catch-all  (1 + 9 + 3 + 9 + 1 + 1 = 24).
DEFAULT_BLOCK_VECTOR = "1, 1,1,1,1,1,1,1,1,1, 1,1,1, 1,1,1,1,1,1,1,1,1, 1, 1"


def _is_numeric(s):
    return re.match(r"^-?\d+(\.\d+)?$", s) is not None


def parse_block_vector(block_vector):
    if block_vector is None:
        block_vector = ""
    parts = [p.strip() for p in block_vector.split(",") if p.strip() != ""]

    if len(parts) != len(SDXL_BLOCKS):
        raise ValueError(
            "[LoraLoaderBlock] block_vector must contain exactly {} entries "
            "(base, input_blocks.0-8, middle_block.0-2, output_blocks.0-8, "
            "out, catch-all), got {}. "
            "Use spaces to split segments: e.g. '1, 1,1,1,1,1,1,1,1,1, 1,1,1, "
            "1,1,1,1,1,1,1,1,1, 1, 1'.".format(len(SDXL_BLOCKS), len(parts))
        )

    ratios = []
    for p in parts:
        if not _is_numeric(p):
            raise ValueError(
                "[LoraLoaderBlock] block_vector entry '{}' is not a number.".format(p)
            )
        ratios.append(float(p))

    return ratios


class LoraLoaderBlock:
    """Loads a LoRA and applies it to the MODEL with per-block weights."""

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        lora_names = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (["SDXL"], {"default": "SDXL"}),
                "lora_name": (lora_names,),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "block_vector": ("STRING", {
                    "multiline": False,
                    "placeholder": "24 comma-separated ratios (use spaces to split segments): base, input_blocks.0-8, middle_block.0-2, output_blocks.0-8, out, catch-all",
                    "default": DEFAULT_BLOCK_VECTOR,
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "doit"
    CATEGORY = "loaders"

    DESCRIPTION = (
        "Loads a LoRA and applies it to the MODEL with per-block weights. "
    )

    @staticmethod
    def _load_lora(lora_name, cached):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if cached is not None and cached[0] == lora_path:
            lora = cached[1]
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        return lora_path, lora

    @staticmethod
    def _assign_block_index(k_unet):
        for idx, (name, prefix) in enumerate(SDXL_BLOCKS):
            if prefix is None:
                continue
            if k_unet.startswith(prefix):
                return idx
        # time_embed. / label_emb. and anything else -> base (index 0)
        return 0

    def doit(self, model, model_type, lora_name, strength, block_vector):
        if strength == 0:
            return (model,)

        lora_path, lora = self._load_lora(lora_name, self.loaded_lora)
        self.loaded_lora = (lora_path, lora)

        key_map = comfy.lora.model_lora_keys_unet(model.model)
        loaded = comfy.lora.load_lora(lora, key_map)

        ratios = parse_block_vector(block_vector)

        new_modelpatcher = model.clone()

        for k, v in loaded.items():
            # `k` is the full model key (e.g. "diffusion_model.input_blocks.0...").
            if isinstance(k, tuple):
                k_str = k[0]
            else:
                k_str = k

            # Strip the diffusion_model. prefix to inspect the unet sub-key.
            if k_str.startswith("diffusion_model."):
                k_unet = k_str[len("diffusion_model."):]
            else:
                k_unet = k_str

            idx = self._assign_block_index(k_unet)
            ratio = ratios[idx]

            if ratio == 0:
                continue

            new_modelpatcher.add_patches({k: v}, strength * ratio)

        return (new_modelpatcher,)


NODE_CLASS_MAPPINGS = {
    "LoraLoaderBlock": LoraLoaderBlock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoaderBlock": "Load LoRA (Block)",
}

__all__ = [
    "LoraLoaderBlock",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

