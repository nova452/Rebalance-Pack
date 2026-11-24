"""
Rebalance-Pack - merged ComfyUI custom node package.

    foundational.py         -> quickwork utility nodes
    utilities.py            -> LoadImagesBlaze, ListDisplay
    model_toolkit.py        -> LoraLoaderBlock
    resize_toolkit.py       -> ImageResolutionCap, ImageAspectRatioCrop,
                               MaskAspectRatioBBox
    mask_toolkit.py         -> UncropImage, UncropMask, BorderMaskDetector, Mask Opacity
    conditioning_rebalance.py -> RebalanceGuider, RebalanceGuiderB, StepRebalance,
                               RebalanceCFG, ConditioningMerge, ConditioningMergeMulti
    sigmas_rebalance.py     -> SigmasRebalance, SamplerRebalance
    krea2.py                -> ConditioningKrea2Rebalance, Krea2EditRebalance,
                               Krea2EncodeRebalance
    ideogram4.py            -> ConditioningIdeogram4Rebalance,
                               Ideogram4EditRebalance, Ideogram4EncodeRebalance
    krea2_block_lora.py     -> Krea2LoraLoaderBlock
    moss_image_block_lora.py -> MOSSImageLoraLoaderBlock
    minimax_h3_block_lora.py -> MiniMaxH3LoraLoaderBlock,
                               MiniMaxH3ImageLoraLoaderBlock
    lora_folder_bulk.py     -> LoraFolderBulkBlocks, LoraFolderBulkBlocksMossImage,
                               LoraFolderBulkBlocksH3, LoraFolderBulkBlocksH3Image

The Omni Node family moved to its own pack, ComfyUI-Omni-Node.
"""

from . import foundational
from . import utilities
from . import model_toolkit
from . import resize_toolkit
from . import mask_toolkit
from . import conditioning_rebalance
from . import sigmas_rebalance
from . import krea2
from . import ideogram4
from . import krea2_block_lora
from . import moss_image_block_lora
from . import minimax_h3_block_lora
from . import lora_folder_bulk

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _module in (foundational, utilities, resize_toolkit, mask_toolkit,
                model_toolkit, conditioning_rebalance, sigmas_rebalance, krea2, ideogram4,
                krea2_block_lora, moss_image_block_lora, minimax_h3_block_lora,
                lora_folder_bulk):
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"
