"""Bulk LoRA folder loaders with per-block strength adjustment.

"Load LoRA Folder (Block)" applies every LoRA inside a folder of the loras
directory onto a Krea 2 model, scaled by one 0.0-1.0 widget per block group
(same layout as ModelMergeKrea2). "Load LoRA Folder Moss Image (Block)" is the
MOSS-Image counterpart, keyed on the native module layout of
comfy.ldm.moss_image.model.MOSSImage. "Load LoRA Folder H3 (Block)" and
"Load LoRA Folder H3 Image (Block)" do the same for MiniMax H3 and H3 Image,
keyed on comfy.ldm.minimax.model.MiniMaxH3Model (H3 Image drops the audio
stream modules). Every widget starts at 0.0, so nothing is
fired until blocks are turned up - useful when stacking tens of LoRAs whose
deltas otherwise blow out the later blocks.

The `method` dropdown picks how the folder's LoRAs combine before the block
vector is applied, for testing which bulk merge keeps the most detail:

- stack (add): every LoRA at full scale (sums with the block ratio).
- average: each LoRA divided by the applied-file count (true mean at 1.0s).
- max/min magnitude (module): per model module, keep only the LoRA whose
  delta magnitude is largest/smallest. Modules that cannot be ranked pass
  through at stack strength.
- strongest file only: apply just the file with the largest overall delta.

Module magnitudes use ||up|| * ||down|| * alpha/rank from the raw LoRA
factors, mirroring comfy.weight_adapter's delta scale without materializing
full merged deltas.
"""

import logging
import os

import torch

import folder_paths
import comfy.utils
import comfy.lora

from .krea2_block_lora import KREA2_BLOCK_GROUPS, _normalize_module
from .moss_image_block_lora import MOSS_BLOCK_GROUPS, _normalize_module as _normalize_module_moss
from .minimax_h3_block_lora import (
    H3_BLOCK_GROUPS,
    H3_IMAGE_BLOCK_GROUPS,
    _normalize_module as _normalize_module_h3,
)

ROOT_FOLDER = "(loras root)"
OTHER_GROUP = "other."

_MERGE_ADD = "stack (add)"
_MERGE_AVG = "average"
_MERGE_MAX = "max magnitude (module)"
_MERGE_MIN = "min magnitude (module)"
_MERGE_BEST = "strongest file only"
MERGE_METHODS = (_MERGE_ADD, _MERGE_AVG, _MERGE_MAX, _MERGE_MIN, _MERGE_BEST)

_LORA_EXTS = (".safetensors", ".sft", ".pt", ".pth", ".ckpt", ".bin")
_MAG_UP = (".lora_up.weight", ".lora_A.weight")
_MAG_DOWN = (".lora_down.weight", ".lora_B.weight")
_TAIL_RESIDUE = (".lora.down.weight", ".lora.up.weight", ".lora.down", ".lora.up",
                 ".lora_a.weight", ".lora_b.weight", ".lora_a", ".lora_b", ".weight", ".bias")
_MAX_CACHE_FILES = 4

ALL_GROUPS = tuple(KREA2_BLOCK_GROUPS) + (OTHER_GROUP,)
MOSS_ALL_GROUPS = tuple(MOSS_BLOCK_GROUPS) + (OTHER_GROUP,)
H3_ALL_GROUPS = tuple(H3_BLOCK_GROUPS) + (OTHER_GROUP,)
H3_IMAGE_ALL_GROUPS = tuple(H3_IMAGE_BLOCK_GROUPS) + (OTHER_GROUP,)


def group_widget_name(group):
    """Widget name for a block group ("blocks.3." -> "blocks_3")."""
    return group.rstrip(".").replace(".", "_")


def _canonical_module(raw, normalize=_normalize_module):
    module = normalize(raw)
    # Kohya underscore-style keys keep a flattened tail after renames.
    for tail in _TAIL_RESIDUE:
        if module.endswith(tail):
            module = module[:-len(tail)]
            break
    if module.startswith("base_model.model."):
        module = module[len("base_model.model."):]
    return module


def _frobenius(tensor):
    return float(torch.linalg.matrix_norm(tensor.detach().float().reshape(tensor.shape[0], -1)).item())


def _module_magnitudes(state_dict, normalize=_normalize_module):
    """Canonical module -> ||up|| * ||down|| * (alpha/rank or 1).

    The scale mirrors comfy.weight_adapter: delta = (alpha/rank) * up @ down.
    """
    ups = {}
    downs = {}
    alphas = {}
    for key in state_dict.keys():
        for suffix in _MAG_UP:
            if key.endswith(suffix):
                ups.setdefault(_canonical_module(key[:-len(suffix)], normalize), []).append(state_dict[key])
                break
        for suffix in _MAG_DOWN:
            if key.endswith(suffix):
                downs.setdefault(_canonical_module(key[:-len(suffix)], normalize), []).append(state_dict[key])
                break
        if key.endswith(".alpha"):
            module = _canonical_module(key[:-len(".alpha")], normalize)
            if module not in alphas:
                try:
                    alphas[module] = float(state_dict[key].detach().reshape(-1)[0].item())
                except (RuntimeError, ValueError):
                    pass
    mags = {}
    for module, up_list in ups.items():
        down_list = downs.get(module)
        if not down_list:
            continue
        down = down_list[0]
        rank = down.reshape(down.shape[0], -1).shape[0]
        alpha = alphas.get(module)
        scale = alpha / rank if alpha is not None else 1.0
        mags[module] = _frobenius(up_list[0]) * _frobenius(down) * scale
    return mags


def _block_group_of_key(k_unet, groups=ALL_GROUPS):
    for group in groups:
        if k_unet == group[:-1] or k_unet.startswith(group):
            return group
    return OTHER_GROUP


def _list_lora_files(folder, recursive):
    """LoRA files inside a loras-relative folder (first matching root wins)."""
    folder = "" if folder == ROOT_FOLDER else folder.replace("\\", "/")
    files = []
    for root in folder_paths.get_folder_paths("loras"):
        base = os.path.join(root, folder)
        if not os.path.isdir(base):
            continue
        base_real = os.path.realpath(base)
        root_real = os.path.realpath(root)
        if os.path.commonpath([root_real, base_real]) != root_real:
            continue  # traversal guard
        if recursive:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for name in filenames:
                    if name.lower().endswith(_LORA_EXTS):
                        files.append(os.path.relpath(os.path.join(dirpath, name), base).replace("\\", "/"))
        else:
            for name in sorted(os.listdir(base)):
                path = os.path.join(base, name)
                if os.path.isfile(path) and name.lower().endswith(_LORA_EXTS):
                    files.append(name)
        break
    files.sort()
    return files


def _folder_choices():
    choices = {ROOT_FOLDER}
    for name in folder_paths.get_filename_list("loras"):
        parts = name.replace("\\", "/").split("/")[:-1]
        for i in range(1, len(parts) + 1):
            choices.add("/".join(parts[:i]))
    return sorted(choices)


class _LoraFolderBulkBase:
    """Applies every LoRA in a folder, scaled per model block (0.0 default).

    All files in the folder are combined with `method`; each block widget is a
    multiplier for that block across the whole collection. A block widget of
    0.0 (the default) silences that block entirely, so a 40-LoRA stack only
    fires the blocks you turn up.

    Subclasses pick the block layout: `GROUPS` is the block-group tuple (with
    OTHER_GROUP appended) and `_normalize` canonicalizes raw LoRA keys into
    native module names for that model.
    """

    GROUPS = ALL_GROUPS
    _normalize = staticmethod(_normalize_module)

    def __init__(self):
        self._lora_cache = {}
        self._mag_cache = None

    @classmethod
    def INPUT_TYPES(s):
        required = {
            "model": ("MODEL",),
            "folder": (_folder_choices(),),
            "method": (MERGE_METHODS, {"default": _MERGE_ADD,
                                       "tooltip": "How the folder's LoRAs combine before the block vector applies."}),
            "recursive": ("BOOLEAN", {"default": True,
                                      "tooltip": "Also load LoRAs in subfolders of the selected folder."}),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                                   "tooltip": "Global multiplier on top of every block weight."}),
        }
        block_info = {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                      "tooltip": "Block multiply across every LoRA in the folder; 0.0 silences the block."}
        for group in s.GROUPS:
            required[group_widget_name(group)] = ("FLOAT", dict(block_info))
        return {"required": required}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "doit"
    CATEGORY = "loaders"

    DESCRIPTION = (
        "Bulk LoRA loader: point it at a loras folder and scale every "
        "model block from 0.0 (off) to 1.0 across the whole collection."
    )

    def _load_lora(self, rel_path):
        path = folder_paths.get_full_path("loras", rel_path)
        if path is None:
            return None
        state_dict = self._lora_cache.get(path)
        if state_dict is None:
            state_dict = comfy.utils.load_torch_file(path, safe_load=True)
            while len(self._lora_cache) >= _MAX_CACHE_FILES:
                self._lora_cache.pop(next(iter(self._lora_cache)))
            self._lora_cache[path] = state_dict
        return path, state_dict

    def _magnitudes(self, entries):
        cache_key = tuple(entry[0] for entry in entries)
        if self._mag_cache and self._mag_cache[0] == cache_key:
            return self._mag_cache[1]
        mags = [_module_magnitudes(state_dict, self._normalize) for _, state_dict, _ in entries]
        self._mag_cache = (cache_key, mags)
        return mags

    def _apply_grouped(self, patcher, loaded, ratio_of, strength, divide_by):
        """Add patches sharing a block ratio in one call per distinct ratio."""
        buckets = {}
        for key, value in loaded.items():
            key_str = key[0] if isinstance(key, tuple) else key
            ratio = ratio_of(key_str)
            if not ratio:
                continue
            buckets.setdefault(ratio, {})[key] = value
        for ratio, patches in buckets.items():
            scale = strength * ratio
            if divide_by:
                scale /= divide_by
            patcher.add_patches(patches, scale)

    @staticmethod
    def _native_key(key_str):
        return key_str[len("diffusion_model."):] if key_str.startswith("diffusion_model.") else key_str

    def doit(self, model, folder=ROOT_FOLDER, method=_MERGE_ADD, recursive=True, strength=1.0, **block_widgets):
        groups = self.GROUPS
        ratios = {}
        for group in groups:
            widget = block_widgets.pop(group_widget_name(group), None)
            try:
                ratios[group] = float(widget) if widget is not None else 0.0
            except (TypeError, ValueError):
                ratios[group] = 0.0

        report = ["method: {}, folder: {}, strength: {}".format(method, folder, strength),
                  "blocks up: " + (", ".join("{}={}".format(g, ratios[g]) for g in groups if ratios[g])
                                   or "none (model unchanged)")]
        if not isinstance(strength, (int, float)) or strength == 0:
            return (model, "\n".join(report + ["strength is 0: model unchanged."]))
        if not any(ratios[g] for g in groups):
            return (model, "\n".join(report))

        files = _list_lora_files(folder, recursive)
        if not files:
            return (model, "\n".join(report + ["No LoRA files found in the folder."]))

        entries = []  # (name, state_dict, path)
        prefix = "" if folder == ROOT_FOLDER else folder.replace("\\", "/")
        for rel in files:
            # _list_lora_files paths are relative to the folder; resolve them
            # against the loras root with the folder prefix included.
            rel_full = "{}/{}".format(prefix, rel) if prefix else rel
            loaded_pair = self._load_lora(rel_full)
            if loaded_pair is None:
                report.append("missing: {}".format(rel))
                continue
            entries.append((rel, loaded_pair[1], loaded_pair[0]))
        if not entries:
            return (model, "\n".join(report))

        key_map = comfy.lora.model_lora_keys_unet(model.model)
        patcher = model.clone()

        if method in (_MERGE_MAX, _MERGE_MIN):
            mags = self._magnitudes(entries)
            winner_index = {}
            for index, mag_map in enumerate(mags):
                for module, mag in mag_map.items():
                    best = winner_index.get(module)
                    if best is None:
                        winner_index[module] = index
                    elif (mag > mags[best][module]) if method == _MERGE_MAX else (mag < mags[best][module]):
                        winner_index[module] = index

            # Map native patch keys back to the raw module each file actually
            # has, compared in canonical form so trainer conventions align.
            native_to_raw = {}
            for raw, native in key_map.items():
                native_key = native[0] if isinstance(native, tuple) else native
                native_to_raw.setdefault(native_key, []).append(raw)

            for index, (name, state_dict, _) in enumerate(entries):
                loaded = comfy.lora.load_lora(state_dict, key_map)
                file_modules = {_canonical_module(raw, self._normalize) for raw in state_dict}

                def ratio_of(key_str, index=index, file_modules=file_modules):
                    module = None
                    for raw in native_to_raw.get(key_str, ()):
                        candidate = _canonical_module(raw, self._normalize)
                        if candidate in file_modules:
                            module = candidate
                            break
                    if module is not None and winner_index.get(module) != index:
                        return 0.0  # another file wins this module
                    return ratios[_block_group_of_key(self._native_key(key_str), groups)]

                self._apply_grouped(patcher, loaded, ratio_of, strength, None)
            report.append("{}: module winners at block weights".format(method))
        elif method == _MERGE_BEST:
            mags = self._magnitudes(entries)
            totals = [sum(mag_map.values()) for mag_map in mags]
            best_index = max(range(len(totals)), key=lambda i: totals[i])
            loaded = comfy.lora.load_lora(entries[best_index][1], key_map)
            self._apply_grouped(patcher, loaded,
                                lambda key_str: ratios[_block_group_of_key(self._native_key(key_str), groups)],
                                strength, None)
            report.append("applied: {} (strongest, total {})".format(entries[best_index][0], totals[best_index]))
        else:
            applied_count = len(entries)
            for name, state_dict, _ in entries:
                loaded = comfy.lora.load_lora(state_dict, key_map)
                self._apply_grouped(patcher, loaded,
                                    lambda key_str: ratios[_block_group_of_key(self._native_key(key_str), groups)],
                                    strength, applied_count if method == _MERGE_AVG else None)
            report.append("{}: {} files x block vector".format(method, applied_count))

        # Drop LoRA state dicts that the next run does not need.
        wanted = {entry[2] for entry in entries}
        for cache_path in list(self._lora_cache.keys()):
            if cache_path not in wanted:
                self._lora_cache.pop(cache_path)

        return (patcher, "\n".join(report))


class LoraFolderBulkBlocks(_LoraFolderBulkBase):
    """Bulk Krea 2 variant: block layout mirrors ModelMergeKrea2."""

    GROUPS = ALL_GROUPS
    _normalize = staticmethod(_normalize_module)


class LoraFolderBulkBlocksMossImage(_LoraFolderBulkBase):
    """Bulk MOSS-Image variant: block layout mirrors the native MOSSImage modules."""

    GROUPS = MOSS_ALL_GROUPS
    _normalize = staticmethod(_normalize_module_moss)


class LoraFolderBulkBlocksH3(_LoraFolderBulkBase):
    """Bulk MiniMax H3 variant: block layout mirrors the native MiniMaxH3Model modules."""

    GROUPS = H3_ALL_GROUPS
    _normalize = staticmethod(_normalize_module_h3)


class LoraFolderBulkBlocksH3Image(_LoraFolderBulkBase):
    """Bulk MiniMax H3 Image variant: the H3 layout without the audio modules."""

    GROUPS = H3_IMAGE_ALL_GROUPS
    _normalize = staticmethod(_normalize_module_h3)


NODE_CLASS_MAPPINGS = {
    "LoraFolderBulkBlocks": LoraFolderBulkBlocks,
    "LoraFolderBulkBlocksMossImage": LoraFolderBulkBlocksMossImage,
    "LoraFolderBulkBlocksH3": LoraFolderBulkBlocksH3,
    "LoraFolderBulkBlocksH3Image": LoraFolderBulkBlocksH3Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraFolderBulkBlocks": "Load LoRA Folder (Block)",
    "LoraFolderBulkBlocksMossImage": "Load LoRA Folder Moss Image (Block)",
    "LoraFolderBulkBlocksH3": "Load LoRA Folder H3 (Block)",
    "LoraFolderBulkBlocksH3Image": "Load LoRA Folder H3 Image (Block)",
}

__all__ = [
    "LoraFolderBulkBlocks",
    "LoraFolderBulkBlocksMossImage",
    "LoraFolderBulkBlocksH3",
    "LoraFolderBulkBlocksH3Image",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
