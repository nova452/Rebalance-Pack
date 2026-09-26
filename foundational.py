"""
    Foundational utility nodes for the OmniNode custom node pack.

    LoadImages - Load a range of images from a directory (with sibling
        .txt captions and alpha masks).
    LoadImageNewest - Load the newest images from a directory (by mtime).
    LoadImageFull - Load a single uploaded image with mask, caption and
        metadata outputs.
    Any        - Wildcard passthrough node.
    Input      - Multiline string input node.
    FloatExact - High-precision float input.
    FloatNormalized - Float input normalized to the 0.00-1.00 range.
    Float10    - Float input in the -10.00 to 10.00 range.
    Concatenate - Concatenate up to nine string sections into one.
    Switch     - Switch between two wildcard inputs based on a boolean.
    StringInline - Multiline string with {{text_a}}/{{text_b}} substitution.
    StringInline5 - Multiline string with {{text_a}}..{{text_e}} substitution.
    StringToList - Split a string into a list by delimiter.
    ListStringIndex - Pick a single item from a string list by index.
    SolidColorImage - Create a blank color image of given dimensions.
    SaveImage - Save an image (png/jpg/jpeg/bmp) with user-supplied metadata

"""

import os
import hashlib

import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

try:
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False

try:
    import folder_paths
    _FOLDER_PATHS_AVAILABLE = True
except ImportError:
    folder_paths = None
    _FOLDER_PATHS_AVAILABLE = False


_VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}


class _AnyType(str):
    """Wildcard type that matches any ComfyUI socket type."""

    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


_ANY = _AnyType("*")


def _normalize_path(path):
    """Normalize a path: strip whitespace, unify separators, and resolve
    relative paths against the current working directory."""
    if not path:
        return path
    path = path.strip()
    path = path.replace('\\', '/')
    path = os.path.normpath(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def _load_single_image(file_path):
    """
    Mirrors the behavior of ComfyUI's LoadImage but operates on an arbitrary
    path. Returns None for the image if the file could not be loaded.
    """
    try:
        img = node_helpers.pillow(Image.open, file_path)
    except Exception as e:
        print(f"Warning: Could not load image {file_path}: {e}")
        return None, None, None

    i = node_helpers.pillow(ImageOps.exif_transpose, img)

    if i.mode == 'I':
        i = i.point(lambda x: x * (1 / 65535))

    image = i.convert("RGB")
    w, h = image.size

    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)[None,]

    if 'A' in i.getbands():
        mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask_np)
        mask_inverted = 1.0 - mask
    elif i.mode == 'P' and 'transparency' in i.info:
        rgba = i.convert('RGBA')
        mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask_np)
        mask_inverted = 1.0 - mask
    else:
        mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
        mask_inverted = 1.0 - mask

    return image_tensor, mask.unsqueeze(0), mask_inverted.unsqueeze(0)


def _read_image_metadata(image_path):

    try:
        with Image.open(image_path) as img:
            info = img.info or {}
    except Exception as e:
        print(f"Warning: Could not read metadata from {image_path}: {e}")
        return ""

    if not info:
        return ""

    lines = []
    for key, value in info.items():
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace")
            except Exception:
                value = repr(value)
        elif not isinstance(value, str):
            value = str(value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _read_sibling_txt(image_path):

    base, _ = os.path.splitext(image_path)
    txt_path = base + ".txt"
    if not os.path.isfile(txt_path):
        return ""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read txt file {txt_path}: {e}")
        return ""


class LoadImages:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "directory_path": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "path/to/directory..",
                "tooltip": "Path to the directory containing images.",
            }),
            "start_index": ("INT", {
                "default": 0,
                "min": 0,
                "step": 1,
                "tooltip": "Index of the first image to load (sorted alphabetically).",
            }),
            "cap": ("INT", {
                "default": 4,
                "min": 1,
                "max": 1024,
                "step": 1,
                "tooltip": "Number of images to load.",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "caption", "metadata")
    OUTPUT_IS_LIST = (True, True, True, True, True)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, directory_path, start_index, cap):
        normalized_path = _normalize_path(directory_path)

        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        files = []
        for f in os.listdir(normalized_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in _VALID_EXTENSIONS:
                files.append(os.path.join(normalized_path, f))

        files.sort()

        end_index = start_index + cap
        selected_files = files[start_index:end_index]

        if not selected_files:
            raise ValueError(
                f"No images found in range [{start_index}:{end_index}] in directory: {directory_path}"
            )

        output_images = []
        output_masks = []
        output_filenames = []
        output_captions = []
        output_metadata = []

        for file_path in selected_files:
            image_tensor, mask, mask_inverted = _load_single_image(file_path)
            if image_tensor is None:
                continue

            output_images.append(image_tensor)
            output_masks.append(mask)
            output_filenames.append(os.path.basename(file_path))
            output_captions.append(_read_sibling_txt(file_path))
            output_metadata.append(_read_image_metadata(file_path))

        if not output_images:
            raise ValueError("No valid images loaded (checked dimensions and validity).")

        return (output_images, output_masks, output_filenames,
                output_captions, output_metadata)

    @classmethod
    def IS_CHANGED(cls, directory_path, start_index, cap):
        normalized_path = _normalize_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""

        files = []
        try:
            for f in os.listdir(normalized_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in _VALID_EXTENSIONS:
                    files.append(os.path.join(normalized_path, f))
        except Exception:
            return float("NaN")

        files.sort()
        end_index = start_index + cap
        selected_files = files[start_index:end_index]

        m = hashlib.sha256()
        for p in selected_files:
            try:
                m.update(p.encode('utf-8'))
                m.update(str(os.path.getmtime(p)).encode('utf-8'))
            except Exception:
                pass
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, directory_path, start_index, cap):
        if directory_path is None:
            return True
        if not isinstance(directory_path, str):
            return True
        normalized_path = _normalize_path(directory_path)
        if not normalized_path:
            return True
        if not os.path.isdir(normalized_path):
            return True
        return True


class LoadImageNewest:
    """Load the newest images from a directory, sorted by modification time
    (newest first)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "directory_path": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "path/to/directory..",
                "tooltip": "Path to the directory containing images.",
            }),
            "cap": ("INT", {
                "default": 1,
                "min": 1,
                "max": 1024,
                "step": 1,
                "tooltip": "Number of newest images to load.",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "caption", "metadata")
    OUTPUT_IS_LIST = (True, True, True, True, True)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, directory_path, cap):
        normalized_path = _normalize_path(directory_path)

        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        files = []
        for f in os.listdir(normalized_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in _VALID_EXTENSIONS:
                files.append(os.path.join(normalized_path, f))

        # Sort by modification time, newest first.
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        selected_files = files[:cap]

        if not selected_files:
            raise ValueError(
                f"No images found in directory: {directory_path}"
            )

        output_images = []
        output_masks = []
        output_filenames = []
        output_captions = []
        output_metadata = []

        for file_path in selected_files:
            image_tensor, mask, mask_inverted = _load_single_image(file_path)
            if image_tensor is None:
                continue

            output_images.append(image_tensor)
            output_masks.append(mask)
            output_filenames.append(os.path.basename(file_path))
            output_captions.append(_read_sibling_txt(file_path))
            output_metadata.append(_read_image_metadata(file_path))

        if not output_images:
            raise ValueError("No valid images loaded (checked dimensions and validity).")

        return (output_images, output_masks, output_filenames,
                output_captions, output_metadata)

    @classmethod
    def IS_CHANGED(cls, directory_path, cap):
        normalized_path = _normalize_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""

        files = []
        try:
            for f in os.listdir(normalized_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in _VALID_EXTENSIONS:
                    files.append(os.path.join(normalized_path, f))
        except Exception:
            return float("NaN")

        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        selected_files = files[:cap]

        m = hashlib.sha256()
        for p in selected_files:
            try:
                m.update(p.encode('utf-8'))
                m.update(str(os.path.getmtime(p)).encode('utf-8'))
            except Exception:
                pass
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, directory_path, cap):
        if directory_path is None:
            return True
        if not isinstance(directory_path, str):
            return True
        normalized_path = _normalize_path(directory_path)
        if not normalized_path:
            return True
        if not os.path.isdir(normalized_path):
            return True
        return True


class LoadImageFull:
    """Load a single image via an upload widget, with image, mask, filename,
    caption, and metadata outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        )
        return {"required": {
            "image": (files, {"image_upload": True}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "caption", "metadata")
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, image):
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)

        image_tensor, mask, mask_inverted = _load_single_image(image_path)
        if image_tensor is None:
            raise ValueError(f"Could not load image: {image}")

        caption = _read_sibling_txt(image_path)
        metadata = _read_image_metadata(image_path)

        return (image_tensor, mask, image, caption, metadata)

    @classmethod
    def IS_CHANGED(cls, image):
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)
        if not os.path.isfile(image_path):
            return ""
        m = hashlib.sha256()
        try:
            m.update(image_path.encode('utf-8'))
            m.update(str(os.path.getmtime(image_path)).encode('utf-8'))
        except Exception:
            return float("NaN")
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if image is None:
            return True
        if not isinstance(image, str):
            return True
        return True


class Any:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "any": (_ANY, {}),
        }}

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("any",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, any):
        return (any,)


class Input:

    SEARCH_ALIASES = [
        "string",
        "text",
        "text box",
        "prompt",
        "multiline",
        "input text",
        "float",
        "INT",
        "text string",
    ]
    ESSENTIALS_CATEGORY = "Basics"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STRING", {
                "default": "",
                "multiline": True,
            }),
        }}

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("OUT",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, string):
        if isinstance(string, str):
            text = string.strip()
            lowered = text.lower()
            if lowered in ("true", "false"):
                return (lowered == "true",)
            try:
                return (int(text),)
            except (ValueError, TypeError):
                pass
            try:
                return (float(text),)
            except (ValueError, TypeError):
                pass
        return (string,)


class FloatExact:

    SEARCH_ALIASES = [
        "float exact",
        "float",
        "decimal",
        "precision",
        "double",
        "high precision float",
    ]
    ESSENTIALS_CATEGORY = "Basics"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "value": ("FLOAT", {
                "default": 0.0,
                "min": -1e18,
                "max": 1e18,
                "step": 0.0000000001,
                "round": 0.0000000001,
            }),
        }}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, value):
        # ComfyUI snaps widget values to the step using binary float math,
        # which introduces artifacts (e.g. 0.30 -> 0.30499999999999994).
        # Round to the widget's precision to clean those up.
        return (round(value, 10),)


class FloatNormalized:
    """Float input normalized to the 0.00-1.00 range."""

    SEARCH_ALIASES = [
        "float normalized",
        "float 1.00",
        "normalized float",
        "unit float",
        "0 to 1",
    ]
    ESSENTIALS_CATEGORY = "Basics"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "value": ("FLOAT", {
                "default": 0.0,
                "min": 0.0,
                "max": 1.00,
                "step": 0.01,
                "round": 0.01,
            }),
        }}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, value):
        # Snap to the 0.01 grid to remove binary float step artifacts
        # (e.g. 0.30 -> 0.30499999999999994).
        return (round(value, 2),)


class Float10:
    """Float input in the -10.00 to 10.00 range with 0.01 step."""

    SEARCH_ALIASES = [
        "float 10.00",
        "float 10",
        "float -10 to 10",
        "float range 10",
        "signed float 10",
    ]
    ESSENTIALS_CATEGORY = "Basics"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "value": ("FLOAT", {
                "default": 0.0,
                "min": -10.00,
                "max": 10.00,
                "step": 0.01,
                "round": 0.01,
            }),
        }}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, value):
        # Snap to the 0.01 grid to remove binary float step artifacts
        # (e.g. 0.30 -> 0.30499999999999994).
        return (round(value, 2),)


class Concatenate:

    SEARCH_ALIASES = [
        "concatenate",
        "text concat",
        "join text",
        "merge text",
        "combine strings",
        "string concat",
        "append text",
        "combine text",
    ]
    ESSENTIALS_CATEGORY = "Text"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {

            "string_1": ("STRING", {"default": "", "multiline": False}),
            "string_2": ("STRING", {"default": "", "multiline": False}),
            "string_3": ("STRING", {"default": "", "multiline": False}),
            "string_4": ("STRING", {"default": "", "multiline": False}),
            "string_5": ("STRING", {"default": "", "multiline": False}),
            "string_6": ("STRING", {"default": "", "multiline": False}),
            "string_7": ("STRING", {"default": "", "multiline": False}),
            "string_8": ("STRING", {"default": "", "multiline": False}),
            "string_9": ("STRING", {"default": "", "multiline": False}),
            "delimiter": ("STRING", {
                "default": "",
                "multiline": False,
                "tooltip": "Inserted between non-empty sections.",
            }),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, delimiter, string_1, string_2, string_3, string_4,
             string_5, string_6, string_7, string_8, string_9):
        sections = [
            string_1, string_2, string_3, string_4, string_5,
            string_6, string_7, string_8, string_9,
        ]
        parts = [s for s in sections if s is not None and s.strip() != ""]
        return (delimiter.join(parts),)


class Switch:
    """Switch between two wildcard (any-type) inputs based on a boolean"""

    SEARCH_ALIASES = [
        "switch",
        "omni switch",
        "toggle",
        "if",
        "boolean switch",
        "any switch",
    ]
    ESSENTIALS_CATEGORY = "Logic"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True passes on_true, False passes on_false.",
                }),
            },
            "optional": {
                "on_true": (_ANY, {"lazy": True}),
                "on_false": (_ANY, {"lazy": True}),
            },
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("OUT",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def check_lazy_status(self, switch, on_true=None, on_false=None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    def main(self, switch, on_true=None, on_false=None):
        return (on_true if switch else on_false,)


class StringInline:
    """Multiline string with optional ``{{text_a}}`` / ``{{text_b}}`` token substitution"""

    SEARCH_ALIASES = [
        "string inline",
        "inline string",
        "text template",
        "string template",
        "text replace",
        "token replace",
        "text_a",
        "text_b",
    ]
    ESSENTIALS_CATEGORY = "Text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "{{text_a}}",
                    "multiline": True,
                }),
            },
            "optional": {
                "text_a": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Replaces every {{text_a}} token in the text.",
                }),
                "text_b": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Replaces every {{text_b}} token in the text.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, text, text_a=None, text_b=None):
        out = text
        if text_a is not None:
            out = out.replace("{{text_a}}", text_a)
        if text_b is not None:
            out = out.replace("{{text_b}}", text_b)
        return (out,)


class StringInline5:

    SEARCH_ALIASES = [
        "string inline 5",
        "inline string 5",
        "text template 5",
        "string template 5",
        "text replace 5",
        "token replace 5",
        "text_a",
        "text_b",
        "text_c",
        "text_d",
        "text_e",
    ]
    ESSENTIALS_CATEGORY = "Text"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "{{text_a}}",
                    "multiline": True,
                }),
            },
            "optional": {
                "text_a": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Replaces every {{text_a}} token in the text.",
                }),
                "text_b": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Replaces every {{text_b}} token in the text.",
                }),
                "text_c": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Replaces every {{text_c}} token in the text.",
                }),
                "text_d": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Replaces every {{text_d}} token in the text.",
                }),
                "text_e": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Replaces every {{text_e}} token in the text.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, text, text_a=None, text_b=None, text_c=None,
             text_d=None, text_e=None):
        out = text
        if text_a is not None:
            out = out.replace("{{text_a}}", text_a)
        if text_b is not None:
            out = out.replace("{{text_b}}", text_b)
        if text_c is not None:
            out = out.replace("{{text_c}}", text_c)
        if text_d is not None:
            out = out.replace("{{text_d}}", text_d)
        if text_e is not None:
            out = out.replace("{{text_e}}", text_e)
        return (out,)


class StringToList:

    SEARCH_ALIASES = [
        "string to list",
        "split string",
        "text to list",
        "delimiter",
        "split",
        "list from string",
        "batch string",
    ]
    ESSENTIALS_CATEGORY = "Text"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STRING", {
                "default": "",
                "multiline": True,
                "tooltip": "Text to split into a list.",
            }),
            "delimiter": ("STRING", {
                "default": "\n",
                "multiline": False,
                "tooltip": "Separator used to split the string.",
            }),
            "strip_items": ("BOOLEAN", {
                "default": True,
                "tooltip": "Strip leading/trailing whitespace from each item.",
            }),
            "skip_empty": ("BOOLEAN", {
                "default": True,
                "tooltip": "Remove empty items from the result.",
            }),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, string, delimiter, strip_items, skip_empty):
        if not delimiter:
            raise ValueError("Delimiter cannot be empty.")

        # Interpret common escape sequences so a literal "\\n" in the widget
        # splits on actual newlines in the input string.
        delimiter = (
            delimiter.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
        )

        parts = string.split(delimiter)

        if strip_items:
            parts = [part.strip() for part in parts]

        if skip_empty:
            parts = [part for part in parts if part != ""]

        return (parts,)


class ListStringIndex:
    """Pick a single item from a string list by index."""

    SEARCH_ALIASES = [
        "list string index",
        "list index",
        "list item",
        "pick from list",
        "list get",
        "string list index",
        "index list",
    ]
    ESSENTIALS_CATEGORY = "Text"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "list": (_ANY, {
                "forceInput": True,
                "tooltip": "A list of strings to pick from.",
            }),
            "index": ("INT", {
                "default": 0,
                "min": 0,
                "max": 9999,
                "step": 1,
                "tooltip": "Zero-based index of the item to return.",
            }),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False,)

    @staticmethod
    def _is_sequence(value):
        # Duck-typed sequence check that excludes strings/bytes.
        if isinstance(value, (str, bytes)):
            return False
        return hasattr(value, "__len__") and hasattr(value, "__getitem__")

    def main(self, list, index):
        # ComfyUI passes INPUT_IS_LIST inputs as lists even for a single
        # value, and the index arrives as a list of values (one per item).
        # NOTE: the parameter is named `list` (matching the widget), so we
        # must not use the builtin `list` type inside this function.
        if list is None:
            raise ValueError("List input is required.")

        items = list if self._is_sequence(list) else [list]

        # If a single non-list value was connected, ComfyUI wraps it in a
        # one-element list; if that element is itself a sequence (a node
        # returning a whole Python list as one output), unwrap it.
        if len(items) == 1 and self._is_sequence(items[0]):
            items = items[0]

        if len(items) == 0:
            raise ValueError("Cannot index into an empty list.")

        # With INPUT_IS_LIST, every input arrives as a list of the same
        # length; take the first index value.
        if self._is_sequence(index):
            if len(index) == 0:
                raise ValueError("Index input is empty.")
            index = index[0]

        if not isinstance(index, int):
            try:
                index = int(index)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid index value: {index!r}")

        if index < 0 or index >= len(items):
            raise ValueError(
                f"Index {index} out of range for list of length {len(items)}."
            )
        item = items[index]
        if item is None:
            return ("",)
        return (str(item),)


class SolidColorImage:

    SEARCH_ALIASES = [
        "solid color",
        "color image",
        "blank image",
        "solid",
        "fill color",
        "color fill",
        "background color",
    ]
    ESSENTIALS_CATEGORY = "Image"

    COLOR_NAMES = {
        "transparent": (0.0, 0.0, 0.0, 0.0),
        "none": (0.0, 0.0, 0.0, 0.0),
        "black": (0.0, 0.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0, 1.0),
        "green": (0.0, 1.0, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "yellow": (1.0, 1.0, 0.0, 1.0),
        "cyan": (0.0, 1.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0, 1.0),
        "orange": (1.0, 0.647, 0.0, 1.0),
        "purple": (0.5, 0.0, 0.5, 1.0),
        "pink": (1.0, 0.753, 0.796, 1.0),
        "brown": (0.647, 0.165, 0.165, 1.0),
        "gray": (0.5, 0.5, 0.5, 1.0),
        "grey": (0.5, 0.5, 0.5, 1.0),
        "silver": (0.753, 0.753, 0.753, 1.0),
        "maroon": (0.5, 0.0, 0.0, 1.0),
        "olive": (0.5, 0.5, 0.0, 1.0),
        "teal": (0.0, 0.5, 0.5, 1.0),
        "navy": (0.0, 0.0, 0.5, 1.0),
        "lime": (0.0, 1.0, 0.0, 1.0),
        "indigo": (0.294, 0.0, 0.51, 1.0),
        "violet": (0.933, 0.51, 0.933, 1.0),
        "beige": (0.961, 0.961, 0.863, 1.0),
    }

    @staticmethod
    def _parse_color(color):
        if not isinstance(color, str):
            color = str(color)
        color = color.strip()
        named = SolidColorImage.COLOR_NAMES.get(color.lower())
        if named is not None:
            return named
        color = color.lstrip('#')
        if len(color) == 3 or len(color) == 4:
            color = ''.join(c * 2 for c in color)
        if len(color) == 6:
            color += "FF"
        if len(color) != 8:
            raise ValueError(
                f"Invalid color '{color}'. Expected #RRGGBB or #RRGGBBAA hex format."
            )
        try:
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
            a = int(color[6:8], 16) / 255.0
        except ValueError:
            raise ValueError(
                f"Invalid color '{color}'. Expected #RRGGBB or #RRGGBBAA hex format."
            )
        return (r, g, b, a)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {
                "default": 1000,
                "min": 1,
                "step": 1,
            }),
            "height": ("INT", {
                "default": 1000,
                "min": 1,
                "step": 1,
            }),
            "color": ("STRING", {
                "default": "transparent",
                "multiline": False,
                "placeholder": "#RRGGBB, #RRGGBBAA, or a color name (transparent, red, orange, ...)",
            }),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "Rebalance-Pack/foundational"

    def main(self, width, height, color):
        r, g, b, a = self._parse_color(color)
        img_np = np.zeros((height, width, 4), dtype=np.float32)
        img_np[:, :, 0] = r
        img_np[:, :, 1] = g
        img_np[:, :, 2] = b
        img_np[:, :, 3] = a
        image_tensor = torch.from_numpy(img_np)[None,]
        return (image_tensor,)

    @classmethod
    def IS_CHANGED(cls, width, height, color):
        return f"{width}_{height}_{color}"


class SaveImage:

    SEARCH_ALIASES = [
        "save image",
        "save",
        "export image",
        "write image",
        "save png",
        "save jpg",
        "save jpeg",
        "save bmp",
        "metadata save",
        "no metadata save",
    ]
    ESSENTIALS_CATEGORY = "Image"

    _SAVE_FORMATS = ["png", "jpg", "jpeg", "bmp"]

    def __init__(self):
        self.output_dir = None
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "tooltip": "Prefix for the saved filenames.",
                }),
                "format": (cls._SAVE_FORMATS, {
                    "default": "png",
                    "tooltip": "Output image format.",
                }),
                "padding": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 10,
                    "step": 1,

                }),
                "metadata": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Metadata text. Supported formats: PNG (text chunk), JPG, JPEG (COM marker)."
                    ),
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
 "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    CATEGORY = "Rebalance-Pack/foundational"

    def save_images(self, images, filename_prefix, format, padding,
                    metadata, prompt=None, extra_pnginfo=None):

        if self.output_dir is None:
            try:
                import folder_paths
                self.output_dir = folder_paths.get_output_directory()
            except ImportError:
                self.output_dir = os.path.join(os.getcwd(), "output")
                os.makedirs(self.output_dir, exist_ok=True)

        format = format.lower()
        if format not in self._SAVE_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Expected one of: {', '.join(self._SAVE_FORMATS)}"
            )

        # Map jpeg -> jpg extension for consistency.
        ext = "jpg" if format in ("jpg", "jpeg") else format

        # Resolve the output folder / counter / subfolder / clean prefix.
        full_output_folder, subfolder, counter, clean_prefix = \
            self._resolve_output(filename_prefix, ext)

        results = []
        subfolder = subfolder or ""
        num_images = len(images)
        max_val = (10 ** padding) - 1 if padding > 0 else None
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # Force RGB for formats that don't support alpha.
            if img.mode != "RGB":
                img.convert("RGB")

            base, fname = self._build_filename(
                clean_prefix, ext, padding, counter,
                batch_number, num_images)
            file_path = os.path.join(full_output_folder, fname)

            # Collision handling: if the file exists, go 1 counter higher
            # until a free slot is found. If the counter limit is exceeded
            # (9999 / 999 / 99 ... depending on padding), fall back to
            # suffixes _1, _2, ... _9, _9_1, _9_2, ... _9_9, _9_9_1, ...
            if os.path.exists(file_path) and max_val is not None:
                while os.path.exists(file_path) and counter < max_val:
                    counter += 1
                    base, fname = self._build_filename(
                        clean_prefix, ext, padding, counter,
                        batch_number, num_images)
                    file_path = os.path.join(full_output_folder, fname)

            if os.path.exists(file_path):
                n = 1
                while os.path.exists(file_path):
                    suffix = self._collision_suffix(n)
                    fname = f"{base}{suffix}.{ext}"
                    file_path = os.path.join(full_output_folder, fname)
                    n += 1

            self._write_image(img, file_path, format, metadata)
            counter += 1

            results.append({
                "filename": fname,
                "subfolder": subfolder,
                "type": self.type,
            })

        # Previews are always hidden (no widget), matching the "Hide" output
        # behavior of the Image Color Match P node.
        return {"ui": {}, "result": (images,)}

    @staticmethod
    def _resolve_output(filename_prefix, ext):
        """Resolve the output folder, subfolder, next counter, and the
        cleaned filename prefix (with any subfolder portion stripped).

        ComfyUI's ``get_save_image_path`` splits a prefix like
        ``shards-bin/shard`` into ``full_output_folder`` (with ``shards-bin``
        appended) and returns ``shard`` as the filename portion. We must use
        that cleaned portion when building filenames, otherwise the subfolder
        gets doubled.
        """
        try:
            import folder_paths
            full_output_folder, filename, counter, subfolder, _ = \
                folder_paths.get_save_image_path(
                    f"{filename_prefix}.{ext}",
                    folder_paths.get_output_directory(),
                )
            # Strip the extension that get_save_image_path appended.
            if filename.endswith(f".{ext}"):
                filename = filename[:-(len(ext) + 1)]
            # ComfyUI's counter scan only matches its own "prefix_00001_"
            # filename shape; our "prefix_0001" files are invisible to it,
            # so it returns a stale counter. Scan the folder ourselves and
            # take the highest of the two.
            scanned = SaveImage._scan_counter(full_output_folder, filename, ext)
            counter = max(counter - 1, scanned) + 1
            return full_output_folder, subfolder, counter, filename
        except ImportError:
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            # Strip any subfolder from the prefix for the fallback path.
            clean = os.path.basename(filename_prefix) or "ComfyUI"
            counter = SaveImage._scan_counter(output_dir, clean, ext) + 1
            return output_dir, "", counter, clean

    @staticmethod
    def _build_filename(prefix, ext, padding, counter, batch_number,
                        num_images):

        if padding > 0:
            max_val = (10 ** padding) - 1
            wrapped = ((counter - 1) % max_val) + 1
            counter_str = f"{wrapped:0{padding}d}"
            base = f"{prefix}_{counter_str}"
        else:
            base = f"{prefix}"

        if num_images > 1:
            base = f"{base}_{batch_number:05}"

        return base, f"{base}.{ext}"

    @staticmethod
    def _collision_suffix(n):

        num_nines = (n - 1) // 9
        last_digit = ((n - 1) % 9) + 1
        return ("_9" * num_nines) + f"_{last_digit}"

    @staticmethod
    def _scan_counter(output_dir, prefix, ext):
        """Scan the output directory for the highest counter value used."""
        highest = 0
        if not os.path.isdir(output_dir):
            return highest
        marker = f"{prefix}_"
        for f in os.listdir(output_dir):
            if not f.startswith(marker) or not f.endswith(f".{ext}"):
                continue
            rest = f[len(marker):]
            num_part = rest.split("_", 1)[0]
            if num_part.isdigit():
                val = int(num_part)
                if val > highest:
                    highest = val
        return highest

    @staticmethod
    def _write_image(img, file_path, format, metadata):

        fmt = format.upper()
        if fmt == "JPEG":
            # JPEG COM marker for arbitrary text.
            if metadata:
                img.save(file_path, format="JPEG",
                         comment=metadata, quality=95)
            else:
                img.save(file_path, format="JPEG", quality=95)
        elif fmt == "BMP":
            # BMP has no native text-metadata support; just save the pixels.
            img.save(file_path, format="BMP")
        else:
            # PNG: write the user metadata
            pnginfo = None
            if metadata:
                pnginfo = PngInfo()
                pnginfo.add_text("meta", metadata)
            img.save(file_path, format="PNG", pnginfo=pnginfo)

    @classmethod
    def IS_CHANGED(cls, images, filename_prefix, format, padding, metadata,
                   prompt=None, extra_pnginfo=None):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "LoadImages": LoadImages,
    "LoadImageNewest": LoadImageNewest,
    "LoadImageFull": LoadImageFull,
    "SolidColorImage": SolidColorImage,
    "Any": Any,
    "Input": Input,
    "FloatExact": FloatExact,
    "FloatNormalized": FloatNormalized,
    "Float10": Float10,
    "Concatenate": Concatenate,
    "Switch": Switch,
    "StringInline": StringInline,
    "StringInline5": StringInline5,
    "StringToList": StringToList,
    "ListStringIndex": ListStringIndex,
    "SaveImages": SaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImages": "Load Images",
    "LoadImageNewest": "Load Image Newest",
    "LoadImageFull": "Load Image Full",
    "SolidColorImage": "Color Image",
    "Any": "Any",
    "Input": "Input",
    "FloatExact": "Float Exact",
    "FloatNormalized": "Float 1.00",
    "Float10": "Float 10.00",
    "Concatenate": "Concatenate",
    "Switch": "Switch",
    "StringInline": "String (Inline)",
    "StringInline5": "String (Inline 5)",
    "StringToList": "String to List",
    "ListStringIndex": "List String Index",
    "SaveImages": "Save Images",
}

__all__ = [
    "LoadImages",
    "LoadImageNewest",
    "LoadImageFull",
    "SolidColorImage",
    "Any",
    "Input",
    "FloatExact",
    "FloatNormalized",
    "Float10",
    "Concatenate",
    "Switch",
    "StringInline",
    "StringInline5",
    "StringToList",
    "ListStringIndex",
    "SaveImage",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
