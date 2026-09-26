import os
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageOps

from comfy.cli_args import args
from comfy.utils import ProgressBar

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    logging.warning("OpenCV not installed; falling back to PIL (slower).")
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Server route: list files in a directory filtered by extension.
# Registered lazily so the combo widget's `remote` option can fetch a fresh
# list (with a refresh button) the same way ComfyUI's built-in Load LoRA does.
# ---------------------------------------------------------------------------
def _register_file_list_route():
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return

    server = PromptServer.instance
    if getattr(server, "_rebalance_file_list_route_registered", False):
        return
    server._rebalance_file_list_route_registered = True

    routes = web.RouteTableDef()

    @routes.get("/rebalance_pack/list_files")
    async def _list_files(request):
        directory_path = request.query.get("directory_path", "")
        ext = request.query.get("ext", "")
        subfolder_depth = int(request.query.get("subfolder_depth", "-1"))

        directory_path = _resolve_directory_path(directory_path)
        if not directory_path or not os.path.isdir(directory_path):
            return web.json_response([])

        exts = tuple(e.strip().lower().lstrip(".") for e in ext.split(",") if e.strip())
        ext_set = frozenset("." + e for e in exts) if exts else None

        files = []
        if subfolder_depth == 0:
            entries = os.listdir(directory_path)
            for name in entries:
                full = os.path.join(directory_path, name)
                if not os.path.isfile(full):
                    continue
                if ext_set is None or os.path.splitext(name)[1].lower() in ext_set:
                    files.append(name)
        else:
            base_depth = directory_path.rstrip(os.sep).count(os.sep)
            for root, _, names in os.walk(directory_path):
                if subfolder_depth != -1:
                    current_depth = root.rstrip(os.sep).count(os.sep) - base_depth
                    if current_depth > subfolder_depth:
                        continue
                for name in names:
                    if ext_set is None or os.path.splitext(name)[1].lower() in ext_set:
                        rel = os.path.relpath(os.path.join(root, name), directory_path)
                        files.append(rel.replace("\\", "/"))
        files.sort()
        return web.json_response(files)

    if not server.app.router.frozen:
        try:
            server.app.add_routes(routes)
        except Exception:
            logging.exception("Rebalance-Pack: failed to register /rebalance_pack/list_files route")


# Register the route eagerly at import time. Custom node modules are loaded
# before the aiohttp router is frozen, so this is safe and guarantees the
# endpoint exists for the ListDisplay frontend.
_register_file_list_route()


# ---------------------------------------------------------------------------
# Server route: list files in a directory filtered by extension.
# Registered lazily so the combo widget's `remote` option can fetch a fresh
# list (with a refresh button) the same way ComfyUI's built-in Load LoRA does.


VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.tga')
_VALID_SET = frozenset(VALID_EXTENSIONS)


def _resolve_directory_path(directory_path):
    if directory_path and not os.path.isabs(directory_path) and args.base_directory:
        return os.path.join(args.base_directory, directory_path)
    return directory_path


def _collect_image_files(directory_path, subfolder_depth):

    image_paths = []
    if subfolder_depth == 0:
        for file in os.listdir(directory_path):
            if os.path.splitext(file)[1].lower() in _VALID_SET:
                image_paths.append(os.path.join(directory_path, file))
    else:
        base_depth = directory_path.rstrip(os.sep).count(os.sep)
        for root, _, files in os.walk(directory_path):
            if subfolder_depth != -1:
                current_depth = root.rstrip(os.sep).count(os.sep) - base_depth
                if current_depth > subfolder_depth:
                    continue
            for file in files:
                if os.path.splitext(file)[1].lower() in _VALID_SET:
                    image_paths.append(os.path.join(root, file))
    image_paths.sort()
    return image_paths


def _interp(src_w, src_h, target_w, target_h):
    return cv2.INTER_AREA if (src_w > target_w or src_h > target_h) else cv2.INTER_LANCZOS4


def _decode_and_resize_cv2(image_path, width, height, method):
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"OpenCV failed to read image: {image_path}")

    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        alpha = None
    else:
        channels = raw.shape[2]
        if channels == 4:
            alpha = raw[:, :, 3]
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        elif channels == 3:
            alpha = None
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        else:
            alpha = None
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    img_h, img_w = raw.shape[:2]

    if (width == -1 and height == -1) or (img_w == width and img_h == height):
        return raw, alpha

    target_w = img_w if width == -1 else width
    target_h = img_h if height == -1 else height

    if method == "stretch":
        resized = cv2.resize(raw, (target_w, target_h), interpolation=_interp(img_w, img_h, target_w, target_h))
        if alpha is not None:
            alpha = cv2.resize(alpha, (target_w, target_h), interpolation=_interp(img_w, img_h, target_w, target_h))
        return resized, alpha

    return _resize_with_aspect_ratio_cv2(raw, alpha, target_w, target_h, method)


def _resize_with_aspect_ratio_cv2(raw, alpha, width, height, mode):
    img_h, img_w = raw.shape[:2]
    aspect_ratio = img_w / img_h
    target_ratio = width / height

    if mode == "crop":
        if aspect_ratio > target_ratio:
            new_width = int(height * aspect_ratio)
            resized = cv2.resize(raw, (new_width, height), interpolation=_interp(img_w, img_h, new_width, height))
            left = (new_width - width) // 2
            out = resized[:, left:left + width]
            alpha_out = None
            if alpha is not None:
                alpha = cv2.resize(alpha, (new_width, height), interpolation=_interp(img_w, img_h, new_width, height))
                alpha_out = alpha[:, left:left + width]
            return out, alpha_out
        else:
            new_height = int(width / aspect_ratio)
            resized = cv2.resize(raw, (width, new_height), interpolation=_interp(img_w, img_h, width, new_height))
            top = (new_height - height) // 2
            out = resized[top:top + height, :]
            alpha_out = None
            if alpha is not None:
                alpha = cv2.resize(alpha, (width, new_height), interpolation=_interp(img_w, img_h, width, new_height))
                alpha_out = alpha[top:top + height, :]
            return out, alpha_out

    elif mode == "pad":
        pad_color = _get_edge_color_cv2(raw)
        return _place_into_canvas_padded(raw, alpha, width, height, aspect_ratio, target_ratio, pad_color)

    return cv2.resize(raw, (width, height)), (
        cv2.resize(alpha, (width, height)) if alpha is not None else None
    )


def _place_into_canvas_padded(raw, alpha, width, height, aspect_ratio, target_ratio, pad_color):
    img_h, img_w = raw.shape[:2]

    if aspect_ratio > target_ratio:
        new_height = int(width / aspect_ratio)
        resized = cv2.resize(raw, (width, new_height), interpolation=_interp(img_w, img_h, width, new_height))
        padding = (height - new_height) // 2
        canvas = np.full((height, width, 3), pad_color, dtype=raw.dtype)
        canvas[padding:padding + new_height, :] = resized
        alpha_canvas = None
        if alpha is not None:
            alpha_resized = cv2.resize(alpha, (width, new_height), interpolation=_interp(img_w, img_h, width, new_height))
            alpha_canvas = np.zeros((height, width), dtype=alpha.dtype)
            alpha_canvas[padding:padding + new_height, :] = alpha_resized
        return canvas, alpha_canvas
    else:
        new_width = int(height * aspect_ratio)
        resized = cv2.resize(raw, (new_width, height), interpolation=_interp(img_w, img_h, new_width, height))
        padding = (width - new_width) // 2
        canvas = np.full((height, width, 3), pad_color, dtype=raw.dtype)
        canvas[:, padding:padding + new_width] = resized
        alpha_canvas = None
        if alpha is not None:
            alpha_resized = cv2.resize(alpha, (new_width, height), interpolation=_interp(img_w, img_h, new_width, height))
            alpha_canvas = np.zeros((height, width), dtype=alpha.dtype)
            alpha_canvas[:, padding:padding + new_width] = alpha_resized
        return canvas, alpha_canvas


def _get_edge_color_cv2(raw):
    h, w = raw.shape[:2]
    edges = np.concatenate([
        raw[0, :, :].reshape(-1, 3),
        raw[h - 1, :, :].reshape(-1, 3),
        raw[:, 0, :].reshape(-1, 3),
        raw[:, w - 1, :].reshape(-1, 3),
    ], axis=0)
    median = np.median(edges, axis=0).astype(np.uint8)
    return (int(median[0]), int(median[1]), int(median[2]))


def _decode_and_resize_pil(image_path, width, height, method):
    with Image.open(image_path) as i:
        i = ImageOps.exif_transpose(i)
        img_w, img_h = i.size

        if (width == -1 and height == -1) or (img_w == width and img_h == height):
            resized = i
        else:
            target_w = img_w if width == -1 else width
            target_h = img_h if height == -1 else height
            if method == "stretch":
                resized = i.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                resized = _resize_with_aspect_ratio_pil(i, target_w, target_h, method)

        rgb = np.array(resized.convert("RGB"), dtype=np.uint8)
        alpha = np.array(resized.getchannel('A'), dtype=np.uint8) if 'A' in resized.getbands() else None
        return rgb, alpha


def _resize_with_aspect_ratio_pil(img, width, height, mode):
    img_w, img_h = img.size
    aspect_ratio = img_w / img_h
    target_ratio = width / height

    if mode == "crop":
        if aspect_ratio > target_ratio:
            new_width = int(height * aspect_ratio)
            img = img.resize((new_width, height), Image.Resampling.LANCZOS)
            left = (new_width - width) // 2
            return img.crop((left, 0, left + width, height))
        else:
            new_height = int(width / aspect_ratio)
            img = img.resize((width, new_height), Image.Resampling.LANCZOS)
            top = (new_height - height) // 2
            return img.crop((0, top, width, top + height))

    elif mode == "pad":
        pad_color = _get_edge_color_pil(img)
        if aspect_ratio > target_ratio:
            new_height = int(width / aspect_ratio)
            img = img.resize((width, new_height), Image.Resampling.LANCZOS)
            padding = (height - new_height) // 2
            padded = Image.new('RGBA', (width, height), pad_color)
            padded.paste(img, (0, padding))
            return padded
        else:
            new_width = int(height * aspect_ratio)
            img = img.resize((new_width, height), Image.Resampling.LANCZOS)
            padding = (width - new_width) // 2
            padded = Image.new('RGBA', (width, height), pad_color)
            padded.paste(img, (padding, 0))
            return padded
    return img.resize((width, height), Image.Resampling.LANCZOS)


def _get_edge_color_pil(img):
    from PIL import ImageStat
    width, height = img.size
    img = img.convert('RGBA')
    top = img.crop((0, 0, width, 1))
    bottom = img.crop((0, height - 1, width, height))
    left = img.crop((0, 0, 1, height))
    right = img.crop((width - 1, 0, width, height))
    edges = Image.new('RGBA', (width * 2 + height * 2, 1))
    edges.paste(top, (0, 0))
    edges.paste(bottom, (width, 0))
    edges.paste(left.resize((height, 1)), (width * 2, 0))
    edges.paste(right.resize((height, 1)), (width * 2 + height, 0))
    stat = ImageStat.Stat(edges)
    return tuple(map(int, stat.median))


def _process_one(image_path, width, height, method):
    if HAS_CV2:
        rgb, alpha = _decode_and_resize_cv2(image_path, width, height, method)
    else:
        rgb, alpha = _decode_and_resize_pil(image_path, width, height, method)
    return image_path, rgb, alpha


class LoadImagesBlaze:
    directory_hashes = {}

    @classmethod
    def IS_CHANGED(cls, directory_path, **kwargs):
        directory_path = _resolve_directory_path(directory_path)
        if not directory_path or not os.path.isdir(directory_path):
            return float("NaN")

        subfolder_depth = kwargs.get('subfolder_depth', 0)
        image_paths = _collect_image_files(directory_path, subfolder_depth)

        combined_hash = hashlib.md5()
        combined_hash.update(directory_path.encode('utf-8'))
        combined_hash.update(str(len(image_paths)).encode('utf-8'))

        for path in image_paths:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            combined_hash.update(f"{path}:{mtime}".encode('utf-8'))

        current_hash = combined_hash.hexdigest()
        old_hash = cls.directory_hashes.get(directory_path)
        cls.directory_hashes[directory_path] = current_hash

        if old_hash == current_hash:
            return old_hash
        return current_hash

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
                "subfolder_depth": ("INT", {"default": 0, "min": -1, "step": 1}),
                "width": ("INT", {"default": 1024, "min": -1, "step": 1}),
                "height": ("INT", {"default": 1024, "min": -1, "step": 1}),
                "method": (["crop", "pad", "stretch", ],),
            },
            "optional": {
                "cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT",)
    RETURN_NAMES = ("image", "mask", "count",)
    FUNCTION = "load_images"
    CATEGORY = "Rebalance-Pack/utilities"

    def load_images(self, directory_path, width, height, cap, start_index, method, subfolder_depth=0):
        directory_path = _resolve_directory_path(directory_path)
        if not directory_path or not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' cannot be found.")

        dir_files = _collect_image_files(directory_path, subfolder_depth)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory_path}'.")

        dir_files = dir_files[start_index:]

        if cap > 0:
            dir_files = dir_files[:cap]

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files to load after start_index/cap in '{directory_path}'.")

        first_path = dir_files[0]
        _, first_rgb, first_alpha = _process_one(first_path, width, height, method)
        out_h, out_w = first_rgb.shape[:2]

        n = len(dir_files)
        image_tensor = torch.empty((n, out_h, out_w, 3), dtype=torch.float32)
        mask_tensor = torch.zeros((n, out_h, out_w), dtype=torch.float32)

        image_tensor[0] = torch.from_numpy(first_rgb.astype(np.float32) / 255.0)
        if first_alpha is not None:
            mask_tensor[0] = 1.0 - torch.from_numpy(first_alpha.astype(np.float32) / 255.0)

        pbar = ProgressBar(n)
        pbar.update_absolute(1)

        max_workers = max(1, min(8, (os.cpu_count() or 4)))
        if n > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_process_one, path, width, height, method)
                    for path in dir_files[1:]
                ]
                for idx, future in enumerate(futures, start=1):
                    path, rgb, alpha = future.result()
                    image_tensor[idx] = torch.from_numpy(rgb.astype(np.float32) / 255.0)
                    if alpha is not None:
                        mask_tensor[idx] = 1.0 - torch.from_numpy(alpha.astype(np.float32) / 255.0)
                    pbar.update_absolute(idx + 1)

        return (image_tensor, mask_tensor, n)


class ListDisplay:
    """filename widget"""

    @classmethod
    def IS_CHANGED(cls, directory_path, ext, filename, **kwargs):
        directory_path = _resolve_directory_path(directory_path)
        if not directory_path or not os.path.isdir(directory_path):
            return float("NaN")
        full = os.path.join(directory_path, filename)
        try:
            return os.path.getmtime(full)
        except OSError:
            return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
                "ext": ("STRING", {"default": "safetensors"}),
                "filename": ([],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SELECT",)
    FUNCTION = "load_file"
    CATEGORY = "Rebalance-Pack/utilities"

    @classmethod
    def VALIDATE_INPUTS(cls, directory_path, ext, filename, **kwargs):

        return True

    def load_file(self, directory_path, ext, filename):
        directory_path = _resolve_directory_path(directory_path)
        if not directory_path or not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' cannot be found.")
        full = os.path.join(directory_path, filename)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"File '{filename}' not found in '{directory_path}'.")
        return (filename,)


NODE_CLASS_MAPPINGS = {
    "LoadImagesBlaze": LoadImagesBlaze,
    "ListDisplay": ListDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImagesBlaze": "Load Images (Blaze)",
    "ListDisplay": "List Display",
}

__all__ = [
    "LoadImagesBlaze",
    "ListDisplay",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
