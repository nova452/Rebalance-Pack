"""Mask toolkit (Rebalance package)."""

import torch


class UncropImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "original_mask": ("MASK",),
                "cropped_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("uncropped_image",)
    FUNCTION = "uncrop"
    CATEGORY = "Rebalance-Pack/image"

    def uncrop(self, original_image, original_mask, cropped_image):

        if original_mask.dim() == 2:
            original_mask = original_mask.unsqueeze(0)
        elif original_mask.dim() == 4:
            if original_mask.shape[1] == 1:
                original_mask = original_mask.squeeze(1)
            elif original_mask.shape[-1] == 1:
                original_mask = original_mask.squeeze(-1)

        out_image = original_image.clone()

        coords = torch.nonzero(original_mask > 0)
        if len(coords) == 0:
            return (out_image,)

        y_min = int(coords[:, 1].min().item())
        x_min = int(coords[:, 2].min().item())

        start_y = y_min
        start_x = x_min

        b_orig, h_orig, w_orig, c_orig = out_image.shape
        b_crop, h_crop, w_crop, c_crop = cropped_image.shape

        if b_orig != b_crop:
            if b_crop == 1:
                cropped_image = cropped_image.expand(b_orig, -1, -1, -1)
            elif b_orig == 1:
                out_image = out_image.expand(b_crop, -1, -1, -1).clone()
            else:
                min_b = min(b_orig, b_crop)
                out_image = out_image[:min_b]
                cropped_image = cropped_image[:min_b]

        c_min = min(c_orig, c_crop)

        paste_y_min = start_y
        paste_y_max = min(start_y + h_crop, h_orig)
        paste_x_min = start_x
        paste_x_max = min(start_x + w_crop, w_orig)

        use_h = paste_y_max - paste_y_min
        use_w = paste_x_max - paste_x_min

        if use_h > 0 and use_w > 0:
            out_image[:, paste_y_min:paste_y_max, paste_x_min:paste_x_max, :c_min] = \
                cropped_image[:, :use_h, :use_w, :c_min]

        return (out_image,)


class UncropMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_mask": ("MASK",),
                "cropped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("uncropped_mask",)
    FUNCTION = "uncrop"
    CATEGORY = "Rebalance-Pack/mask"

    def uncrop(self, original_mask, cropped_mask):
        if original_mask.dim() == 2:
            original_mask = original_mask.unsqueeze(0)
        elif original_mask.dim() == 4:
            original_mask = original_mask.squeeze(1)

        if cropped_mask.dim() == 2:
            cropped_mask = cropped_mask.unsqueeze(0)
        elif cropped_mask.dim() == 4:
            cropped_mask = cropped_mask.squeeze(1)

        out_mask = torch.zeros_like(original_mask)

        coords = torch.nonzero(original_mask > 0)
        if len(coords) == 0:
            return (out_mask,)

        y_min = int(coords[:, 1].min().item())
        x_min = int(coords[:, 2].min().item())

        start_y = y_min
        start_x = x_min

        b_crop, h_crop, w_crop = cropped_mask.shape
        b_orig, h_orig, w_orig = original_mask.shape

        if b_orig != b_crop:
            if b_crop == 1:
                cropped_mask = cropped_mask.expand(b_orig, -1, -1)
            elif b_orig == 1:
                out_mask = out_mask.expand(b_crop, -1, -1).clone()
            else:
                min_b = min(b_orig, b_crop)
                out_mask = out_mask[:min_b]
                cropped_mask = cropped_mask[:min_b]

        paste_y_min = start_y
        paste_y_max = min(start_y + h_crop, out_mask.shape[1])
        paste_x_min = start_x
        paste_x_max = min(start_x + w_crop, out_mask.shape[2])

        use_h = paste_y_max - paste_y_min
        use_w = paste_x_max - paste_x_min

        if use_h > 0 and use_w > 0:
            out_mask[:, paste_y_min:paste_y_max, paste_x_min:paste_x_max] = cropped_mask[:, :use_h, :use_w]

        return (out_mask,)


class BorderMaskDetector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_top": ("BOOLEAN", {"default": True}),
                "detect_bottom": ("BOOLEAN", {"default": True}),
                "detect_left": ("BOOLEAN", {"default": True}),
                "detect_right": ("BOOLEAN", {"default": True}),
                "threshold": ("FLOAT", {"default": 0.80, "min": 0.01, "max": 1.0, "step": 0.01}),
                "color_tolerance": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.001}),
                "max_scan_depth": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "cropped_image")
    FUNCTION = "detect_and_mask"
    CATEGORY = "Rebalance-Pack/image"

    def detect_and_mask(self, image, detect_top, detect_bottom, detect_left, detect_right, threshold, color_tolerance, max_scan_depth):
        B, H, W, C = image.shape
        device = image.device

        masks = torch.zeros((B, H, W), dtype=torch.float32, device=device)

        top_borders = torch.zeros(B, dtype=torch.long, device=device) - 1
        bottom_borders = torch.zeros(B, dtype=torch.long, device=device) + H
        left_borders = torch.zeros(B, dtype=torch.long, device=device) - 1
        right_borders = torch.zeros(B, dtype=torch.long, device=device) + W

        img_rgb = image[..., :3]

        for b in range(B):
            img = img_rgb[b]

            top_border = -1
            bottom_border = H
            left_border = -1
            right_border = W

            max_y_scan = int(H * max_scan_depth)
            max_x_scan = int(W * max_scan_depth)

            if detect_top:
                for y in range(max_y_scan):
                    row = img[y, :, :]
                    median = torch.median(row, dim=0).values
                    dist = torch.max(torch.abs(row - median), dim=-1).values
                    match_count = torch.sum(dist <= color_tolerance).item()
                    if match_count >= threshold * W:
                        top_border = y
                    else:
                        break

            if detect_bottom:
                for y in range(H - 1, H - 1 - max_y_scan, -1):
                    row = img[y, :, :]
                    median = torch.median(row, dim=0).values
                    dist = torch.max(torch.abs(row - median), dim=-1).values
                    match_count = torch.sum(dist <= color_tolerance).item()
                    if match_count >= threshold * W:
                        bottom_border = y
                    else:
                        break

            y_start = top_border + 1
            y_end = bottom_border
            active_h = y_end - y_start

            if active_h > 0:
                if detect_left:
                    for x in range(max_x_scan):
                        col = img[y_start:y_end, x, :]
                        median = torch.median(col, dim=0).values
                        dist = torch.max(torch.abs(col - median), dim=-1).values
                        match_count = torch.sum(dist <= color_tolerance).item()
                        if match_count >= threshold * active_h:
                            left_border = x
                        else:
                            break

                if detect_right:
                    for x in range(W - 1, W - 1 - max_x_scan, -1):
                        col = img[y_start:y_end, x, :]
                        median = torch.median(col, dim=0).values
                        dist = torch.max(torch.abs(col - median), dim=-1).values
                        match_count = torch.sum(dist <= color_tolerance).item()
                        if match_count >= threshold * active_h:
                            right_border = x
                        else:
                            break

            top_borders[b] = top_border
            bottom_borders[b] = bottom_border
            left_borders[b] = left_border
            right_borders[b] = right_border

            if top_border >= 0:
                masks[b, 0 : top_border + 1, :] = 1.0
            if bottom_border < H:
                masks[b, bottom_border : H, :] = 1.0
            if left_border >= 0:
                masks[b, :, 0 : left_border + 1] = 1.0
            if right_border < W:
                masks[b, :, right_border : W] = 1.0

        crop_top = int(torch.max(top_borders).item()) + 1
        crop_bottom = int(torch.min(bottom_borders).item())
        crop_left = int(torch.max(left_borders).item()) + 1
        crop_right = int(torch.min(right_borders).item())

        if crop_top >= crop_bottom or crop_left >= crop_right:
            cropped_image = image.clone()
        else:
            cropped_image = image[:, crop_top:crop_bottom, crop_left:crop_right, :]

        return (masks, cropped_image)


class MaskOpacity:
    """
    opacity = 1.0 keeps the mask unchanged (100% opacity).
    opacity = 0.0 makes the mask fully transparent (0% opacity).
    Values in between linearly scale the intensity.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "adjust_opacity"
    CATEGORY = "Rebalance-Pack/mask"

    def adjust_opacity(self, mask, opacity):
        if mask.dim() == 4:
            if mask.shape[1] == 1:
                mask = mask.squeeze(1)
            elif mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0)

        out = mask * float(opacity)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "UncropImage": UncropImage,
    "UncropMask": UncropMask,
    "BorderMaskDetector": BorderMaskDetector,
    "MaskOpacity": MaskOpacity,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UncropImage": "Uncrop Image",
    "UncropMask": "Uncrop Mask",
    "BorderMaskDetector": "Border Mask & Crop Detector",
    "MaskOpacity": "Mask Opacity",
}
