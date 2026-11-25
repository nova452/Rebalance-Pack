# 24/11 Latest Update
* New sampling nodes
* Block-weighted LoRA loaders for MOSS-Image and MiniMax H3 / H3 Image
* Omni Node moved to another repository. [OmniNode](https://github.com/nova452/OmniNode)
* New sampler solution for Moss Image's blur issue:
<img width="2060" height="1096" alt="moss_rebalance_comparison" src="https://github.com/user-attachments/assets/7fadb9aa-39ee-4d3f-b7e2-825b233fec67" />

[workflow.json](https://github.com/nova452/Rebalance-Pack/blob/main/workflows/moss_image_rebalance_sigmas.json)

# 10/08
* New core utility nodes
* The Omni Node family now lives in its own pack, ComfyUI-Omni-Node

# 12/07
* Supports multi character references for stronger in-context posing
* New utility nodes added

<img width="2838" height="1387" alt="Screenshot (586)" src="https://github.com/user-attachments/assets/fbc29244-021b-4c7e-90f0-f0138342e15d" />

[workflow.json](https://github.com/nova452/Rebalance-Pack/blob/main/workflows/Krea%202%20Edit%20(Full%20Context).json)

# 29/06
* Clear logic allowing for more complex edits
* Now supports negative prompting for stripping undesired effects

<img width="1623" height="1227" alt="Screenshot (563)" src="https://github.com/user-attachments/assets/35ad346b-b4f7-4a83-8181-2891357d5dd0" />

[workflow.json](https://github.com/nova452/Rebalance-Pack/blob/main/workflows/krea2_image_edit_workflow%20(3).json)

# Image Editing with Krea 2 and Ideogram 4 Models
<img width="2616" height="2463" alt="showcase (1)" src="https://github.com/user-attachments/assets/aae61b58-a0b2-49c3-8718-ec32b0ec180e" />

# Rebalance Pack
A set of evolving nodes that streamline the best use practices in ComfyUI.


foundational: Input, Any, Switch, Load Images, Load Images (Blaze), Load Image Newest, Load Image Full, Save Images, Color Image, Concatenate, String (Inline), String (Inline 5), String to List, List String Index, List Display, Float Exact, Float 1.00, Float 10.00

utility: Image Resolution Cap, Image Aspect Ratio Crop, Mask Aspect Ratio BBox, Uncrop Image, Uncrop Mask, Border Mask & Crop Detector, Mask Opacity

conditioning: Conditioning Merge, Conditioning Merge (Multi), Conditioning Merge (List), Conditioning Freeze, Conditioning Unfreeze, Qwen VL List Encode Rebalance, Qwen VL Image Embeds Encode, Qwen VL Image Embeds Merge

sampling: Rebalance Guider, Rebalance Guider B, Rebalance CFG Custom, Step Rebalance, Sigmas Rebalance, Sampler Rebalance

lora: Load LoRA (Block), Load LoRA Krea2 (Block), Load LoRA Moss Image (Block), Load LoRA H3 (Block), Load LoRA H3 Image (Block), Load LoRA Folder (Block), Load LoRA Folder Moss Image (Block), Load LoRA Folder H3 (Block), Load LoRA Folder H3 Image (Block)

model specific: Conditioning Krea2 Rebalance, Krea 2 Image Edit Rebalance, Krea 2 Image Edit Rebalance Steering, Krea 2 Encode Rebalance, Conditioning Ideogram4 Rebalance, Ideogram 4 Image Edit Rebalance, Ideogram 4 Encode Rebalance

```bash
git clone https://github.com/nova452/Rebalance-Pack.git
```
