
# 26/09 Latest Update: Various New Functions
* Streamlined Qwen VL encoder and many QoL nodes
* New conditioning merge methods
* Minor needed fixes for the encode nodes

# 10/08 Prompt-based Omni Node in ComfyUI
* Omni Node family extended
* New core utility nodes
<img width="2254" height="1176" alt="Screenshot (10)" src="https://github.com/user-attachments/assets/e01edd61-c064-42f2-81e7-a6463d8447f3" />


# 29/07 OmniNode
A sole ComfyUI node that lets you paste in the body of a
node class, rebuilding its own inputs/outputs to mirror the source.
<img width="2889" height="1521" alt="627795071-43d362c4-cfa6-42b7-b2a6-3c29ad8a769a" src="https://github.com/user-attachments/assets/2158258c-6b47-491e-b859-886615afb048" />

# 12/07
* Supports multi character references for stronger in-context posing
* New utility nodes added

<img width="2838" height="1387" alt="Screenshot (586)" src="https://github.com/user-attachments/assets/fbc29244-021b-4c7e-90f0-f0138342e15d" />

[workflow.json](https://github.com/nova452/Rebalance-Pack/blob/main/workflows/Krea%202%20Edit%20(Full%20Context).json)

# 29/06
* Clear logic allowing for more complex edits
* Now supports negative prompting for stripping undesired effects

<img width="1623" height="1227" alt="Screenshot (563)" src="https://github.com/user-attachments/assets/35ad346b-b4f7-4a83-8181-2891357d5dd0" />

[workflow.json](https://github.com/Rebalance-Pack/blob/main/workflows/krea2_image_edit_workflow%20(3).json)

# Image Editing with Krea 2 and Ideogram 4 Models
<img width="2616" height="2463" alt="showcase (1)" src="https://github.com/user-attachments/assets/aae61b58-a0b2-49c3-8718-ec32b0ec180e" />

# Rebalance Pack
A set of evolving nodes that streamline the best use practices in ComfyUI.


foundational: Omni Node, Input, Load Images, Load Image Newest, Load Image Full, Color Image, Concatenate, String (Inline), String to List, List String Index, Float Exact, Float 1.00, Float 10.00, Any, Switch

utility: Image Resolution Cap, Image Aspect Ratio Crop, Mask Aspect Ratio Crop, Mask Uncrop, Image Uncrop, Border Mask, Mask Opacity,

conditioning: Conditioning Merge, Conditioning Merge Multi, Qwen VL List Encode Rebalance, Conditioning Freeze, Conditioning Unfreeze

model specific: Load LoRA (Block), Conditioning Krea 2 Rebalance, Krea 2 Edit Rebalance, Krea 2 Encode Rebalance, Conditioning Ideogram 4 Rebalance, Ideogram 4 Edit Rebalance, Ideogram 4 Encode Rebalance

```bash
git clone https://github.com/nova452/Rebalance-Pack.git
```
