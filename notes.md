# models

## meta 
 VITs - currently have three models from them ViTs, these are MAE with losses available
    VITs - mae and msn exist
 DEITS - then they also have a school of deit s , these are img classification models, sooo again, easy to eval
 DINOv2 (maybe exclude the giant) and for feature extraction too and img classification
 I-jepa is there too
## google 
 VIT - they also have their own school of ViTs
 https://github.com/google-research/vision_transformer?tab=readme-ov-file#available-vit-models

 LIT - yet to check LiT

 SIgLIP - zero shot image classification ig (check krta hun kya hai)

 MLP mixer (idk most probably not)
## tnt
 TNT - https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch
 classification (apparently)

## microsoft
 swin - image classification 
 Beit 

## nvdia

NVILA -  uses SiglipVisionModel 
VILA - uses SiglipVisionModel, and InternVisionModel
cosmosnemotron - uses Siglip-400M

segmentation models
SegFormer-B0: 3.7M parameters
SegFormer-B1: 13.7M parameters
SegFormer-B2: 27.5M parameters
SegFormer-B3: 47.3M parameters
SegFormer-B4: 64.1M parameters
SegFormer-B5: 84.7M parameters

## EVA 2
can do .. idk i am tired of listing models now
https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth

## clip
https://huggingface.co/openai/clip-vit-large-patch14-336

## IBM
 crossvit (classification)
------------------------------
image gen

##nvidia
 sana - text to image Linear diffusion transformer .. lol wtf idk what to do of that

| **Model**                      | **Architecture**                                                                 |
|--------------------------------|---------------------------------------------------------------------------------|
| `facebook/vit-mae-base`        | Vision Transformer (ViT)                                                        |
| `microsoft/beit-base-patch16`  | Vision Transformer (ViT)                                                        |
| `google/siglip-base-patch16`   | Vision Transformer (ViT)                                                        |
| `facebook/deit-small-distilled`| Vision Transformer (ViT)                                                        |
| `openai/clip-vit-base-patch32` | Vision Transformer (ViT)                                                        |
| `facebook/dinov2-small`        | Vision Transformer (ViT)                                                        |
| `facebook/vit-msn`             | Vision Transformer (ViT)                                                        |
| `facebook/levit`               | Vision Transformer with convolutional elements                                  |

| **Model**                      | **Architecture**                                                                 |
|--------------------------------|---------------------------------------------------------------------------------|
| `microsoft/swin-base-simmim`   | Swin Transformer (hierarchical, shifted windows)                                 |
| `ibm/crossvit`                 | Hybrid Vision Transformer with multi-scale tokens                                |
| `nvidia/segformer`             | Transformer with hierarchical encoder and lightweight MLP decoder                |
| `tnt`                          | Nested Transformer: local patch transformers within global patch transformers    |
| `eva-2`                        | Enhanced Vision Transformer with large-scale pretraining improvements            |
| `facebook/ijepa`               | Image Joint-Embedding Predictive Architecture                                    |
| `google/lit`                   | Locked Image-Text Tuning                                                         |




for cka matrices should be n X neural rep size
n = number of examples

n07745940- strawberry 

n07920052 - espresso

n09472597 - volcano

n03991062 - pot, flowerpot

n01484850 - shark

n02504013 - indian elephan, elephas maximus


# google vit, facebook deit

$$
S_{ij} = \text{NPIB}(E_i, E_j) =
\frac{\sum\limits_{x \in E_i} \sum\limits_{y \in E_j} p(x, y) \log \frac{p(x,y)}{p(x)p(y)}}
{\sqrt{\left(\sum\limits_{x \in E_i} p(x) \log p(x) \right) \cdot \left(\sum\limits_{y \in E_j} p(y) \log p(y) \right)}}
$$
