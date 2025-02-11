# [Reimplement MViTv2](https://arxiv.org/abs/2112.01526)

**You can simply run the demo here: https://drive.google.com/drive/folders/1ix7_-aAQ-4BHLx6p4_M2A5VcYZw1C1FO?usp=sharing (NYU email required) **

Xu's PyTorch implementation of **MViTv2**, from the following paper:

[MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526). CVPR 2022.\
Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer*

---

MViT is a multiscale transformer which serves as a general vision backbone for different visual recognition tasks.

## Some Major Changes: 

1. Inflated 2D architecture to 3D.
2. Support 3D model initialization using 2D weights (Positional embeddings were not included in the initialization if initialized with 2D weights.)
3. More demos and experiments enabling video classification.



## Data Preparation

You can download the ImageNet-1K classification dataset or use any other dataset and structure the data as follows:
```
/path/to/your_dataset/
  train/
    n0/
      imgfoo1.jpeg
    n1/
      imgfoo2.jpeg
  val/
    n0/
      imgfoo3.jpeg
    n1/
      imgfoo4.jpeg
  test/
    n0/
      imgfoo5.jpeg
    n1/
      imgfoo6.jpeg
```

Do note that some data processing methods were designed for ImageNet data.



## Installation

Quick demo can be done by running the codes in the demo folder. The easiest way to run the demo's is to load the whole repository (pretty small size) into a google drive and place it under the directory /content/drive/MyDrive/Reimplementation/MViT/mvit/ and run the demo files.

Alternatively, you can simply run the demo that's already in my google drive: https://drive.google.com/drive/folders/1ix7_-aAQ-4BHLx6p4_M2A5VcYZw1C1FO?usp=sharing (NYU email required) 

For the complete setup, you can check [INSTALL.md](INSTALL.md) for installation instructions.

## Training

Here we can train a standard MViTv2 model from scratch by:
```
python tools/main.py \
  --cfg configs/MViTv2_T.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 256 \
```

## Finetuning

Here we can finetune a standard MViTv2 model using pretrained weight by:
```
python tools/main.py \
  --cfg configs/MViTv2_T.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 256 \
```

## Evaluation

To evaluate a pretrained MViT model:
```
python tools/main.py \
  --cfg configs/test/MViTv2_T_test.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TEST.BATCH_SIZE 256 \
```

# Pre-trained Models
### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | 1k model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| MViTv2-T | 224x224 | 82.3 | 24M | 4.7G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_T_in1k.pyth) |
| MViTv2-S | 224x224 | 83.6 | 35M | 7.0G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_S_in1k.pyth) |
| MViTv2-B | 224x224 | 84.4 | 52M | 10.2G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in1k.pyth) |
| MViTv2-L | 224x224 | 85.3 | 218M | 42.1G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_L_in1k.pyth) |

### ImageNet-21K trained models

| name | resolution |acc@1 | #params | FLOPs | 21k model | 1k model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| MViTv2-B | 224x224 | - | 52M | 10.2G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in21k.pyth) | - |
| MViTv2-L | 224x224 | 87.5 | 218M | 42.1G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_L_in21k.pyth) | - |
| MViTv2-H | 224x224 | 88.0 | 667M | 120.6G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_H_in21k.pyth) | - |

## Acknowledgement
This repository is built based on the facebookresearch implementation of [mvit](https://github.com/facebookresearch/mvit).

## License
MViT is released under the [Apache 2.0 license](LICENSE).


```
