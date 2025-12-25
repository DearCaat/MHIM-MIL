# MHIM-MIL
Official repo of **Multiple Instance Learning Framework with Masked Hard Instance Mining for Whole Slide Image Classification**, ICCV 2023. [[arXiv]](https://arxiv.org/abs/2307.15254) [[Slide]](doc/iccv_oral.pdf) [[Oral]](https://www.youtube.com/watch?v=ePZTtX0_tRQ)

Official repo of **Multiple Instance Learning Framework with Masked Hard Instance Mining for Gigapixel Histopathology Image Analysis**, IJCV 2025. [[arXiv]](https://arxiv.org/abs/2509.11526) [[Huggingface Dataset]](https://huggingface.co/datasets/Dearcat/CPathPatchFeature)

![](doc/vis_v2.png)

## News
- 2025-12: **MHIM-v2** is accepted by **IJCV 2025**. Released code of **MHIM-v2**.
- 2025-09: Released **MHIM-v2: a more concise and effective method, with stronger and broader generalizability, and enhanced interpretability.** [[arxiv]](https://arxiv.org/abs/2509.11526)
- 2023-07: MHIM-MIL is accepted by **ICCV 2023**, and selected as an **oral presentation**.[[arXiv]](https://arxiv.org/abs/2307.15254) [[Slide]](doc/iccv_oral.pdf) [[Oral]](https://www.youtube.com/watch?v=ePZTtX0_tRQ)

## Overview

| Branch Name | Link |
|-------------|------|
| 2.0 Version Branch | [master](https://github.com/DearCaat/MHIM-MIL/tree/master) (latest) |
| 1.0 Version Branch | [1.0 Version](https://github.com/DearCaat/MHIM-MIL/tree/v1) |

## Usage
### 1. Environment Preparation

We recommend using Docker for a reproducible environment. Alternatively, you can install dependencies via PyPI.

##### Option 1: Docker (Recommended)

1. Download the Docker Image from [Google Drive](https://drive.google.com/file/d/1JYdbYewxx2JFXz6SlE9srojLvOuh2xFD/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1vA6raEdupp-D1ddPw3PITg?pwd=2025) (Password: 2025)
2. Load the Docker image:
    ```bash
    docker load -i XXX.tar
    ```
    (Replace `XXX.tar` with the downloaded file name.)
3. Run the Docker container:
    ```bash
    docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864\
                -v /path/to/your_code:/workspace/code \
                -v /path/to/your_data:/workspace/dataset \
                -v /path/to/your_output:/workspace/output \
                --name mhim \
                --runtime=nvidia \
                -e NVIDIA_VISIBLE_DEVICES=all \
                -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
                -d mhim:latest /bin/bash
    ```

##### Option 2: PyPI

1.  Create a new Python environment:
    ```bash
    conda create -n mhim python=3.9
    conda activate mhim
    ```
2.  Install the required packages. 
    A complete list of requirements can be found in [requirements.txt](./requirements.txt).
    ```bash
    pip install -r requirements.txt
    ```

### 2. Data Preprocessing

##### Download Preprocessed Feature

We provide preprocessed patch features for all datasets. You can download them from:
[Hugginface](https://huggingface.co/datasets/Dearcat/CPathPatchFeature), [ModelScope](https://www.modelscope.cn/datasets/HENGFANG/CPathPatchFeature), [Baidu Netdisk](https://pan.baidu.com/s/1OuiIP3sB68IGZeId4s4K7Q?pwd=ujtq) (Password: ujtq)

##### Preprocess Raw Data

If you have raw Whole-Slide Image (WSI) data, you can preprocess it as follows:

1. **Patching** (Following [CLAM](https://github.com/mahmoodlab/CLAM/)):

    ```bash
    python CLAM/create_patches_fp.py --source YOUR_DATA_DIRECTORY \
                                     --save_dir YOUR_RESULTS_DIRECTORY \
                                     --patch_size 256 \
                                     --step_size 256 \
                                     --patch_level 0 \
                                     --preset YOUR_PRESET_FILE \
                                     --seg \
                                     --patch
    ```
    *Replace placeholders like `YOUR_DATA_DIRECTORY` with your actual paths and parameters. Preset files are officially provided by CLAM.*

2. **Feature Extraction** (Modify on the official [CLAM](https://github.com/mahmoodlab/CLAM/) repository to support the encoders of ResNet-50, [CHIEF](https://github.com/hms-dbmi/CHIEF), [UNI](https://github.com/mahmoodlab/UNI) and [GIGAP](https://github.com/prov-gigapath/prov-gigapath)):
   
    > You can also extract all the required features following the process of [TRIDENT](https://github.com/mahmoodlab/TRIDENT).

    ```bash
    CUDA_VISIBLE_DEVICES=$TARGET_GPUs python CLAM/extract_features_fp.py \
                                        --data_h5_dir DIR_TO_COORDS \
                                        --data_slide_dir DATA_DIRECTORY \
                                        --csv_path CSV_FILE_NAME \
                                        --feat_dir FEATURES_DIRECTORY \
                                        --slide_ext .svs \
                                        --model_name resnet50_trunc/uni_v1/chief/gigap
    ```

## Training
> **⚠️ Note:** We've significantly refactored the codebase! If you spot any issues, please let us know. You can still use the old version in the v1 branch.

You can use [wandb](https://wandb.ai/site) to track the training process, add `--wandb` to the command line.

For different patch encoders, you should use different input dimensions, use `--input_dim` to specify the input dimension (ResNet-50: 1024, PLIP: 512, UNI-v1: 1024, and so on).

### Prepare Your Own Initiation Weight
`model=mhim_pure baseline=[attn,selfattn,dsmil]`
```shell
# baselines on Call dataset (diagnosis)
python3 main.py --project=$PROJECT_NAME --datasets=call --dataset_root=$DATASET_PATH --csv_path=$LABEL_CSV_PATH --output_path=$OUTPUT_PATH -c=./config/feat_cls.yaml --title=call_$BASELINE --model=pure --model=mhim_pure --baseline=$BASELINE
# baselines on TCGA-NSCLC dataset (sub-type)
python3 main.py --project=$PROJECT_NAME --datasets=nsclc --dataset_root=$DATASET_PATH --csv_path=$LABEL_CSV_PATH --output_path=$OUTPUT_PATH -c=./config/feat_cls.yaml --title=NSCLC_$BASELINE --model=pure --model=mhim_pure --baseline=$BASELINE
# baselines on TCGA-BRCA dataset (sub-type)
python3 main.py --project=$PROJECT_NAME --datasets=brca --dataset_root=$DATASET_PATH --csv_path=$LABEL_CSV_PATH --output_path=$OUTPUT_PATH -c=./config/feat_cls.yaml --title=BRCA_$BASELINE --model=pure --model=mhim_pure --baseline=$BASELINE
# baselines on TCGA-BLCA dataset (survival)
python3 main.py --project=$PROJECT_NAME --datasets=surv_blca --dataset_root=$DATASET_PATH --csv_path=$LABEL_CSV_PATH --output_path=$OUTPUT_PATH -c=./config/feat_surv.yaml --title=BLCA_$BASELINE --model=pure --model=mhim_pure --baseline=$BASELINE
# baselines on TCGA-LUAD dataset (survival)
python3 main.py --project=$PROJECT_NAME --datasets=surv_luad --dataset_root=$DATASET_PATH --csv_path=$LABEL_CSV_PATH --output_path=$OUTPUT_PATH -c=./config/feat_surv.yaml --title=LUAD_$BASELINE --model=pure --model=mhim_pure --baseline=$BASELINE
# baselines on TCGA-LUSC dataset (survival)
python3 main.py --project=$PROJECT_NAME --datasets=surv_lusc --dataset_root=$DATASET_PATH --csv_path=$LABEL_CSV_PATH --output_path=$OUTPUT_PATH -c=./config/feat_surv.yaml --title=LUSC_$BASELINE --model=pure --model=mhim_pure --baseline=$BASELINE
```

### Training MHIM-MIL

We recommend performing a grid search for the following hyperparameters to achieve optimal performance:
- `mask_ratio_h`: {0.01, 0.03, 0.05}
- `merge_ratio`: {0.8, 0.9}
- `merge_k`: {1, 5, 10}

```shell
# Grid search on hyperparameters
# Replace $DATASET_NAME with one of [call, nsclc, brca, surv_blca, surv_luad, surv_lusc]
# Replace $CONFIG_FILE with ./config/feat_cls.yaml (for diagnosis and sub-type) or ./config/feat_surv.yaml (for survival)

python3 main.py --project=$PROJECT_NAME \
    --datasets=$DATASET_NAME \
    --dataset_root=$DATASET_PATH \
    --csv_path=$LABEL_CSV_PATH \
    --output_path=$OUTPUT_PATH \
    --teacher_init=$TEACHER_WEIGHT_PATH \
    -c=$CONFIG_FILE \
    --title=${DATASET_NAME}_mhim_${BASELINE}_${OTHER_HYPERPARAMETERS} \
    --model=mhim \
    --baseline=$BASELINE \
    --attn2score \
    --merge_enable \
    --merge_mm=0.9999 \
    --aux_alpha=0.5 \
    --mask_ratio_h=$MASK_RATIO_H \
    --merge_ratio=$MERGE_RATIO \
    --merge_k=$MERGE_K
```


## Citing MHIM-MIL
If you find this work useful in your research, please consider citing:
```
@InProceedings{Tang_2023_ICCV,
    author    = {Tang, Wenhao and Huang, Sheng and Zhang, Xiaoxian and Zhou, Fengtao and Zhang, Yi and Liu, Bo},
    title     = {Multiple Instance Learning Framework with Masked Hard Instance Mining for Whole Slide Image Classification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4078-4087}
}

@misc{tang2025multipleinstancelearningframework,
      title={Multiple Instance Learning Framework with Masked Hard Instance Mining for Gigapixel Histopathology Image Analysis}, 
      author={Wenhao Tang and Sheng Huang and Heng Fang and Fengtao Zhou and Bo Liu and Qingshan Liu},
      year={2025},
      eprint={2509.11526},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.11526}, 
}
```
