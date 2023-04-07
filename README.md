# MHIM-MIL
Training codes and logs of ICCV 2023 submission 6302

## Training
```shell
python3 main.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=3 --teacher_init=./modules/init_ckp/c16_3fold_init_abmil_seed2021 --title=abmil_101_mr50h1-0r50_is --baseline=attn --num_workers=0 --cl_alpha=0.1 --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --mrh_sche --init_stu_type=fc --mask_ratio=0.5 --mask_ratio_l=0. --seed=2021
```

```shell
python3 main.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --cv_fold=3 --teacher_init=./modules/init_ckp/c16_3fold_init_transmil_seed2021 --mask_ratio_h=0.03 --mask_ratio_hr=0.5 --mrh_sche --title=transmil_101_sml80h3-0r50_mmcos_is --mask_ratio=0. --mask_ratio_l=0.8 --cl_alpha=0.1 --mm_sche --init_stu_type=fc --attn_layer=0 --seed=2021
```