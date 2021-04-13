#Confident_Learning_for_Noisy-labeled_Medical_Image_Segmentation

This repository is for 2D Noisy-labeled Medical Image Segmentation with Confident Learning introduced by the following paper

Minqing Zhang, Jiantao Gao, Zhen Lyu, Weibing Zhao, Qin Wang, Weizhen Ding, Sheng Wang, Zhen Li* and Shuguang Cui, "Characterizing Label Errors: Confident Learning for Noisy-labeled Image Segmentation", MICCAI 2020. [Paper](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_70)  

You can consider citing this paper if it offer help in your work.

Zhang M, Gao J, Lyu Z, et al. Characterizing Label Errors: Confident Learning for Noisy-Labeled Image Segmentation[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020: 721-730.

![Pipeline](https://www.overleaf.com/project/5e61e23151ea080001abc1af/file/5e71af21516bf00001fd32b3)  

## Environments
All of the experiments reported in the paper were conducted under the following configuration. Other configurations might not be guaranteed feasible. <br>

Ubuntu 16.04.5 <br>
CUDA 10.0.130 <br>
Pytorch 1.2.0 <br>

## Organization
This project comprises of 10 folders and 2 scripts, and each of which is going to be described in the following <br>
<br>
/common    : general interfaces like model saving <br>
/config    : configurations related to training models <br>
/dataset   : dataset implement according to pytorch <br>
/jsrt_data : the original JSRT chest X-ray image dataset utilized to conduct our experiments <br>
/logger    : code involved with training logging <br>
/loss      : loss functions  <br>
/metrics   : metrics like dice-coefficient <br>
/models    : saving models <br>
/net       : network architectures <br>
/utils     : scripts involved with synthesizing noisy-labeled datasets and generating confident maps <br>

train_pixel_level_classification.py : segmentation model training <br>
test_pixel_level_classification.py  : testing a model <br>

## Instructions
1. synthesizing a noisy-labeled dataset with the script utils/noisy_dataset_generation_test.py (three variables: alpha, class_name and beta need to be specified, refering to our paper for more implementation details) <br>
2. preparing for teacher model training by specified settings in config/config_confident_learning_pixel_level_classification.py <br>
3. training teacher models with the script train_pixel_level_classification.py <br>
4. characterizing label errors with the script utils/confident_map_generation_test.py <br>
5. preparing for student model training by specified settings in config/config_confident_learning_pixel_level_classification.py  <br>
6. training a student model with the script train_pixel_level_classification.py <br>
7. testing the student model with the script test_pixel_level_classification.py <br>
