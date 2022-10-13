# Test Time Adaptation in Medical Image Analysis

`Test Time Adaptation` (**TTA**) also known as `Source-free Domain Adaptation`(**SFDA**),  `Unsupervised Model Adaptation` (**UMA**).  

This is a curated list of **TTA** in medical image analysis, which also contains the generic **TTA** methodology. Welcome to add papers by pulling request or raising issues. 

## Overview

- [Tasks](#task)
  - [Segmentation](#segmentation)
  - [Classification](#classification)
  - [Detection](#detection)
- [Methodology](#methodology)
  - [Information Entropy](#information-entropy)
  - [Pseudo Labeling](#pseudo-labeling)
  - [Batch Normalization](#batch-normalization)
- [Dataset](#dataset)
- [Miscellaneous](#miscellaneous)

## Tasks
### Segmentation

| Date | First & Last Author | Title | Paper & Code |
| ---- | ------------------- | ----- | ------------ |
| 2020.12 | NeeravKarani & EnderKonukoglu | Test-time adaptable neural networks for robust medical image segmentation | [MIA](https://www.sciencedirect.com/science/article/pii/S1361841520302711), [code](https://github.com/neerakara/test-time-adaptable-neural-networks-for-domain-generalization)              |
| 2021.06 | YufanHea & JerryL.Prince | Autoencoder based self-supervised test-time adaptation for medical image analysis      | [MIA](https://www.sciencedirect.com/science/article/am/pii/S1361841521001821), [code](https://github.com/YufanHe/self-domain-adapted-network) |
| 2021.09 | Minhao Hu & Shaoting Zhang  | Fully Test-Time Adaptation for Image Segmentation | [MICCAI2021](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_24) |
| 2022.09 | Mathilde Bateson & Ismail Ben Ayed | Test-Time Adaptation with Shape Moments for Image Segmentation |[MICCAI2022](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_70), [code](https://github.com/mathilde-b/TTA)  |
| 2022.10 | Hao Li & Ipek Oguz | Self-supervised Test-Time Adaptation for Medical Image Segmentation | [MLCN2022](https://link.springer.com/chapter/10.1007/978-3-031-17899-3_4), [code](https://github.com/HaoLi12345/TTA) |
| 2022.04 | ChenYang & YixuanYuan | Source free domain adaptation for medical image segmentation with fourier style mining | [MIA](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001049), [code](https://github.com/CityU-AIM-Group/SFDA-FSM) |
| 2022.05 | Yang Hongzheng & Dou Qi | DLTTA: Dynamic Learning of Test-Time Adaptation for Cross-domain Medical Images | [TMI2022](https://ieeexplore.ieee.org/document/9830762),[code](https://github.com/med-air/DLTTA) |
| 2022.06 | Devavrat Tomar & Behzad Bozorgtabar | OptTTA: Learnable Test-Time Augmentation for Source-Free Medical Image Segmentation Under Domain Shift | [MIDL2022](https://openreview.net/pdf?id=B6HdQaY_iR), [code](https://openreview.net/pdf?id=B6HdQaY_iR) |
| 2022.02 | Neerav Karani & Ender Konukoglu | A Field of Experts Prior for Adapting Neural Networks at Test Time | [arxiv](https://arxiv.org/abs/2202.05271) |
|      |                     |       |              |
|      |                     |       |              |


### Classification

| Date | First & Last Author | Title | Paper & Code |
| ---- | ------------------- | ----- | ------------ |
| 2022.09 | Wenao Ma & Qi Dou | Test-Time Adaptation with Calibration of Medical Image Classification Nets for Label Distribution Shift | [MICCAI2022](https://link.springer.com/chapter/10.1007/978-3-031-16437-8_30),[code](https://github.com/med-air/TTADC) |


### Detection

| Date | First & Last Author | Title | Paper & Code |
| ---- | ------------------- | ----- | ------------ |
|      |                     |       |              |
|      |                     |       |              |
|      |                     |       |              |
|      |                     |       |              |

## Methodology

### Information Entropy
| Date    | First & Last Author                    | Title                                                        | Paper & Code                                                 |
| ------- | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2021.03 | Dequan Wang & Trevor Darrell | Tent: Fully test-time adaptation by entropy minimization | [ICLR2021]((https://arxiv.org/pdf/2006.10726)), [code](https://github.com/DequanWang/tent.git) |
### Pseudo Labeling
### Batch Normalization
| Date    | First & Last Author                    | Title                                                        | Paper & Code                                                 |
| ------- | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2020.06 | Zachary Nado & Jasper Snoek | Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift | [arxiv](https://arxiv.org/abs/2006.10963), [code](https://colab.research.google.com/drive/11N0wDZnMQQuLrRwRoumDCrhSaIhkqjof) |
| 2021.03 | Dequan Wang & Trevor Darrell | Tent: Fully test-time adaptation by entropy minimization | [ICLR2021]((https://arxiv.org/pdf/2006.10726)), [code](https://github.com/DequanWang/tent.git) |

## Dataset

| Dataset | Task | Modality & No | Description |
| ---- | ------------------- | ----- | ------------ |
| [M&Ms](https://www.ub.edu/mnms/) | Cardiac Segmentation | CMR & 375 volumes & 3 classes | Four scanners in six centers |
| [SCGM]()     | Spinal Cord Grey Matter Segmentation | MRI & 2 classes      | Four scanners in four centers |
|  [SAML](https://liuquande.github.io/SAML/) | Prostate MRI Segmentation | MRI & 116 volumes & 1 classes | Various scanners in six centers | 
|      |                     |       |              |

## Miscellaneous
- [Awesome Source-free Test-time Adaptation](https://github.com/YuejiangLIU/awesome-source-free-test-time-adaptation.git)






