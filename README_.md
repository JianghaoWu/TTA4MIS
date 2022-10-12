* # Test-Time-Adaptation-for-Medical-Image-segmentation

* **[New], We are reformatting the codebase to support the 5-fold cross-validation and randomly select labeled cases, the reformatted methods in this [Branch](https://github.com/HiLab-git/SSL4MIS/tree/cross_val_dev)**. 

* Recently, semi-supervised image segmentation has become a hot topic in medical image computing, unfortunately, there are only a few open-source codes and datasets, since the privacy policy and others. For easy evaluation and fair comparison, we are trying to build a semi-supervised medical image segmentation benchmark to boost the semi-supervised learning research in the medical image computing community. **If you are interested, you can push your implementations or ideas to this repo or contact [me](https://luoxd1996.github.io/) at any time**.  

* This repo has re-implemented these semi-supervised methods (with some modifications for semi-supervised medical image segmentation, more details please refer to these original works): (1) [Mean Teacher](https://papers.nips.cc/paper/6719-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results.pdf); (2) [Entropy Minimization](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf); (3) [Deep Adversarial Networks](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_47); (4) [Uncertainty Aware Mean Teacher](https://arxiv.org/pdf/1907.07034.pdf); (5) [Interpolation Consistency Training](https://arxiv.org/pdf/1903.03825.pdf); (6) [Uncertainty Rectified Pyramid Consistency](https://arxiv.org/pdf/2012.07042.pdf); (7) [Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226); (8) [Cross Consistency Training](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ouali_Semi-Supervised_Semantic_Segmentation_With_Cross-Consistency_Training_CVPR_2020_paper.pdf); (9) [Deep Co-Training](https://openaccess.thecvf.com/content_ECCV_2018/papers/Siyuan_Qiao_Deep_Co-Training_for_ECCV_2018_paper.pdf); (10) [Cross Teaching between CNN and Transformer](https://arxiv.org/pdf/2112.04894.pdf); (11) [FixMatch](https://arxiv.org/abs/2001.07685). In addition, several backbones networks (both 2D and 3D) are also supported in this repo, such as **UNet, nnUNet, VNet, AttentionUNet, ENet, Swin-UNet, etc**.

* This project was originally developed for our previous works [URPC](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_30) ([MICCAI2021](https://miccai2021.org/en/) early accept). Now and future, we are still working on extending it to be more user-friendly and support more approaches to further boost and ease this topic research. **If you use this codebase in your research, please cite the following works**:

		@article{luo2021ctbct,
    		title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
    		author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
    		booktitle={Medical Imaging with Deep Learning},
    		year={2022}}

		@InProceedings{luo2021urpc,
		author={Luo, Xiangde and Liao, Wenjun and Chen, Jieneng and Song, Tao and Chen, Yinan and Zhang, Shichuan and Chen, Nianyong and Wang, Guotai and Zhang, Shaoting},
		title={Efficient Semi-supervised Gross Target Volume of Nasopharyngeal Carcinoma Segmentation via Uncertainty Rectified Pyramid Consistency},
		booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021},
		year={2021},
		pages={318--329}}
		 
		@InProceedings{luo2021dtc,
		title={Semi-supervised Medical Image Segmentation through Dual-task Consistency},
		author={Luo, Xiangde and Chen, Jieneng and Song, Tao and  Wang, Guotai},
		journal={AAAI Conference on Artificial Intelligence},
		year={2021},
		pages={8801-8809}}
		
		@misc{ssl4mis2020,
		title={{SSL4MIS}},
		author={Luo, Xiangde},
		howpublished={\url{https://github.com/HiLab-git/SSL4MIS}},
		year={2020}}
	
## Literature reviews of Test Time Adaptation for medical image segmentation (**TTA4MIS**).
|Date|The First and Last Authors|Title|Code|Reference|
|---|---|---|---|---|
|2022-05|W. Huang and F. Wu|Semi-Supervised Neuron Segmentation via Reinforced Consistency Learning|[Code](https://github.com/weih527/SSNS-Net)|[TMI2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9777694)|
|2022-05|C. Lee and M. Chung|Voxel-wise Adversarial Semi-supervised Learning for Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2205.06987.pdf)|
|2022-05|Y. Lin and X. Li|Calibrating Label Distribution for Class-Imbalanced Barely-Supervised Knee Segmentation|[Code](https://github.com/xmed-lab/CLD-Semi)|[MICCAI2022](https://arxiv.org/pdf/2205.03644.pdf)|
|2022-05|K. Zheng and J. Wei|Double Noise Mean Teacher Self-Ensembling Model for Semi-Supervised Tumor Segmentation|None|[ICASSP2022](https://ieeexplore.ieee.org/abstract/document/9746957)|
|2022-04|Y. Xiao and G. Yang|Semi-Supervised Segmentation of Mitochondria from Electron Microscopy Images Using Spatial Continuity|[Code](https://github.com/cbmi-group/MPP)|[ISBI2022](https://ieeexplore.ieee.org/document/9761519)|
|2022-04|H. He and V. Grau|Semi-Supervised Coronary Vessels Segmentation from Invasive Coronary Angiography with Connectivity-Preserving Loss Function|None|[ISBI2022](https://ieeexplore.ieee.org/document/9761695)|
|2022-04|B. Thompson and J. Voisey|Pseudo-Label Refinement Using Superpixels for Semi-Supervised Brain Tumour Segmentation|None|[ISBI2022](https://ieeexplore.ieee.org/document/9761681)|
|2022-04|Z li and X. Fan|Coupling Deep Deformable Registration with Contextual Refinement for Semi-Supervised Medical Image Segmentation|None|[ISBI2022](https://ieeexplore.ieee.org/document/9761683)|
|2022-04|A. Xu and X. Xia|Ca-Mt: A Self-Ensembling Model for Semi-Supervised Cardiac Segmentation with Elliptical Descriptor Based Contour-Aware|None|[ISBI2022](https://ieeexplore.ieee.org/abstract/document/9761666)|
|2022-04|X. Wang and S. Chen|SSA-Net: Spatial Self-Attention Network for COVID-19 Pneumonia Infection Segmentation with Semi-supervised Few-shot Learning|None|[MedIA2022](https://www.sciencedirect.com/science/article/pii/S1361841522001062)|
|2022-04|Z. Zhang and X. Tian|Discriminative Error Prediction Network for Semi-supervised Colon Gland Segmentation|None|[MedIA2022](https://www.sciencedirect.com/science/article/pii/S1361841522001050)|
|2022-04|Z. Xiao and W. Zhang|Efficient Combination of CNN and Transformer for Dual-Teacher Uncertainty-Aware Guided Semi-Supervised Medical Image Segmentation|None|[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4081789)|
|2022-04|K. Han and Z. Liu|An Effective Semi-supervised Approach for Liver CT Image Segmentation|None|[JBHI2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9757875)|
|2022-04|J. Yang and Q. Chen|Self-Supervised Sequence Recovery for SemiSupervised Retinal Layer Segmentation|None|[JBHI2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9756342)|
|2022-04|T. Cheng and C. Cheng|Feature-enhanced Adversarial Semi-supervised Semantic Segmentation Network for Pulmonary Embolism Annotation|None|[Arxiv](https://arxiv.org/ftp/arxiv/papers/2204/2204.04217.pdf)|
|2022-04|K. Wang and Y. Wang|Semi-supervised Medical Image Segmentation via a Tripled-uncertainty Guided Mean Teacher Model with Contrastive Learning|None|[MedIA2022](https://www.sciencedirect.com/science/article/pii/S1361841522000925)|
|2022-04|M. Liu and Q. He|CCAT-NET: A Novel Transformer Based Semi-supervised Framework for Covid-19 Lung Lesion Segmentation|None|[ISBI2022](https://arxiv.org/ftp/arxiv/papers/2204/2204.02839.pdf)|
|2022-03|Y. Liu and G. Carneiro|Translation Consistent Semi-supervised Segmentation for 3D Medical Images|[Code](https://github.com/yyliu01/TraCoCo)|[Arxiv](https://arxiv.org/pdf/2203.14523.pdf)|
|2022-03|Z. Xu and R. Tong|All-Around Real Label Supervision: Cyclic Prototype Consistency Learning for Semi-supervised Medical Image Segmentation|None|[JBHI2022](https://ieeexplore.ieee.org/document/9741294)|
|2022-03|M. Huang and Q. Feng|Semi-Supervised Hybrid Spine Network for Segmentation of Spine MR Images|[Code](https://github.com/Meiyan88/SSHSNet)|[Arxiv](https://arxiv.org/pdf/2203.12151.pdf)|
|2022-03|S. Adiga V and H. Lombaert|Leveraging Labeling Representations in Uncertainty-based Semi-supervised Segmentation|None|[Arxiv](https://arxiv.org/pdf/2203.05682.pdf)|
|2022-03|M. Tran and T. Peng|S5CL: Unifying Fully-Supervised, Self-Supervised, and Semi-Supervised Learning Through Hierarchical Contrastive Learning|[Code](https://github.com/manuel-tran/s5cl)|[Arxiv](https://arxiv.org/pdf/2203.07307.pdf)|
|2022-03|M. Waerebeke and J. Dole|On the pitfalls of entropy-based uncertainty for multi-class semi-supervised segmentation|None|[Arxiv](https://arxiv.org/pdf/2203.03587.pdf)|
|2022-03|W. Cui and R. M. Leahy|Semi-supervised Learning using Robust Loss|None|[Arxiv](https://arxiv.org/pdf/2203.01524.pdf)|
|2022-03|Y. Wu and J. Cai|Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2203.01324.pdf)|
|2022-02|Y. Hua and L. Zhang|Uncertainty-Guided Voxel-Level Supervised Contrastive Learning for Semi-Supervised Medical Image Segmentation|None|[IJNS2022](https://www.worldscientific.com/doi/epdf/10.1142/S0129065722500162)|
|2022-02|Y. Shu and W. Li|Cross-Mix Monitoring for Medical Image Segmentation with Limited Supervision|None|[TMM2022](https://ieeexplore.ieee.org/abstract/document/9721091)|
|2022-02|H. Huang and H. Hu|MTL-ABS3Net: Atlas-Based Semi-Supervised Organ Segmentation Network with Multi-Task Learning for Medical Images|None|[JHBI2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9721677)|
|2022-02|H. Wu and J. Qin|Semi-supervised Segmentation of Echocardiography Videos via Noise-resilient Spatiotemporal Semantic Calibration and Fusion|None|[MedIA2022](https://www.sciencedirect.com/science/article/pii/S1361841522000494)|
|2022-02|Z. Liu and C. Zhao|Semi-supervised Medical Image Segmentation via Geometry-aware Consistency Training|None|[Arxiv](https://arxiv.org/ftp/arxiv/papers/2202/2202.06104.pdf)|
|2022-02|X. Zhao and G. Li|Cross-level Contrastive Learning and Consistency Constraint for Semi-supervised Medical Image Segmentation|[Code](https://github.com/ShinkaiZ/CLCC-semi)|[ISBI2022](https://arxiv.org/pdf/2202.04074.pdf)|
|2022-02|H. Basak and A. Chatterjee|An Embarrassingly Simple Consistency Regularization Method for Semi-Supervised Medical Image Segmentation|[Code](https://github.com/hritam-98/ICT-MedSeg)|[ISBI2022](https://arxiv.org/abs/2202.00677)|
|2022-01|Q. Chen and D. Ming|Semi-supervised 3D Medical Image Segmentation Based on Dual-task Consistent joint Leanrning and Task-Level Regularization|None|[TCBB2022](https://ieeexplore.ieee.org/document/9689970)|
|2022-01|H. Yao and X. Li|Enhancing Pseudo Label Quality for Semi-Supervised Domain-Generalized Medical Image Segmentation|None|[AAAI2022](https://arxiv.org/pdf/2201.08657.pdf)|
|2021-12|S. Li and X. Yang|Semi-supervised Cardiac MRI Segmentation Based on Generative Adversarial Network and Variational Auto-Encoder|None|[BIBM2021](https://ieeexplore.ieee.org/document/9669685)|
|2021-12|N. Zhang and Y. Zhang|Semi-supervised Medical Image Segmentation with Distribution Calibration and Non-local Semantic Constraint|None|[BIBM2021](https://ieeexplore.ieee.org/document/9669560)|
|2021-12|S. Liu and G. Cao|Shape-aware Multi-task Learning for Semi-supervised 3D Medical Image Segmentation|None|[BIBM2021](https://ieeexplore.ieee.org/document/9669523)|
|2021-12|X. Xu and P. Yan|Shadow-consistent Semi-supervised Learning for Prostate Ultrasound Segmentation|[Code](https://github.com/DIAL-RPI/SCO-SSL)|[TMI2021](https://ieeexplore.ieee.org/document/9667363)|
|2021-12|L. Hu and Y. Wang|Semi-supervised NPC segmentation with uncertainty and attention guided consistency|None|[KBS2021](https://www.sciencedirect.com/science/article/abs/pii/S0950705121011205)|
|2021-12|J. Peng and M. Pedersoli|Self-Paced Contrastive Learning for Semi-supervised Medical Image Segmentation with Meta-labels|[Code](https://github.com/jizongFox/Self-paced-Contrastive-Learning)|[NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf)|
|2021-12|Y. Xie and Y. Xia|Intra- and Inter-pair Consistency for Semi-supervised Gland Segmentation|None|[TIP2021](https://ieeexplore.ieee.org/document/9662661)|
|2021-12|K. Chaitanya and E. Konukoglu|Local contrastive loss with pseudo-label based self-training for semi-supervised medical image segmentation|[Code](https://github.com/krishnabits001/pseudo_label_contrastive_training)|[Arxiv](https://arxiv.org/pdf/2112.09645.pdf)|
## Code for semi-supervised medical image segmentation.
Some implementations of semi-supervised learning methods can be found in this [Link](https://github.com/Luoxd1996/SSL4MIS/tree/master/code).

## Conclusion
* This repository provides daily-update literature reviews, algorithms' implementation, and some examples of using PyTorch for semi-supervised medical image segmentation. The project is under development. Currently, it supports 2D and 3D semi-supervised image segmentation and includes five widely-used algorithms' implementations.
	
* In the next two or three months, we will provide more algorithms' implementations, examples, and pre-trained models.

## Questions and Suggestions
* If you have any questions or suggestions about this project, please contact me through email: `luoxd1996@gmail.com` or QQ Group (Chinese):`906808850`. 
