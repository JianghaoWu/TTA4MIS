# CUDA_VISIBLE_DEVICES=2 python 01_train_source_2d.py --cfg cfgs/prostate/source.yaml
# CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/prostate/source_test.yaml
# CUDA_VISIBLE_DEVICES=1 python 01_adaptation.py --cfg cfgs/prostate/norm_prostate.yaml
# CUDA_VISIBLE_DEVICES=1 python 01_adaptation.py --cfg cfgs/prostate/tent_prostate.yaml
# CUDA_VISIBLE_DEVICES=1 python 01_adaptation.py --cfg cfgs/prostate/cotta_prostate.yaml
# CUDA_VISIBLE_DEVICES=1 python 01_adaptation.py --cfg cfgs/prostate/meant.yaml
# CUDA_VISIBLE_DEVICES=2 python 01_adaptation.py --cfg cfgs/prostate/sar_prostate.yaml

# CUDA_VISIBLE_DEVICES=2 python 01_adaptation.py --cfg cfgs/prostate/wjh01_prostate.yaml
# CUDA_VISIBLE_DEVICES=0 python 01_adaptation.py --cfg cfgs/prostate/upl_prostate.yaml