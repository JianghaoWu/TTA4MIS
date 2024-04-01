# CUDA_VISIBLE_DEVICES=2 python 01_train_source_2d.py --cfg cfgs/fb/source.yaml
# CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/fb/source_test.yaml
# CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/fb/tent.yaml
CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/fb/norm.yaml
# CUDA_VISIBLE_DEVICES=2 python 01_adaptation.py --cfg cfgs/mms/source.yaml
CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/fb/cotta.yaml
# CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/fb/meant.yaml
# CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/fb/wjh01.yaml
# CUDA_VISIBLE_DEVICES=2 python 01_train_source_2d.py --cfg cfgs/prostate/source.yaml
# CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/prostate/source_test.yaml
# CUDA_VISIBLE_DEVICES=2 python 01_adaptation.py --cfg cfgs/prostate/norm_prostate.yaml
# CUDA_VISIBLE_DEVICES=2 python 01_adaptation.py --cfg cfgs/prostate/tent_prostate.yaml

# CUDA_VISIBLE_DEVICES=2 python 01_adaptation.py --cfg cfgs/prostate/cotta_prostate.yaml
# CUDA_VISIBLE_DEVICES=2 python 01_adaptation.py --cfg cfgs/prostate/sar_prostate.yaml
# CUDA_VISIBLE_DEVICES=2 python 01_adaptation.py --cfg cfgs/prostate/wjh01.yaml
# CUDA_VISIBLE_DEVICES=1 python 01_adaptation.py --cfg cfgs/mms/wjh01.yaml
# CUDA_VISIBLE_DEVICES=3 python 01_adaptation.py --cfg cfgs/prostate/wjh01_prostate.yaml
