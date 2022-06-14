export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./PyMIC-master

## 1. train pseudo label generator
python ./PyMIC-master/pymic/net_run/net_run.py train ./config_net/unet2d5_1.cfg
