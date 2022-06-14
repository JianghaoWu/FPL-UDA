export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./PyMIC-master

## 2. get normalized uncertainty of each case 
python ./PyMIC-master/pymic/net_run/net_run_get_uncertainty.py train ./config_net/unet2d5_1.cfg



