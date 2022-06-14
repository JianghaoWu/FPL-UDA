export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./PyMIC-master

## 3. train final segmentor
python ./PyMIC-master/pymic/net_run/net_run.py train ./config_net/unet2d5_1.cfg
## infer
# python ./PyMIC-master/pymic/net_run/net_run.py test ./config_net/unet2d5_1.cfg
## evaluation
# python ./PyMIC-master/pymic/util/evaluation_seg.py ./config_net/evaluation.cfg

