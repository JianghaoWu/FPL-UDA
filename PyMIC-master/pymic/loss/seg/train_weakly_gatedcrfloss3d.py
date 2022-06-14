import argparse
import logging
import os
import random
import shutil
import sys
import time
from medpy import metric

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from train_weakly_gatedcrfloss2d import random_crop
from utils import losses, metrics, ramps, gated_crf_loss3d, gated_crf_loss_orig

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def test_single_volume(image, label, net, classes):
    label =  label.squeeze(0).cpu().detach().numpy()
    input = image.cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
            net(input), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def down_spacing_VS(spacing):
    spacing_down = [(x * 2).float().cuda() for x in spacing]
    spacing_down[0] = spacing[0].float().cuda()  # not change depth dimension
    return spacing_down

def down_spacing_BraTS(spacing):
    spacing_down = [(x * 2).float().cuda() for x in spacing]
    return spacing_down

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=args.in_channels, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, csv_file="data_train_sd2021_anno_expd.csv", transform=transforms.Compose([
        RandomGenerator()]))
    db_val = BaseDataSets(base_dir=args.root_path, csv_file="data_valid_sd2021.csv", transform=None)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=2)
    gated_crf_loss = gated_crf_loss3d.ModelLossSemsegGatedCRF3D()
    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = [eval(x) for x in args.kernel_radius]
    crop_size = [eval(x) for x in args.random_crop_size]
    if args.root_path == "../data/VS_data":
        down_size = (crop_size[0], crop_size[1] // 2, crop_size[2] // 2)
        down_spacing = down_spacing_VS
    elif args.root_path == "../data/BraTS2019":
        down_size = (crop_size[0] // 2, crop_size[1] // 2, crop_size[2] // 2)
        down_spacing = down_spacing_BraTS
    else:
        assert ValueError, "{0:} no such directory".format(args.root_path)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            iter_num = iter_num + 1
            id, size = sampled_batch["id"], sampled_batch["size"]
            spacing = sampled_batch["spacing"]
            spacing.reverse()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            new_label_batch = label_batch.clone()
            new_label_batch[label_batch == 0] = 2  # ignore_index = 2
            new_label_batch[label_batch == 1] = 0
            new_label_batch[label_batch == 2] = 1
            label_batch = new_label_batch
            volume_batch, label_batch = random_crop(volume_batch, label_batch, crop_size)

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            outputs_soft_down = F.interpolate(outputs_soft, size=down_size, mode="trilinear")
            volume_batch_down = F.interpolate(volume_batch, size=down_size, mode="trilinear")
            spacing_down = down_spacing(spacing)
            loss_gated_crf = gated_crf_loss(outputs_soft_down, loss_gatedcrf_kernels_desc, loss_gatedcrf_radius, \
            volume_batch_down, down_size[0], down_size[1], down_size[2], spacing_down)["loss"]
            loss = loss_ce + loss_gated_crf * 0.01 * ramps.sigmoid_rampup(iter_num, max_iterations // 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_gated_crf', loss_gated_crf, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_gated_crf: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_gated_crf.item()))

            if iter_num % 100 == 0:
                batch_idx = random.randint(0, volume_batch.size(0) - 1)
                depth_idx = int(volume_batch.size(2) / 2)
                depth_slice = slice(depth_idx - 2, depth_idx + 2)
                image = volume_batch[batch_idx, 0:1, depth_slice, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
                image = make_grid(image, image.size(0), normalize=True)
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(outputs_soft, dim=1, keepdim=True)[batch_idx, 0:1, \
                    depth_slice, :, :].permute(1, 0, 2, 3).repeat(1, 3, 1, 1).float()
                outputs = make_grid(outputs, outputs.size(0), normalize=True)
                writer.add_image('train/Prediction', outputs, iter_num)
                labs = label_batch[batch_idx, depth_slice, :, :].unsqueeze(1).repeat(1, 3, 1, 1).float()
                labs = make_grid(labs, labs.size(0), normalize=True)
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/BraTS2019', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='BraTS2019/weakly_gatedcrfloss3d', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--in_channels', type=int,  default=1,
                        help='input channel of network, i.e. number of data modality')
    parser.add_argument('--num_classes', type=int,  default=2,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument("--kernel_radius", nargs="+", help="loss_gatedcrf_radius", required=True)
    parser.add_argument("--random_crop_size", nargs="+", help="crop volume to a samller size DxHxW", required=True)
    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}".format(args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
