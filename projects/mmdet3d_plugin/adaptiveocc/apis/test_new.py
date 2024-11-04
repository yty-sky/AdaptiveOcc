# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
import mmcv
import numpy as np
import pycocotools.mask as mask_util
#import open3d as o3d
import pdb
from ocnn.octree import Octree
from ocnn.octree.shuffled_key import xyz2key, key2xyz
from projects.mmdet3d_plugin.datasets.evaluation_metrics import octree_to_voxel

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]


def custom_single_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, is_vis=False):
    """Test model with single gpus."""

    model.eval()
    occ_results = []
    occ_results_short = []
    occ_results_medium = []
    occ_results_long = []

    dataset = data_loader.dataset

    prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            batch_size = 1

            if isinstance(result, dict):
                if 'evaluation' in result.keys():
                    occ_results.append(result['evaluation'][0])
                    occ_results_short.append(result['evaluation'][1])
                    occ_results_medium.append(result['evaluation'][2])
                    occ_results_long.append(result['evaluation'][3])

        for _ in range(batch_size):
            prog_bar.update()

    if is_vis:
        return

    return occ_results, occ_results_short, occ_results_medium, occ_results_long

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, is_vis=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    occ_results = []
    occ_results_short = []
    occ_results_medium = []
    occ_results_long = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            if isinstance(result, dict):
                if 'evaluation' in result.keys():
                    occ_results.append(result['evaluation'][0])
                    occ_results_short.append(result['evaluation'][1])
                    occ_results_medium.append(result['evaluation'][2])
                    occ_results_long.append(result['evaluation'][3])
                    batch_size = int(len(result['evaluation']) / 4)
            else:
                batch_size = 1

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    if is_vis:
        return

    # collect results from all ranks
    if gpu_collect:
        occ_results = collect_results_gpu(occ_results, len(dataset))
        occ_results_short = collect_results_gpu(occ_results_short, len(dataset))
        occ_results_medium = collect_results_gpu(occ_results_medium, len(dataset))
        occ_results_long = collect_results_gpu(occ_results_long, len(dataset))
    else:
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        occ_results = collect_results_cpu(occ_results, len(dataset), tmpdir)
        occ_results_short = collect_results_cpu(occ_results_short, len(dataset), tmpdir)
        occ_results_medium = collect_results_cpu(occ_results_medium, len(dataset), tmpdir)
        occ_results_long = collect_results_cpu(occ_results_long, len(dataset), tmpdir)

    return occ_results, occ_results_short, occ_results_medium, occ_results_long

def custom_single_gpu_test_ray(model, data_loader, is_octree=True):

    model.eval()
    tp_cnt_list = []
    gt_cnt_list = []
    pred_cnt_list = []
    tp_cnt_short_list = []
    gt_cnt_short_list = []
    pred_cnt_short_list = []

    dataset = data_loader.dataset
    batch_size = 1

    prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    iou = {}

    for j, data in enumerate(data_loader):
        with torch.no_grad():

            batch = data['img_metas'].data[0][0]
            data_id = batch['data_id']
            scene_id = batch['scene_id']
            frame_id = batch['frame_id']
            occ_gt_path = batch['occ_path']
            occ_gt_numpy = np.load(occ_gt_path)
            occ_gt = torch.tensor(occ_gt_numpy).squeeze(0).long()
            volume_gt = torch.zeros((200, 200, 32), dtype=torch.int64).cuda()
            occ_gt = occ_gt.cuda()
            if not is_octree:
                volume_gt[occ_gt[:, 0], occ_gt[:, 1], occ_gt[:, 2]] = occ_gt[:, 3]
            else:
                octree_gt = Octree(depth=4, pc_range=[-10, -10, -2.6, 10, 10, 0.6], occ_size=[25, 25, 4]) # new fast version
                octree_gt = octree_gt.cuda()
                octree_gt.build_octree_aux(occ_gt, [2, 4, 5], True, 6)
                for i in range(4):
                    pred = octree_gt.gt[i]
                    mask = (pred != 0) * (pred != 6)
                    pred = pred[mask]
                    key = octree_gt.keys[i][mask]
                    x, y, z = octree_gt.key2xyz(key, i)
                    gt_value = torch.stack([x, y, z, pred], dim=1)
                    volume_gt = octree_to_voxel(gt_value, volume_gt, i)

            occ_pred_path = batch['pred_path']
            occ_pred = np.load(occ_pred_path)
            occ_pred = torch.tensor(occ_pred).cuda()

            lidar_origin = torch.tensor(batch['poses'])
            lidar_origin = lidar_origin.unsqueeze(0).cuda()

            tp_cnt, gt_cnt, pred_cnt, tp_cnt_short, gt_cnt_short, pred_cnt_short = model(occ_pred=occ_pred, occ_gt=volume_gt, lidar_origin=lidar_origin)
            tp_cnt_list.append(tp_cnt)
            gt_cnt_list.append(gt_cnt)
            pred_cnt_list.append(pred_cnt)
            tp_cnt_short_list.append(tp_cnt_short)
            gt_cnt_short_list.append(gt_cnt_short)
            pred_cnt_short_list.append(pred_cnt_short)

            iou[data_id] = [f"{scene_id}_{frame_id}", tp_cnt_short[0] / (gt_cnt_short + pred_cnt_short - tp_cnt_short[0])]

        for _ in range(batch_size):
            prog_bar.update()

        if j == 20:
            break

    return tp_cnt_list, gt_cnt_list, pred_cnt_list, tp_cnt_short_list, gt_cnt_short_list, pred_cnt_short_list, iou


def custom_multi_gpu_test_ray(model, data_loader, tmpdir=None, gpu_collect=False, is_octree=True):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    tp_cnt_list = []
    gt_cnt_list = []
    pred_cnt_list = []
    tp_cnt_short_list = []
    gt_cnt_short_list = []
    pred_cnt_short_list = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    batch_size = 1
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    iou = {}
    for j, data in enumerate(data_loader):
        with torch.no_grad():
            print(j)
            batch = data['img_metas'].data[0][0]
            scene_id = batch['scene_id']
            frame_id = batch['frame_id']
            data_id = batch['data_id']
            occ_gt_path = batch['occ_path']
            occ_gt_numpy = np.load(occ_gt_path)
            occ_gt = torch.tensor(occ_gt_numpy).squeeze(0).long()
            occ_gt = occ_gt.cuda()
            volume_gt = torch.zeros((200, 200, 32), dtype=torch.int64).cuda()
            if not is_octree:
                volume_gt[occ_gt[:, 0], occ_gt[:, 1], occ_gt[:, 2]] = occ_gt[:, 3]
            else:
                octree_gt = Octree(depth=4, pc_range=[-10, -10, -2.6, 10, 10, 0.6], occ_size=[25, 25, 4]) # new fast version
                octree_gt = octree_gt.cuda()
                octree_gt.build_octree_aux(occ_gt, [2, 4, 5], True, 6)
                for i in range(4):
                    pred = octree_gt.gt[i]
                    mask = (pred != 0) * (pred != 6)
                    pred = pred[mask]
                    key = octree_gt.keys[i][mask]
                    x, y, z = octree_gt.key2xyz(key, i)
                    gt_value = torch.stack([x, y, z, pred], dim=1)
                    volume_gt = octree_to_voxel(gt_value, volume_gt, i)

            occ_pred_path = batch['pred_path']
            occ_pred = np.load(occ_pred_path)
            occ_pred = torch.tensor(occ_pred).cuda()

            lidar_origin = torch.tensor(batch['poses'])
            lidar_origin = lidar_origin.unsqueeze(0).cuda()

            tp_cnt, gt_cnt, pred_cnt, tp_cnt_short, gt_cnt_short, pred_cnt_short = model(occ_pred=occ_pred, occ_gt=volume_gt, lidar_origin=lidar_origin)
            tp_cnt_list.append(tp_cnt)
            gt_cnt_list.append(gt_cnt)
            pred_cnt_list.append(pred_cnt)
            tp_cnt_short_list.append(tp_cnt_short)
            gt_cnt_short_list.append(gt_cnt_short)
            pred_cnt_short_list.append(pred_cnt_short)

            iou[data_id] = [f"{scene_id}_{frame_id}", tp_cnt_short[0] / (gt_cnt_short + pred_cnt_short - tp_cnt_short[0])]

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

        # if j == 5:
        #     break

    # collect results from all ranks
    if gpu_collect:
        tp_cnt_list = collect_results_cpu(tp_cnt_list, len(dataset), tmpdir)
        gt_cnt_list = collect_results_cpu(gt_cnt_list, len(dataset), tmpdir)
        pred_cnt_list = collect_results_cpu(pred_cnt_list, len(dataset), tmpdir)
        tp_cnt_short_list = collect_results_cpu(tp_cnt_short_list, len(dataset), tmpdir)
        gt_cnt_short_list = collect_results_cpu(gt_cnt_short_list, len(dataset), tmpdir)
        pred_cnt_short_list = collect_results_cpu(pred_cnt_short_list, len(dataset), tmpdir)
    else:
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        tp_cnt_list = collect_results_cpu(tp_cnt_list, len(dataset), tmpdir)
        gt_cnt_list = collect_results_cpu(gt_cnt_list, len(dataset), tmpdir)
        pred_cnt_list = collect_results_cpu(pred_cnt_list, len(dataset), tmpdir)
        tp_cnt_short_list = collect_results_cpu(tp_cnt_short_list, len(dataset), tmpdir)
        gt_cnt_short_list = collect_results_cpu(gt_cnt_short_list, len(dataset), tmpdir)
        pred_cnt_short_list = collect_results_cpu(pred_cnt_short_list, len(dataset), tmpdir)
        iou = collect_results_cpu_dict(iou, len(dataset), tmpdir)

    return tp_cnt_list, gt_cnt_list, pred_cnt_list, tp_cnt_short_list, gt_cnt_short_list, pred_cnt_short_list, iou



def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def collect_results_cpu_dict(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = {}
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.update(mmcv.load(part_file))

        shutil.rmtree(tmpdir)
        return part_list

def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)

