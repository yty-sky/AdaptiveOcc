import numpy as np
import torch
import chamfer
from ocnn.octree import Octree
from ocnn.octree.shuffled_key import xyz2key, key2xyz

distance_table = [80, 160, 240, 320]
distance_table2 = [40, 80, 120, 160]
results_all = np.zeros(17)

def voxel_to_vertices(voxel, img_metas, thresh=0.5):
    x = torch.linspace(0, voxel.shape[0] - 1, voxel.shape[0])
    y = torch.linspace(0, voxel.shape[1] - 1, voxel.shape[1])
    z = torch.linspace(0, voxel.shape[2] - 1, voxel.shape[2])
    X, Y, Z = torch.meshgrid(x, y, z)
    vv = torch.stack([X, Y, Z], dim=-1).to(voxel.device)

    vertices = vv[voxel > thresh]
    vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
    vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
    vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]

    return vertices

def gt_to_vertices(gt, img_metas):
    gt[:, 0] = (gt[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
    gt[:, 1] = (gt[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
    gt[:, 2] = (gt[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]
    return gt

def gt_to_voxel(gt, img_metas):
    voxel = np.zeros(img_metas['occ_size'])
    voxel[gt[:, 0].astype(np.int), gt[:, 1].astype(np.int), gt[:, 2].astype(np.int)] = gt[:, 3]

    return voxel

def octree_to_voxel(pred_value, pred_volume, depth):

    N = 2 ** (4 - depth)
    label_value = pred_value[:, 3]
    pred_value = pred_value[:, :3] * N
    x = torch.arange(0, N)
    y = torch.arange(0, N)
    z = torch.arange(0, N)
    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z)
    x_grid = x_grid.cuda()
    y_grid = y_grid.cuda()
    z_grid = z_grid.cuda()
    pred_volume[pred_value[:, 0][:, None, None, None] + x_grid,
                pred_value[:, 1][:, None, None, None] + y_grid,
                pred_value[:, 2][:, None, None, None] + z_grid] = label_value[:, None, None, None]

    return pred_volume

def octree_to_voxel_new(pred_value, pred_volume, depth):

    N = 2 ** (3 - depth)
    label_value = pred_value[:, 3]
    pred_value = pred_value[:, :3] * N
    x = torch.arange(0, N)
    y = torch.arange(0, N)
    z = torch.arange(0, N)
    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z)
    x_grid = x_grid.cuda()
    y_grid = y_grid.cuda()
    z_grid = z_grid.cuda()
    pred_volume[pred_value[:, 0][:, None, None, None] + x_grid,
                pred_value[:, 1][:, None, None, None] + y_grid,
                pred_value[:, 2][:, None, None, None] + z_grid] = label_value[:, None, None, None]

    return pred_volume

def eval_3d(verts_pred, verts_trgt, threshold=.5):
    d1, d2, idx1, idx2 = chamfer.forward(verts_pred.unsqueeze(0).type(torch.float), verts_trgt.unsqueeze(0).type(torch.float))
    dist1 = torch.sqrt(d1).cpu().numpy()
    dist2 = torch.sqrt(d2).cpu().numpy()
    cd = dist1.mean() + dist2.mean()
    precision = np.mean((dist1<threshold).astype('float'))
    recal = np.mean((dist2<threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = np.array([np.mean(dist1),np.mean(dist2),cd, precision,recal,fscore])
    return metrics

def evaluation_reconstruction(pred_occ, gt_occ, img_metas):
    results = []
    for i in range(pred_occ.shape[0]):
        
        vertices_pred = voxel_to_vertices(pred_occ[i], img_metas, thresh=0.25) #set low thresh for class imbalance problem
        vertices_gt = gt_to_vertices(gt_occ[i][..., :3], img_metas)
        
        metrics = eval_3d(vertices_pred.type(torch.double), vertices_gt.type(torch.double)) #must convert to double, a bug in chamfer
        results.append(metrics)
    return np.stack(results, axis=0)

def evaluation_semantic(pred_occ, pred_octree, gt_occ, img_metas, class_num):
    results = []

    pred_volume = torch.zeros((400, 400, 32), dtype=torch.int64)
    pred_volume = pred_volume.cuda()
    volume_gt = torch.zeros((400, 400, 32), dtype=torch.int64)
    volume_gt = volume_gt.cuda()
    volume_gt[gt_occ[:, 0], gt_occ[:, 1], gt_occ[:, 2]] = gt_occ[:, 3]

    # _, pred = torch.max(torch.softmax(pred_occ[-1], dim=1), dim=1)
    # pred_volume = pred[0,:]


    for i in range(len(pred_occ)):
        _, pred = torch.max(torch.softmax(pred_occ[i], dim=1), dim=1)
        mask = (pred != 0) * (pred != 17)
        pred = pred[mask]
        key = pred_octree.keys[i][mask]
        x, y, z = pred_octree.key2xyz(key, i)
        pred_value = torch.stack([x, y, z, pred], dim=1)
        pred_volume = octree_to_voxel(pred_value, pred_volume, i)

        # pred = octree_gt.gt[i]
        # mask = (pred != 0) * (pred != 17)
        # pred = pred[mask]
        # key = octree_gt.keys[i][mask]
        # x, y, z = key2xyz(key, i)
        # gt_value = torch.stack([x, y, z, pred], dim=1)
        # volume_gt = octree_to_voxel(gt_value, volume_gt, i)

    gt_i, pred_i = volume_gt.cpu().numpy(), pred_volume.cpu().numpy()
    mask = (gt_i != 255)

    dist_low2, dist_low1, dist_high1, dist_high2 = distance_table[0], distance_table[1], distance_table[2], distance_table[3]
    mask_A = np.zeros((400, 400, 32), dtype=np.bool)
    mask_B = np.zeros((400, 400, 32), dtype=np.bool)
    mask_A[dist_low1:dist_high1, dist_low1:dist_high1, :] = True
    mask_B[dist_low2:dist_high2, dist_low2:dist_high2, :] = True
    mask_B = mask_B & ~mask_A
    mask_C = ~(mask_A | mask_B)

    score = np.zeros((class_num, 3))
    score_short = np.zeros((class_num, 3))
    score_medium = np.zeros((class_num, 3))
    score_long = np.zeros((class_num, 3))

    for j in range(class_num):
        if j == 0: #class 0 for geometry IoU
            score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
            score[j][1] += (gt_i[mask] != 0).sum()
            score[j][2] += (pred_i[mask] != 0).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] != 0) * (pred_i[mask & mask_A] != 0)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] != 0).sum()
            score_short[j][2] += (pred_i[mask & mask_A] != 0).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] != 0) * (pred_i[mask & mask_B] != 0)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] != 0).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] != 0).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] != 0) * (pred_i[mask & mask_C] != 0)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] != 0).sum()
            score_long[j][2] += (pred_i[mask & mask_C] != 0).sum()

        else:
            score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
            score[j][1] += (gt_i[mask] == j).sum()
            score[j][2] += (pred_i[mask] == j).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] == j) * (pred_i[mask & mask_A] == j)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] == j).sum()
            score_short[j][2] += (pred_i[mask & mask_A] == j).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] == j) * (pred_i[mask & mask_B] == j)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] == j).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] == j).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] == j) * (pred_i[mask & mask_C] == j)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] == j).sum()
            score_long[j][2] += (pred_i[mask & mask_C] == j).sum()


    results.append(score)
    results.append(score_short)
    results.append(score_medium)
    results.append(score_long)

    return np.stack(results, axis=0)



def evaluation_octree_semantic(pred_occ, pred_octree, gt_occ, img_metas, class_num):
    results = []

    pred_volume = torch.zeros((400, 400, 32), dtype=torch.int64)
    pred_volume = pred_volume.cuda()
    volume_gt = torch.zeros((400, 400, 32), dtype=torch.int64)
    volume_gt = volume_gt.cuda()

    octree_gt = Octree(depth=5, pc_range=[-50, -50, -5.0, 50, 50, 3.0], occ_size=[25, 25, 4])
    octree_gt.cuda()
    octree_gt.build_octree(torch.squeeze(gt_occ, 0), img_metas['build_octree'], img_metas['build_octree_up'])

    for i in range(len(pred_occ)):
        _, pred = torch.max(torch.softmax(pred_occ[i], dim=1), dim=1)
        mask = (pred != 0) * (pred != 17)
        pred = pred[mask]
        key = pred_octree.keys[i][mask]
        x, y, z = pred_octree.key2xyz(key, i)
        pred_value = torch.stack([x, y, z, pred], dim=1)
        pred_volume = octree_to_voxel(pred_value, pred_volume, i)

    for i in range(5):
        pred = octree_gt.gt[i]
        mask = (pred != 0) * (pred != 17)
        pred = pred[mask]
        key = octree_gt.keys[i][mask]
        x, y, z = octree_gt.key2xyz(key, i)
        gt_value = torch.stack([x, y, z, pred], dim=1)
        volume_gt = octree_to_voxel(gt_value, volume_gt, i)

    gt_i, pred_i = volume_gt.cpu().numpy(), pred_volume.cpu().numpy()
    mask = (gt_i != 255)

    dist_low2, dist_low1, dist_high1, dist_high2 = distance_table[0], distance_table[1], distance_table[2], distance_table[3]
    mask_A = np.zeros((400, 400, 32), dtype=np.bool)
    mask_B = np.zeros((400, 400, 32), dtype=np.bool)
    mask_A[dist_low1:dist_high1, dist_low1:dist_high1, :] = True
    mask_B[dist_low2:dist_high2, dist_low2:dist_high2, :] = True
    mask_B = mask_B & ~mask_A
    mask_C = ~(mask_A | mask_B)

    score = np.zeros((class_num, 3))
    score_short = np.zeros((class_num, 3))
    score_medium = np.zeros((class_num, 3))
    score_long = np.zeros((class_num, 3))

    for j in range(class_num):
        if j == 0: #class 0 for geometry IoU
            score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
            score[j][1] += (gt_i[mask] != 0).sum()
            score[j][2] += (pred_i[mask] != 0).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] != 0) * (pred_i[mask & mask_A] != 0)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] != 0).sum()
            score_short[j][2] += (pred_i[mask & mask_A] != 0).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] != 0) * (pred_i[mask & mask_B] != 0)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] != 0).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] != 0).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] != 0) * (pred_i[mask & mask_C] != 0)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] != 0).sum()
            score_long[j][2] += (pred_i[mask & mask_C] != 0).sum()

        else:
            score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
            score[j][1] += (gt_i[mask] == j).sum()
            score[j][2] += (pred_i[mask] == j).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] == j) * (pred_i[mask & mask_A] == j)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] == j).sum()
            score_short[j][2] += (pred_i[mask & mask_A] == j).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] == j) * (pred_i[mask & mask_B] == j)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] == j).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] == j).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] == j) * (pred_i[mask & mask_C] == j)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] == j).sum()
            score_long[j][2] += (pred_i[mask & mask_C] == j).sum()


    results.append(score)
    results.append(score_short)
    results.append(score_medium)
    results.append(score_long)

    return np.stack(results, axis=0)


def evaluation_gt_semantic(octree, gt_occ, img_metas, class_num):
    results = []

    pred_volume = torch.zeros((400, 400, 32), dtype=torch.int64)
    pred_volume = pred_volume.cuda()
    volume_gt = torch.zeros((400, 400, 32), dtype=torch.int64)
    volume_gt = volume_gt.cuda()
    volume_gt[gt_occ[:, 0], gt_occ[:, 1], gt_occ[:, 2]] = gt_occ[:, 3]

    # _, pred = torch.max(torch.softmax(pred_occ[-1], dim=1), dim=1)
    # pred_volume = pred[0,:]


    for i in range(5):
        pred = octree.gt[i]
        mask = (pred != 0) * (pred != 17) *  (pred != 255)
        pred = pred[mask]
        key = octree.keys[i][mask]
        x, y, z = octree.key2xyz(key, i)
        pred_value = torch.stack([x, y, z, pred], dim=1)
        pred_volume = octree_to_voxel(pred_value, pred_volume, i)

        # pred = octree_gt.gt[i]
        # mask = (pred != 0) * (pred != 17)
        # pred = pred[mask]
        # key = octree_gt.keys[i][mask]
        # x, y, z = key2xyz(key, i)
        # gt_value = torch.stack([x, y, z, pred], dim=1)
        # volume_gt = octree_to_voxel(gt_value, volume_gt, i)

    gt_i, pred_i = volume_gt.cpu().numpy(), pred_volume.cpu().numpy()
    mask = (gt_i != 255)

    dist_low2, dist_low1, dist_high1, dist_high2 = distance_table[0], distance_table[1], distance_table[2], distance_table[3]
    mask_A = np.zeros((400, 400, 32), dtype=np.bool)
    mask_B = np.zeros((400, 400, 32), dtype=np.bool)
    mask_A[dist_low1:dist_high1, dist_low1:dist_high1, :] = True
    mask_B[dist_low2:dist_high2, dist_low2:dist_high2, :] = True
    mask_B = mask_B & ~mask_A
    mask_C = ~(mask_A | mask_B)

    score = np.zeros((class_num, 3))
    score_short = np.zeros((class_num, 3))
    score_medium = np.zeros((class_num, 3))
    score_long = np.zeros((class_num, 3))

    for j in range(class_num):
        if j == 0: #class 0 for geometry IoU
            score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
            score[j][1] += (gt_i[mask] != 0).sum()
            score[j][2] += (pred_i[mask] != 0).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] != 0) * (pred_i[mask & mask_A] != 0)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] != 0).sum()
            score_short[j][2] += (pred_i[mask & mask_A] != 0).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] != 0) * (pred_i[mask & mask_B] != 0)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] != 0).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] != 0).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] != 0) * (pred_i[mask & mask_C] != 0)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] != 0).sum()
            score_long[j][2] += (pred_i[mask & mask_C] != 0).sum()

        else:
            score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
            score[j][1] += (gt_i[mask] == j).sum()
            score[j][2] += (pred_i[mask] == j).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] == j) * (pred_i[mask & mask_A] == j)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] == j).sum()
            score_short[j][2] += (pred_i[mask & mask_A] == j).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] == j) * (pred_i[mask & mask_B] == j)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] == j).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] == j).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] == j) * (pred_i[mask & mask_C] == j)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] == j).sum()
            score_long[j][2] += (pred_i[mask & mask_C] == j).sum()


    results.append(score)
    results.append(score_short)
    results.append(score_medium)
    results.append(score_long)

    return np.stack(results, axis=0)

def print_results(results_all):
    result_tmp = results_all
    result_tmp = result_tmp / result_tmp[-1]
    print(result_tmp)

def evaluation_gt(gt_occ, img_metas, class_num):

    results = np.zeros(class_num)
    for j in range(1, class_num):
        semantic_bool = (gt_occ[:,3] == j)
        results[j-1] = torch.sum(semantic_bool)

    total = np.sum(results)
    results[class_num-1]=total
    global results_all
    results_all = results_all + results
    print_results(results_all)
    return results
    # return np.stack(results, axis=0)

def evaluation_semantic_new(pred_occ, pred_octree, gt_occ, img_metas, class_num):
    results = []

    pred_volume = torch.zeros((200, 200, 16), dtype=torch.int64)
    pred_volume = pred_volume.cuda()
    volume_gt = torch.zeros((200, 200, 16), dtype=torch.int64)
    volume_gt = volume_gt.cuda()
    volume_gt[gt_occ[:, 0], gt_occ[:, 1], gt_occ[:, 2]] = gt_occ[:, 3]
    # pred_volume = pred_occ.squeeze()

    for i in range(len(pred_occ)):
        _, pred = torch.max(torch.softmax(pred_occ[i], dim=1), dim=1)
        mask = (pred != 0) * (pred != 17)
        pred = pred[mask]
        key = pred_octree.keys[i][mask]
        x, y, z = pred_octree.key2xyz(key, i)
        pred_value = torch.stack([x, y, z, pred], dim=1)
        pred_volume = octree_to_voxel_new(pred_value, pred_volume, i)

    gt_i, pred_i = volume_gt.cpu().numpy(), pred_volume.cpu().numpy()
    mask = (gt_i != 255)

    dist_low2, dist_low1, dist_high1, dist_high2 = distance_table2[0], distance_table2[1], distance_table2[2], distance_table2[3]
    mask_A = np.zeros((200, 200, 16), dtype=np.bool)
    mask_B = np.zeros((200, 200, 16), dtype=np.bool)
    mask_A[dist_low1:dist_high1, dist_low1:dist_high1, :] = True
    mask_B[dist_low2:dist_high2, dist_low2:dist_high2, :] = True
    mask_B = mask_B & ~mask_A
    mask_C = ~(mask_A | mask_B)

    score = np.zeros((class_num, 3))
    score_short = np.zeros((class_num, 3))
    score_medium = np.zeros((class_num, 3))
    score_long = np.zeros((class_num, 3))

    for j in range(class_num):
        if j == 0: #class 0 for geometry IoU
            score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
            score[j][1] += (gt_i[mask] != 0).sum()
            score[j][2] += (pred_i[mask] != 0).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] != 0) * (pred_i[mask & mask_A] != 0)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] != 0).sum()
            score_short[j][2] += (pred_i[mask & mask_A] != 0).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] != 0) * (pred_i[mask & mask_B] != 0)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] != 0).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] != 0).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] != 0) * (pred_i[mask & mask_C] != 0)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] != 0).sum()
            score_long[j][2] += (pred_i[mask & mask_C] != 0).sum()

        else:
            score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
            score[j][1] += (gt_i[mask] == j).sum()
            score[j][2] += (pred_i[mask] == j).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] == j) * (pred_i[mask & mask_A] == j)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] == j).sum()
            score_short[j][2] += (pred_i[mask & mask_A] == j).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] == j) * (pred_i[mask & mask_B] == j)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] == j).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] == j).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] == j) * (pred_i[mask & mask_C] == j)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] == j).sum()
            score_long[j][2] += (pred_i[mask & mask_C] == j).sum()


    results.append(score)
    results.append(score_short)
    results.append(score_medium)
    results.append(score_long)

    return np.stack(results, axis=0)


def evaluation_octree_semantic_new(pred_occ, pred_octree, gt_occ, img_metas, class_num):
    results = []

    pred_volume = torch.zeros((200, 200, 16), dtype=torch.int64)
    pred_volume = pred_volume.cuda()
    volume_gt = torch.zeros((200, 200, 16), dtype=torch.int64)
    volume_gt = volume_gt.cuda()

    octree_gt = Octree(depth=4, pc_range=[-50, -50, -5.0, 50, 50, 3.0], occ_size=[25, 25, 2])
    octree_gt.cuda()
    octree_gt.build_octree(torch.squeeze(gt_occ, 0), img_metas[0]['build_octree'], img_metas[0]['build_octree_up'], 17)

    for i in range(len(pred_occ)):
        _, pred = torch.max(torch.softmax(pred_occ[i], dim=1), dim=1)
        mask = (pred != 0) * (pred != 17)
        pred = pred[mask]
        key = pred_octree.keys[i][mask]
        x, y, z = pred_octree.key2xyz(key, i)
        pred_value = torch.stack([x, y, z, pred], dim=1)
        pred_volume = octree_to_voxel_new(pred_value, pred_volume, i)

    for i in range(4):
        pred = octree_gt.gt[i]
        mask = (pred != 0) * (pred != 17)
        pred = pred[mask]
        key = octree_gt.keys[i][mask]
        x, y, z = octree_gt.key2xyz(key, i)
        gt_value = torch.stack([x, y, z, pred], dim=1)
        volume_gt = octree_to_voxel_new(gt_value, volume_gt, i)

    gt_i, pred_i = volume_gt.cpu().numpy(), pred_volume.cpu().numpy()
    mask = (gt_i != 255)

    dist_low2, dist_low1, dist_high1, dist_high2 = distance_table2[0], distance_table2[1], distance_table2[2], distance_table2[3]
    mask_A = np.zeros((200, 200, 16), dtype=np.bool)
    mask_B = np.zeros((200, 200, 16), dtype=np.bool)
    mask_A[dist_low1:dist_high1, dist_low1:dist_high1, :] = True
    mask_B[dist_low2:dist_high2, dist_low2:dist_high2, :] = True
    mask_B = mask_B & ~mask_A
    mask_C = ~(mask_A | mask_B)

    score = np.zeros((class_num, 3))
    score_short = np.zeros((class_num, 3))
    score_medium = np.zeros((class_num, 3))
    score_long = np.zeros((class_num, 3))

    for j in range(class_num):
        if j == 0: #class 0 for geometry IoU
            score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
            score[j][1] += (gt_i[mask] != 0).sum()
            score[j][2] += (pred_i[mask] != 0).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] != 0) * (pred_i[mask & mask_A] != 0)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] != 0).sum()
            score_short[j][2] += (pred_i[mask & mask_A] != 0).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] != 0) * (pred_i[mask & mask_B] != 0)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] != 0).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] != 0).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] != 0) * (pred_i[mask & mask_C] != 0)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] != 0).sum()
            score_long[j][2] += (pred_i[mask & mask_C] != 0).sum()

        else:
            score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
            score[j][1] += (gt_i[mask] == j).sum()
            score[j][2] += (pred_i[mask] == j).sum()

            score_short[j][0] += ((gt_i[mask & mask_A] == j) * (pred_i[mask & mask_A] == j)).sum()
            score_short[j][1] += (gt_i[mask & mask_A] == j).sum()
            score_short[j][2] += (pred_i[mask & mask_A] == j).sum()

            score_medium[j][0] += ((gt_i[mask & mask_B] == j) * (pred_i[mask & mask_B] == j)).sum()
            score_medium[j][1] += (gt_i[mask & mask_B] == j).sum()
            score_medium[j][2] += (pred_i[mask & mask_B] == j).sum()

            score_long[j][0] += ((gt_i[mask & mask_C] == j) * (pred_i[mask & mask_C] == j)).sum()
            score_long[j][1] += (gt_i[mask & mask_C] == j).sum()
            score_long[j][2] += (pred_i[mask & mask_C] == j).sum()


    results.append(score)
    results.append(score_short)
    results.append(score_medium)
    results.append(score_long)

    return np.stack(results, axis=0)