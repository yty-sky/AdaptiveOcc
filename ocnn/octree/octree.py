# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from typing import Union, List

from ocnn.utils import meshgrid, scatter_add, cumsum, trunc_div
from .points import Points
from .shuffled_key import KeyLUT, xyz2key, key2xyz
from typing import Optional, Union

# 50m  0-10m  0.5  10-30m  1.0  30m-50m  2.0
distance_table = [39.5000, 79.5000, 119.5000, 159.5000]

class Octree:
  r''' Builds an octree from an input point cloud.

  Args:
    depth (int): The octree depth.
    occ_size (list): [
    batch_size (int): The octree batch size.
    device (torch.device or str): Choose from :obj:`cpu` and :obj:`gpu`.
        (default: :obj:`cpu`)

  .. note::
    The octree data structure requires that if an octree node has children nodes,
    the number of children nodes is exactly 8, in which some of the nodes are
    empty and some nodes are non-empty. The properties of an octree, including
    :obj:`keys`, :obj:`children` and :obj:`neighs`, contain both non-empty and
    empty nodes, and other properties, including :obj:`features`, :obj:`normals`
    and :obj:`points`, contain only non-empty nodes.

  .. note::
    The point cloud must be in range :obj:`[-1, 1]`.
  '''

  def __init__(self, depth: int, pc_range: list, occ_size: list, batch_size: int = 1,
               device: Union[torch.device, str] = 'cpu', **kwargs):
    super().__init__()

    self.depth = depth
    self.pc_range = pc_range
    self.occ_size = occ_size
    self.batch_size = batch_size
    self.device = device
    self.init_octree()
    self._key_lut = KeyLUT(occ_size=occ_size)

  def xyz2key(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            depth: int, b: Optional[Union[torch.Tensor, int]] = None):
    return xyz2key(self._key_lut, x, y, z, depth, b)

  def key2xyz(self, key: torch.Tensor, depth: int):
    return key2xyz(self._key_lut, key, depth)

  def init_octree(self):
    r''' Resets the Octree status and constructs several lookup tables.
    '''

    # octree features in each octree layers
    num = self.depth
    self.keys = [None] * num
    self.children = [None] * num
    self.neighs = [None] * num
    self.gt = [None] * num
    self.gt_aux = [None] * num

    # octree node numbers in each octree layers.
    # TODO: decide whether to settle them to 'gpu' or not?
    self.nnum = torch.zeros(num, dtype=torch.int32)
    self.nnum_nempty = torch.zeros(num, dtype=torch.int32)

    # the following properties are valid after `merge_octrees`.
    # TODO: make them valid after `octree_grow`, `octree_split` and `build_octree`
    batch_size = self.batch_size
    self.batch_nnum = torch.zeros(num, batch_size, dtype=torch.int32)
    self.batch_nnum_nempty = torch.zeros(num, batch_size, dtype=torch.int32)

    # construct the ooc level
    key = torch.arange(self.occ_size[0] * self.occ_size[1] * self.occ_size[2], dtype=torch.int64)
    self.keys[0] = key
    self.nnum[0] = self.occ_size[0] * self.occ_size[1] * self.occ_size[2]

    # construct the look up tables for neighborhood searching
    device = self.device
    center_grid = self.rng_grid(2, 3)    # (8, 3)
    displacement = self.rng_grid(-1, 1)  # (27, 3)
    neigh_grid = center_grid.unsqueeze(1) + displacement  # (8, 27, 3)
    parent_grid = trunc_div(neigh_grid, 2)
    child_grid = neigh_grid % 2
    self.lut_parent = torch.sum(parent_grid * torch.tensor([9, 3, 1], device=device), dim=2)
    self.lut_child = torch.sum(child_grid * torch.tensor([4, 2, 1], device=device), dim=2)

    # lookup tables for different kernel sizes
    self.lut_kernel = {
        '222': torch.tensor([13, 14, 16, 17, 22, 23, 25, 26], device=device),
        '311': torch.tensor([4, 13, 22], device=device),
        '131': torch.tensor([10, 13, 16], device=device),
        '113': torch.tensor([12, 13, 14], device=device),
        '331': torch.tensor([1, 4, 7, 10, 13, 16, 19, 22, 25], device=device),
        '313': torch.tensor([3, 4, 5, 12, 13, 14, 21, 22, 23], device=device),
        '133': torch.tensor([9, 10, 11, 12, 13, 14, 15, 16, 17], device=device),
    }

  def key(self, depth: int, nempty: bool = False):
    r''' Returns the shuffled key of each octree node.

    Args:
      depth (int): The depth of the octree.
      nempty (bool): If True, returns the results of non-empty octree nodes.
    '''

    key = self.keys[depth]
    if nempty:
      mask = self.nempty_mask(depth)
      key = key[mask]
    return key

  def xyzb(self, depth: int, nempty: bool = False):
    r''' Returns the xyz coordinates and the batch indices of each octree node.

    Args:
      depth (int): The depth of the octree.
      nempty (bool): If True, returns the results of non-empty octree nodes.
    '''

    key = self.key(depth, nempty)
    return self.key2xyz(key, depth)

  def batch_id(self, depth: int, nempty: bool = False):
    r''' Returns the batch indices of each octree node.

    Args:
      depth (int): The depth of the octree.
      nempty (bool): If True, returns the results of non-empty octree nodes.
    '''

    batch_id = self.keys[depth] >> 48
    if nempty:
      mask = self.nempty_mask(depth)
      batch_id = batch_id[mask]
    return batch_id

  def nempty_mask(self, depth: int):
    r''' Returns a binary mask which indicates whether the cooreponding octree
    node is empty or not.

    Args:
      depth (int): The depth of the octree.
    '''

    return self.children[depth] >= 0

  # 距离适应，建立一个表 [0, 10] 8  [10, 30] 6  [30, 50] 4
  # 第4层
  #
  #
  # 第1层： []


# x, y, z = (x+0.5) / self.occ_size[0], (y+0.5) / self.occ_size[1], (z+0.5) / self.occ_size[2]
# x_real = (x + 0.5) * (self.pc_range[3] - self.pc_range[0]) / (self.occ_size[0] * 2 ** i) + self.pc_range[0]
# y_real = (y + 0.5) * (self.pc_range[4] - self.pc_range[1]) / (self.occ_size[1] * 2 ** i) + self.pc_range[1]
# z_real = (z + 0.5) * (self.pc_range[5] - self.pc_range[2]) / (self.occ_size[2] * 2 ** i) + self.pc_range[2]
#
# x_real = torch.tensor([-30, -10, 10, 30])
# y_real = torch.tensor([-30, -10, 10, 30])
# x0 = (x_real - self.pc_range[0]) * (self.occ_size[0] * 2 ** i) / (self.pc_range[3] - self.pc_range[0]) - 0.5
# print(x0)
# y0 = (y_real - self.pc_range[1]) * (self.occ_size[1] * 2 ** i) / (self.pc_range[4] - self.pc_range[1]) - 0.5
# z0 = (z_real - self.pc_range[2]) * (self.occ_size[2] * 2 ** i) / (self.pc_range[5] - self.pc_range[2]) - 0.5

  def build_octree(self, occ_gt, build_octree, build_octree_up, num_classes):
    r''' Builds an octree from a point cloud.

    Args:
      point_cloud (Points): The input point cloud.

    .. note::
      Currently, the batch size of the point cloud must be 1.
    '''
    # 又构建过程我们发现，key很关键，是导出children，points等其他元素的关键
    # 主题思路是自底向上构建整个真值树

    # 距离适应，建立一个表  [0, 10] 6  [10, 30] 4  [30, 50] 3  0.25
    #                    [0, 10] 7  [10, 30] 5  [30, 50] 4  0.5
    #                    [0, 10] 8  [10, 30] 6  [30, 50] 5  1

    all_classes = num_classes + 1
    x, y, z = occ_gt[:, 0], occ_gt[:, 1], occ_gt[:, 2]
    key = self.xyz2key(x, y, z, self.depth - 1)
    # 对key值进行排序
    key, sorted_indice = torch.sort(key)
    gt = occ_gt[sorted_indice, 3]

    x, y, z = occ_gt[sorted_indice, 0], occ_gt[sorted_indice, 1], occ_gt[sorted_indice, 2]
    mask = torch.zeros_like(x, dtype=torch.int32)
    dist_low2, dist_low1, dist_high1, dist_high2 = distance_table[0], distance_table[1], distance_table[2], distance_table[3]
    # Rectangle A [-10, 10]
    mask_A = (x >= dist_low1) & (x <= dist_high1) & (y >= dist_low1) & (y <= dist_high1)
    mask[mask_A] = build_octree[0]
    # Rectangle B [-30, 30] / A
    mask_B = (x >= dist_low2) & (x <= dist_high2) & (y >= dist_low2) & (y <= dist_high2) & ~mask_A
    mask[mask_B] = build_octree[1]
    # Rectangle C [-50, 50] / (A, B)
    mask_C = ~(mask_A | mask_B)
    mask[mask_C] = build_octree[2]

    self.keys[self.depth - 1] = key.to(torch.int64)
    self.gt[self.depth - 1] = gt.to(torch.int32)

    for i in range(self.depth - 1, 0, -1):

      key = self.keys[i] >> 3
      # 从中剖分近似well和近似 not well
      node_key, idx, counts = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
      mask = torch.zeros_like(node_key, dtype=torch.int32).scatter_(0, idx, mask)

      # result_middle: [N,18], N是当前层尺度体素数目，18是语义数，哪一个体素是什么语义，就在对应的语义格子上置1
      # result: [M,18], M是上一层尺度体素数目，18是语义数，哪一个体素里面的八划分格子上有什么语义，就在对应的语义格子上加1

      result_middle = torch.zeros([key.shape[0], all_classes], dtype=torch.int32)
      result_middle = result_middle.to(mask.device)                              # 扩展17维度，融合self.gt[i]的数据和idx
      result = torch.zeros([node_key.shape[0], all_classes], dtype=torch.int32)
      result = result.to(mask.device)

      gt = self.gt[i]
      gt = gt.to(torch.int64)
      result_middle = result_middle.scatter_add_(1, torch.unsqueeze(gt, 1), torch.ones_like(torch.unsqueeze(key, 1), dtype=torch.int32))
      result = result.scatter_add_(0, idx.unsqueeze(1).expand(-1, all_classes), result_middle)
      max_counts, max_indices = torch.max(result, dim=1)
      max_indices = max_indices.to(torch.int32)

      mask = self.octree_grow_up(node_key, max_counts, max_indices, mask, i-1, build_octree_up, num_classes)

    for i in range(0, self.depth):
      self.gt[i][self.gt[i] == 0] = 255

    for i in range(0, self.depth):
      if i == 0:
        pkey = torch.arange(self.occ_size[0] * self.occ_size[1] * self.occ_size[2], device=self.device)
      else:
        mask_equal_17 = (self.gt[i-1] == all_classes-1)
        pkey = self.keys[i-1][mask_equal_17]
        pkey = (pkey.unsqueeze(-1) << 3) + torch.arange(8, device=key.device)
        pkey = pkey.view(-1)

      # 制作key与真值
      mask_isin_1 = torch.isin(pkey, self.keys[i])
      mask_isin_2 = torch.isin(self.keys[i], pkey)
      expanded_gt = torch.zeros_like(pkey, dtype=torch.int32)
      expanded_gt[mask_isin_1] = self.gt[i][mask_isin_2]

      self.keys[i] = pkey
      self.gt[i] = expanded_gt


  def build_octree_aux(self, occ_gt, build_octree, build_octree_up, num_classes):
    r''' Builds an octree from a point cloud.

    Args:
      point_cloud (Points): The input point cloud.

    .. note::
      Currently, the batch size of the point cloud must be 1.
    '''
    # 又构建过程我们发现，key很关键，是导出children，points等其他元素的关键
    # 主题思路是自底向上构建整个真值树

    # 距离适应，建立一个表  [0, 10] 6  [10, 30] 4  [30, 50] 3  0.25
    #                    [0, 10] 7  [10, 30] 5  [30, 50] 4  0.5
    #                    [0, 10] 8  [10, 30] 6  [30, 50] 5  1

    all_classes = num_classes + 1
    x, y, z = occ_gt[:, 0], occ_gt[:, 1], occ_gt[:, 2]
    key = self.xyz2key(x, y, z, self.depth - 1)
    # 对key值进行排序
    key, sorted_indice = torch.sort(key)
    gt = occ_gt[sorted_indice, 3]

    x, y, z = occ_gt[sorted_indice, 0], occ_gt[sorted_indice, 1], occ_gt[sorted_indice, 2]
    mask = torch.zeros_like(x, dtype=torch.int32)
    dist_low2, dist_low1, dist_high1, dist_high2 = distance_table[0], distance_table[1], distance_table[2], distance_table[3]
    # Rectangle A [-10, 10]
    mask_A = (x >= dist_low1) & (x <= dist_high1) & (y >= dist_low1) & (y <= dist_high1)
    mask[mask_A] = build_octree[0]
    # Rectangle B [-30, 30] / A
    mask_B = (x >= dist_low2) & (x <= dist_high2) & (y >= dist_low2) & (y <= dist_high2) & ~mask_A
    mask[mask_B] = build_octree[1]
    # Rectangle C [-50, 50] / (A, B)
    mask_C = ~(mask_A | mask_B)
    mask[mask_C] = build_octree[2]

    self.keys[self.depth - 1] = key.to(torch.int64)
    self.gt[self.depth - 1] = gt.to(torch.int32)
    self.gt_aux[self.depth - 1] = gt.to(torch.int32)

    for i in range(self.depth - 1, 0, -1):

      key = self.keys[i] >> 3
      # 从中剖分近似well和近似 not well
      node_key, idx, counts = torch.unique(key, sorted=True, return_inverse=True, return_counts=True)
      mask = torch.zeros_like(node_key, dtype=torch.int32).scatter_(0, idx, mask)

      # result_middle: [N,18], N是当前层尺度体素数目，18是语义数，哪一个体素是什么语义，就在对应的语义格子上置1
      # result: [M,18], M是上一层尺度体素数目，18是语义数，哪一个体素里面的八划分格子上有什么语义，就在对应的语义格子上加1

      result_middle = torch.zeros([key.shape[0], all_classes], dtype=torch.int32)
      result_middle = result_middle.to(mask.device)                              # 扩展17维度，融合self.gt[i]的数据和idx
      result = torch.zeros([node_key.shape[0], all_classes], dtype=torch.int32)
      result = result.to(mask.device)

      result_middle_aux = torch.zeros([key.shape[0], all_classes-1], dtype=torch.int32)
      result_middle_aux = result_middle_aux.to(mask.device)                              # 扩展17维度，融合self.gt[i]的数据和idx
      result_aux = torch.zeros([node_key.shape[0], all_classes-1], dtype=torch.int32)
      result_aux = result_aux.to(mask.device)

      gt = self.gt[i]
      gt = gt.to(torch.int64)
      result_middle = result_middle.scatter_add_(1, torch.unsqueeze(gt, 1), torch.ones_like(torch.unsqueeze(key, 1), dtype=torch.int32))
      result = result.scatter_add_(0, idx.unsqueeze(1).expand(-1, all_classes), result_middle)
      max_counts, max_indices = torch.max(result, dim=1)
      max_indices = max_indices.to(torch.int32)

      gt_aux = self.gt_aux[i]
      gt_aux = gt_aux.to(torch.int64)
      result_middle_aux = result_middle_aux.scatter_add_(1, torch.unsqueeze(gt_aux, 1), torch.ones_like(torch.unsqueeze(key, 1), dtype=torch.int32))
      result_aux = result_aux.scatter_add_(0, idx.unsqueeze(1).expand(-1, all_classes-1), result_middle_aux)
      max_counts_aux, max_indices_aux = torch.max(result_aux, dim=1)
      max_indices_aux = max_indices_aux.to(torch.int32)

      mask = self.octree_grow_up_aux(node_key, max_counts, max_indices, mask, i-1, build_octree_up, num_classes, max_indices_aux)

    for i in range(0, self.depth):
      self.gt[i][self.gt[i] == 0] = 255

    for i in range(0, self.depth):
      if i == 0:
        pkey = torch.arange(self.occ_size[0] * self.occ_size[1] * self.occ_size[2], device=self.device)
      else:
        mask_equal_17 = (self.gt[i-1] == all_classes-1)
        pkey = self.keys[i-1][mask_equal_17]
        pkey = (pkey.unsqueeze(-1) << 3) + torch.arange(8, device=key.device)
        pkey = pkey.view(-1)

      # 制作key与真值
      mask_isin_1 = torch.isin(pkey, self.keys[i])
      mask_isin_2 = torch.isin(self.keys[i], pkey)
      expanded_gt = torch.zeros_like(pkey, dtype=torch.int32)
      expanded_gt_aux = torch.zeros_like(pkey, dtype=torch.int32)
      expanded_gt[mask_isin_1] = self.gt[i][mask_isin_2]
      expanded_gt_aux[mask_isin_1] = self.gt_aux[i][mask_isin_2]

      self.keys[i] = pkey
      self.gt[i] = expanded_gt
      self.gt_aux[i] = expanded_gt_aux


  def octree_grow_full(self, depth: int, update_neigh: bool = True):
    r''' Builds the full octree, which is essentially a dense volumetric grid.

    Args:
      depth (int): The depth of the octree.
      update_neigh (bool): If True, construct the neighborhood indices.
    '''

    # check
    assert depth <= self.full_depth, 'error'

    # node number
    num = 1 << (3 * depth)
    self.nnum[depth] = num * self.batch_size  # 第n层节点数
    self.nnum_nempty[depth] = num * self.batch_size  # 第n层非空节点数

    # update key
    key = torch.arange(num, dtype=torch.long, device=self.device)
    bs = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
    key = key.unsqueeze(0) | (bs.unsqueeze(1) << 48)
    self.keys[depth] = key.view(-1)

    # update children
    self.children[depth] = torch.arange(
        num * self.batch_size, dtype=torch.int32, device=self.device)

    # update neigh if needed
    if update_neigh:
      self.construct_neigh(depth)

  def octree_split(self, split: torch.Tensor, depth: int):
    r''' Sets whether the octree nodes in :attr:`depth` are splitted or not.

    Args:
      split (torch.Tensor): The input tensor with its element indicating status
          of each octree node: 0 - empty, 1 - non-empty or splitted.
      depth (int): The depth of current octree.
    '''

    # split -> children
    empty = split == 0
    sum = cumsum(split, dim=0, exclusive=True)
    children, nnum_nempty = torch.split(sum, [split.shape[0], 1])
    children[empty] = -1

    # boundary case, make sure that at least one octree node is splitted
    if nnum_nempty == 0:
      nnum_nempty = 1
      children[0] = 0

    # update octree
    self.children[depth] = children
    self.nnum_nempty[depth] = nnum_nempty

  def octree_grow_up(self, key, counts, indices, mask, depth, build_octree_up, num_classes):

    counts = counts + mask
    counts_mask = (counts >= 8)
    indices_mask = (indices != num_classes)
    indices[~(counts_mask & indices_mask)] = num_classes  # 未满足聚合特性的置于17, not well (未满足聚合特性： 本身最多索引就是17，或者最多索引不是17，但count<8）
    if build_octree_up:
      mask = torch.clamp(mask - 1, min=0)
    else:
      mask = torch.clamp(mask, min=0)

    self.keys[depth] = key
    self.gt[depth] = indices

    return mask

  def octree_grow_up_aux(self, key, counts, indices, mask, depth, build_octree_up, num_classes, indices_aux):

    counts = counts + mask
    counts_mask = (counts >= 8)
    indices_mask = (indices != num_classes)
    indices[~(counts_mask & indices_mask)] = num_classes  # 未满足聚合特性的置于17, not well (未满足聚合特性： 本身最多索引就是17，或者最多索引不是17，但count<8）
    if build_octree_up:
      mask = torch.clamp(mask - 1, min=0)
    else:
      mask = torch.clamp(mask, min=0)

    self.keys[depth] = key
    self.gt[depth] = indices
    self.gt_aux[depth] = indices_aux

    return mask


  def octree_grow_down(self, depth: int, update_neigh: bool = True):
    r''' Grows the octree and updates the relevant properties. And in most
    cases, call :func:`Octree.octree_split` to update the splitting status of
    the octree before this function.

    Args:
      depth (int): The depth of the octree.
      update_neigh (bool): If True, construct the neighborhood indices.
    '''

    # node number
    nnum = self.nnum_nempty[depth-1] * 8
    self.nnum[depth] = nnum
    self.nnum_nempty[depth] = nnum

    # update keys
    key = self.key(depth-1, nempty=True)
    batch_id = (key >> 48) << 48
    key = (key & ((1 << 48) - 1)) << 3
    key = key | batch_id
    key = key.unsqueeze(1) + torch.arange(8, device=key.device)
    self.keys[depth] = key.view(-1)

    # update children
    self.children[depth] = torch.arange(
        nnum, dtype=torch.int32, device=self.device)

    # update neighs
    if update_neigh:
      self.construct_neigh(depth)

  def construct_neigh(self, depth: int):
    r''' Constructs the :obj:`3x3x3` neighbors for each octree node.

    Args:
      depth (int): The octree depth with a value larger than 0 (:obj:`>0`).
    '''

    if depth == 0:
      key = self.keys[0]
      x, y, z = self.key2xyz(key, depth)
      xyz = torch.stack([x, y, z], dim=-1)  # (N,  3)
      grid = self.rng_grid(min=-1, max=1)   # (27, 3)  构造邻域偏移网格
      xyz = xyz.unsqueeze(1) + grid         # (N, 27, 3)   获取每一个网格的邻域偏移网格
      xyz = xyz.view(-1, 3)                 # (N*27, 3)
      neigh = self.xyz2key(xyz[:, 0], xyz[:, 1], xyz[:, 2], depth=depth)  # 获得邻域的key
      bs = torch.arange(self.batch_size, dtype=torch.int32, device=self.device)
      neigh = neigh + bs.unsqueeze(1)  # (N*27,) + (B, 1) -> (B, N*27)
      bound = torch.Tensor([self.occ_size[0], self.occ_size[1], self.occ_size[2]]).unsqueeze(0)
      invalid = torch.logical_or((xyz < 0).any(1), (xyz >= bound).any(1))  # 超出边界的置为1
      neigh[:, invalid] = -1
      self.neighs[depth] = neigh.view(-1, 27)  # (B*N, 27)
    else:
      child_p = self.children[depth-1]
      nempty = child_p >= 0
      neigh_p = self.neighs[depth-1][nempty]   # (N, 27)  上一层的非空体素的邻域拿出来，因为当前体素肯定在上一层非空体素内
      neigh_p = neigh_p[:, self.lut_parent]    # (N, 8, 27)  当前8个体素的邻域所对应的上一层体素的keys的索引，小于0说明超出边界
      child_p = child_p[neigh_p]               # (N, 8, 27)  判断上一层体素是否为空
      invalid = torch.logical_or(child_p < 0, neigh_p < 0)   # (N, 8, 27)
      neigh = child_p * 8 + self.lut_child
      neigh[invalid] = -1
      self.neighs[depth] = neigh.view(-1, 27)

  def construct_all_neigh(self):
    r''' A convenient handler for constructing all neighbors.
    '''

    for depth in range(1, self.depth+1):
      self.construct_neigh(depth)

  def search_xyzb(self, query: torch.Tensor, depth: int, nempty: bool = False):
    r''' Searches the octree nodes given the query points.

    Args:
      query (torch.Tensor): The coordinates of query points with shape
          :obj:`(N, 4)`. The first 3 channels of the coordinates are :obj:`x`,
          :obj:`y`, and :obj:`z`, and the last channel is the batch index. Note
          that the coordinates must be in range :obj:`[0, 2^depth)`.
      depth (int): The depth of the octree layer. nemtpy (bool): If true, only
          searches the non-empty octree nodes.
    '''

    key = self.xyz2key(query[:, 0], query[:, 1], query[:, 2], query[:, 3], depth)
    idx = self.search_key(key, depth, nempty)
    return idx

  def search_key(self, query: torch.Tensor, depth: int, nempty: bool = False):
    r''' Searches the octree nodes given the query points.

    Args:
      query (torch.Tensor): The keys of query points with shape :obj:`(N,)`,
          which are computed from the coordinates of query points.
      depth (int): The depth of the octree layer. nemtpy (bool): If true, only
          searches the non-empty octree nodes.
    '''

    key = self.key(depth, nempty)
    # `torch.bucketize` is similar to `torch.searchsorted`.
    # I choose `torch.bucketize` here because it has fewer dimension checks,
    # resulting in slightly better performance according to the docs of
    # pytorch-1.9.1, since `key` is always 1-D sorted sequence.
    idx = torch.bucketize(query, key)

    valid = idx < key.shape[0]  # invalid if out of bound
    found = key[idx[valid]] == query[valid]
    valid[valid.clone()] = found
    idx[valid.logical_not()] = -1
    return idx

  def get_neigh(self, depth: int, kernel: str = '333', stride: int = 1,
                nempty: bool = False):
    r''' Returns the neighborhoods given the depth and a kernel shape.

    Args:
      depth (int): The octree depth with a value larger than 0 (:obj:`>0`).
      kernel (str): The kernel shape from :obj:`333`, :obj:`311`, :obj:`131`,
          :obj:`113`, :obj:`222`, :obj:`331`, :obj:`133`, and :obj:`313`.
      stride (int): The stride of neighborhoods (:obj:`1` or :obj:`2`). If the
          stride is :obj:`2`, always returns the neighborhood of the first
          siblings.
      nempty (bool): If True, only returns the neighborhoods of the non-empty
          octree nodes.
    '''

    if stride == 1:
      neigh = self.neighs[depth]
    elif stride == 2:
      # clone neigh to avoid self.neigh[depth] being modified
      neigh = self.neighs[depth][::8].clone()
    else:
      raise ValueError('Unsupported stride {}'.format(stride))

    if nempty:
      child = self.children[depth]
      if stride == 1:
        nempty_node = child >= 0
        neigh = neigh[nempty_node]
      valid = neigh >= 0
      neigh[valid] = child[neigh[valid]].long()  # remap the index

    if kernel == '333':
      return neigh
    elif kernel in self.lut_kernel:
      lut = self.lut_kernel[kernel]
      return neigh[:, lut]
    else:
      raise ValueError('Unsupported kernel {}'.format(kernel))

  def to_points(self, rescale: bool = True):
    r''' Converts averaged points in the octree to a point cloud.

    Args:
      rescale (bool): rescale the xyz coordinates to [-1, 1] if True.
    '''

    depth = self.depth
    batch_size = self.batch_size

    # by default, use the average points generated when building the octree
    # from the input point cloud
    xyz = self.points[depth]
    batch_id = self.batch_id(depth, nempty=True)

    # xyz is None when the octree is predicted by a neural network
    if xyz is None:
      x, y, z, batch_id = self.xyzb(depth, nempty=True)
      xyz = torch.stack([x, y, z], dim=1) + 0.5

    # normalize xyz to [-1, 1] since the average points are in range [0, 2^d]
    if rescale:
      scale = 2 ** (1 - depth)
      xyz = self.points[depth] * scale - 1.0

    # construct Points
    out = Points(xyz, self.normals[depth], self.features[depth],
                 batch_id=batch_id, batch_size=batch_size)
    return out

  def to(self, device: Union[torch.device, str], non_blocking: bool = False):
    r''' Moves the octree to a specified device.

    Args:
      device (torch.device or str): The destination device.
      non_blocking (bool): If True and the source is in pinned memory, the copy
          will be asynchronous with respect to the host. Otherwise, the argument
          has no effect. Default: False.
    '''

    if isinstance(device, str):
      device = torch.device(device)

    #  If on the save device, directly retrun self
    if self.device == device:
      return self

    def list_to_device(prop):
      return [p.to(device, non_blocking=non_blocking)
              if isinstance(p, torch.Tensor) else None for p in prop]

    # Construct a new Octree on the specified device
    octree = Octree(self.depth, self.pc_range, self.occ_size, self.batch_size, device)
    octree.keys = list_to_device(self.keys)
    octree.children = list_to_device(self.children)
    octree.neighs = list_to_device(self.neighs)
    octree.gt = list_to_device(self.gt)
    octree.gt_aux = list_to_device(self.gt)
    # octree.features = list_to_device(self.features)
    # octree.normals = list_to_device(self.normals)
    # octree.points = list_to_device(self.points)
    octree.nnum = self.nnum.clone()  # TODO: whether to move nnum to the self.device?
    octree.nnum_nempty = self.nnum_nempty.clone()
    octree.batch_nnum = self.batch_nnum.clone()
    octree.batch_nnum_nempty = self.batch_nnum_nempty.clone()
    self.device = device
    return octree

  def cuda(self, non_blocking: bool = False):
    r''' Moves the octree to the GPU. '''

    return self.to('cuda', non_blocking)

  def cpu(self):
    r''' Moves the octree to the CPU. '''

    return self.to('cpu')

  def rng_grid(self, min, max):
    r''' Builds a mesh grid in :obj:`[min, max]` (:attr:`max` included).
    '''

    rng = torch.arange(min, max+1, dtype=torch.long, device=self.device)
    grid = meshgrid(rng, rng, rng, indexing='ij')
    grid = torch.stack(grid, dim=-1).view(-1, 3)  # (27, 3)
    return grid


def merge_octrees(octrees: List['Octree']):
  r''' Merges a list of octrees into one batch.

  Args:
    octrees (List[Octree]): A list of octrees to merge.
  '''

  # init and check
  octree = Octree(depth=octrees[0].depth, full_depth=octrees[0].full_depth,
                  batch_size=len(octrees), device=octrees[0].device)
  for i in range(1, octree.batch_size):
    condition = (octrees[i].depth == octree.depth and
                 octrees[i].full_depth == octree.full_depth and
                 octrees[i].device == octree.device)
    assert condition, 'The check of merge_octrees failed'

  # node num
  batch_nnum = torch.stack(   # [6, 4]
      [octrees[i].nnum for i in range(octree.batch_size)], dim=1)
  batch_nnum_nempty = torch.stack(  # [6, 4]
      [octrees[i].nnum_nempty for i in range(octree.batch_size)], dim=1)
  octree.nnum = torch.sum(batch_nnum, dim=1)
  octree.nnum_nempty = torch.sum(batch_nnum_nempty, dim=1) # 做了加和
  octree.batch_nnum = batch_nnum
  octree.batch_nnum_nempty = batch_nnum_nempty    # batch也传进去
  nnum_cum = cumsum(batch_nnum_nempty, dim=1, exclusive=True)

  # merge octre properties
  for d in range(octree.depth+1):
    # key  一个疑问，为什么不像neigh那样表示，而是去加1<<48
    keys = [None] * octree.batch_size
    for i in range(octree.batch_size):
      key = octrees[i].keys[d] & ((1 << 48) - 1)  # clear the highest bits
      keys[i] = key | (i << 48) # 48这个操作，盲猜是假设最大深度不超过16层
    octree.keys[d] = torch.cat(keys, dim=0)  # 把一个batch中的所有keys拼接起来了，每过一个数据就加 1 << 48

    # children
    children = [None] * octree.batch_size
    for i in range(octree.batch_size):
      child = octrees[i].children[d].clone()  # !! `clone` is used here to avoid
      mask = child >= 0                       # !! modifying the original octrees
      child[mask] = child[mask] + nnum_cum[d, i]
      children[i] = child
    octree.children[d] = torch.cat(children, dim=0)   # 把一个batch中的所有children拼接起来了

    # features
    if octrees[0].features[d] is not None and d == octree.depth:
      features = [octrees[i].features[d] for i in range(octree.batch_size)]
      octree.features[d] = torch.cat(features, dim=0)

    # normals
    if octrees[0].normals[d] is not None and d == octree.depth:
      normals = [octrees[i].normals[d] for i in range(octree.batch_size)]
      octree.normals[d] = torch.cat(normals, dim=0)

    # points
    if octrees[0].points[d] is not None and d == octree.depth:
      points = [octrees[i].points[d] for i in range(octree.batch_size)]
      octree.points[d] = torch.cat(points, dim=0)

  return octree
