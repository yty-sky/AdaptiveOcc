# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional, Union

# 假设坐标是 X1Y1Z1_x2y2z1_x3y3z3_x4y4z4
# 其中 (X1,Y1,Z1)的取值范围是(0~127, 0~127, 0~9) -> occ_size
# 其中 (x2,y2,z2)、(x3,y3,z3)、(x4,y4,z4)的取值范围是(0~1, 0~1, 0~1)


class KeyLUT:
  def __init__(self, occ_size):

    self.occ_size = occ_size
    r256 = torch.arange(256, dtype=torch.int64)
    r512 = torch.arange(512, dtype=torch.int64)
    zero = torch.zeros(256, dtype=torch.int64)

    device = torch.device('cpu')
    self._encode = {device: (self.xyz2key(r256, zero, zero, 8),
                             self.xyz2key(zero, r256, zero, 8),
                             self.xyz2key(zero, zero, r256, 8))}
    self._decode = {device: self.key2xyz(r512, 9)}

  def encode_lut(self, device=torch.device('cpu')):
    if device not in self._encode:
      cpu = torch.device('cpu')
      self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
    return self._encode[device]

  def decode_lut(self, device=torch.device('cpu')):
    if device not in self._decode:
      cpu = torch.device('cpu')
      self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
    return self._decode[device]

  def xyz2key(self, x, y, z, depth):
    key = torch.zeros_like(x)
    for i in range(depth):
      mask = 1 << i
      key = (key | ((x & mask) << (2 * i + 2)) |
             ((y & mask) << (2 * i + 1)) |
             ((z & mask) << (2 * i + 0)))
    return key

  def key2xyz(self, key, depth):
    x = torch.zeros_like(key)
    y = torch.zeros_like(key)
    z = torch.zeros_like(key)
    for i in range(depth):
      x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
      y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
      z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
    return x, y, z

# remember to modify this, when you change the occ_size
# _key_lut = KeyLUT(occ_size=[50, 50, 4])
# _key_lut = KeyLUT(occ_size=[25, 25, 2])

def xyz2key(_key_lut: KeyLUT, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
            depth: int, b: Optional[Union[torch.Tensor, int]] = None):
  r'''Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys

  Args:
    x (torch.Tensor): The x coordinate.
    y (torch.Tensor): The y coordinate.
    z (torch.Tensor): The z coordinate.
    b (torch.Tensor or int): The batch index of the coordinates.
    depth (int): The depth of the shuffled key.
  '''

  EX, EY, EZ = _key_lut.encode_lut(x.device)
  occ_x, occ_y, occ_z = _key_lut.occ_size
  stride_x, stride_y, stride_z = occ_y * occ_z, occ_z, 1
  x, y, z = x.long(), y.long(), z.long()

  shift_depth = depth
  x0, y0, z0 = x >> shift_depth, y >> shift_depth, z >> shift_depth
  x1, y1, z1 = x - (x0 << shift_depth), y - (y0 << shift_depth), z - (z0 << shift_depth)
  key = (x0 * stride_x + y0 * stride_y + z0) << (shift_depth*3)

  mask = (1 << shift_depth) - 1  # 31  (1,1,1,1,1)
  key = key + (EX[x1 & mask] | EY[y1 & mask] | EZ[z1 & mask])

  return key


def key2xyz(_key_lut: KeyLUT, key: torch.Tensor, depth: int):
  r'''Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
  and the batch index based on pre-computed look up tables.

  Args:
    key (torch.Tensor): The shuffled key.
    depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
  '''

  DX, DY, DZ = _key_lut.decode_lut(key.device)
  occ_x, occ_y, occ_z = _key_lut.occ_size
  stride_x, stride_y, stride_z = occ_y * occ_z, occ_z, 1
  x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

  shift_depth = depth
  key_0 = key >> (shift_depth*3)
  x0 = torch.floor_divide(key_0, stride_x)
  y0 = torch.floor_divide(key_0 - x0 * stride_x, stride_y)
  z0 = torch.floor_divide(key_0 - x0 * stride_x -y0 * stride_y, 1)
  x0, y0, z0 = x0 << shift_depth, y0 << shift_depth, z0 << shift_depth

  key = key - (key_0 << (shift_depth*3))

  key = key & ((1 << 48) - 1)
  n = (depth + 2) // 3
  for i in range(n):
    k = key >> (i * 9) & 511
    x = x | (DX[k] << (i * 3))
    y = y | (DY[k] << (i * 3))
    z = z | (DZ[k] << (i * 3))

  x, y, z = x+x0, y+y0, z+z0
  return x, y, z
