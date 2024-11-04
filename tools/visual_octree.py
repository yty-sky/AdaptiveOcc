import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='visual')
    parser.add_argument('file', help='npy file')
    parser.add_argument('--is_gt', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    colors = np.array(
        [
            [0, 0, 0, 255],       # empty
            [255, 120, 50, 255],  # barrier              orangey
            [255, 192, 203, 255],  # bicycle              pink
            [255, 255, 0, 255],  # bus                  yellow
            [0, 150, 245, 255],  # car                  blue
            [0, 255, 255, 255],  # construction_vehicle cyan
            [200, 180, 0, 255],  # motorcycle           dark orange
            [255, 0, 0, 255],  # pedestrian           red
            [255, 240, 150, 255],  # traffic_cone         light yellow
            [135, 60, 0, 255],  # trailer              brown
            [160, 32, 240, 255],  # truck                purple
            [255, 0, 255, 255],  # driveable_surface    dark pink
            [139, 137, 137, 255],
            [75, 0, 75, 255],  # sidewalk             dard purple
            [150, 240, 80, 255],  # terrain              light green
            [230, 230, 250, 255],  # manmade              white
            [0, 175, 0, 255],  # vegetation           green
            [255, 228, 181, 255],  # 17             dark cyan
            [178, 37, 123, 255],
            [0, 191, 255, 255]
        ]
    ).astype(np.uint8)

    voxel_size_init = 4
    pc_range = [-50, -50, -5, 50, 50, 3]
    visual_dir = f"{args.file}/"

    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))

    for i in range(4):

        if args.is_gt:
            visual_path = f"{visual_dir}gt_{i}.npy"
        else:
            visual_path = f"{visual_dir}pred_{i}.npy"

        fov_voxels = np.load(visual_path)
        fov_voxels = fov_voxels.astype(np.float64)

        fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
        fov_voxels = fov_voxels[fov_voxels[..., 3] != 255]
        fov_voxels = fov_voxels[fov_voxels[..., 3] != 17]

        voxel_size = voxel_size_init / (2 ** int(i))
        fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
        fov_voxels[:, 0] += pc_range[0]
        fov_voxels[:, 1] += pc_range[1]
        fov_voxels[:, 2] += pc_range[2]

        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="viridis",
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=19,
        )

        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    voxel_size = 0.25
    x = np.arange(0, 8)
    y = np.arange(0, 16)
    z = np.arange(0, 6)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
    x_grid = (x_grid.flatten() + 0.5) * voxel_size - 1
    y_grid = (y_grid.flatten() + 0.5) * voxel_size - 2
    z_grid = (z_grid.flatten() + 0.5) * voxel_size - 1.5
    label = np.full(x_grid.shape, 18)
    plt_plot_fov = mlab.points3d(
        x_grid,
        y_grid,
        z_grid,
        label,
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    mlab.show()


if __name__ == '__main__':
    main()

# python ./tools/visual_octree.py visual_dir/n015-2018-08-02-17-28-51+0800__LIDAR_TOP__1533202409949248
# python ./tools/visual_octree.py visual_dir/n015-2018-08-02-17-28-51+0800__LIDAR_TOP__1533202409949248 --is_gt
