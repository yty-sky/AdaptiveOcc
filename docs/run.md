# Train and Test

Train AdaptiveOcc with 4 RTX3090 GPUs 
```
./tools/dist_train.sh ./projects/configs/adaptiveocc/adaptiveocc.py 4  ./work_dirs/adaptiveocc
```

Train AdaptiveOcc with 1 RTX3090 GPUs 
```
python ./tools/train.py ./projects/configs/adaptiveocc/adaptiveocc.py --work-dir ./work_dirs/output --deterministic --no-validate
```

Eval AdaptiveOcc with 4 RTX3090 GPUs
```
./tools/dist_test.sh ./projects/configs/adaptiveocc/adaptiveocc_inference.py ./path/to/ckpts.pth 4
```

Eval AdaptiveOcc with 1 RTX3090 GPUs
```
python ./tools/test.py ./projects/configs/adaptiveocc/adaptiveocc_inference.py ./path/to/ckpts.pth --deterministic --eval bbox
```

Visualize occupancy groundtruth:
```
python ./tools/visual.py $npy_path$
```

Visualize occupancy predictions, occupancy predictions and the multi-scale occupancy groundtruth:

First, you need to generate prediction results. Here we use whole validation set as an example.
```
./tools/dist_test.sh ./projects/configs/adaptiveocc/adaptiveocc_inference_vis.py ./path/to/ckpts.pth 4
# python ./tools/test.py ./projects/configs/adaptiveocc/adaptiveocc_inference_vis.py ./path/to/ckpts.pth --deterministic --eval bbox
```
You will get prediction results in './visual_dir'. You can directly use meshlab to visualize .ply files or run visual_octree.py to visualize raw .npy files with mayavi:
```
python ./tools/visual_octree.py visual_dir/$npy_path$
```

Visualize multi-scale occupancy groundtruth:
```
python ./tools/visual_octree.py visual_dir/$npy_path$ --is_gt
```

Visualize occupancy groundtruth:
```
python ./tools/visual.py $npy_path$
```