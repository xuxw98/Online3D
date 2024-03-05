### Prepare ScanNet-SV data for training monocular RGB-D perception model


**Step 1.** 
For ScanNet 2D data, the processed 2D data can be downloaded from [HERE](https://cloud.tsinghua.edu.cn/d/641cd2b7a123467d98a6/). Run `cat 2D.tar.* > 2D.tar` to merge the files. Then skip to Step 2.

Or you can process ScanNet 2D data yourself by following the steps below.
For 2D instance source data, run:
```
python download-scannet.py -o <ScanNet root> --type _2d-instance.zip
``` 

Then extract `_2d-instance.zip` into `2D_info` folder, whose structure follows: 

```
2D_info
└── scenexxxx_xx_2d-instance/instance/xxxxxx.png
```

Link the '2D_info' folder to this level of directory. 


Then process 2D instance data by:
```
python prepare_2d_ins.py --scannet_path ./2D_info --output_path ./2D --scene_index_file ./meta_data/scannet_train.txt
```


**Step 2.** For scannet_frames_25k data, run: 

```
python download-scannet.py -o <ScanNet root> --preprocessed_frames 
``` 

Link or move the 'scannet_frames_25k' folder to this level of directory.
Follow [votenet](https://github.com/facebookresearch/votenet/tree/main/scannet) to download 3D data. 
Link or move the 'scans' folder to this level of directory.


Then process SV data by running `python generate_18cls_gt_data.py`, which will create two folders named `scannet_sv_18cls_train` and `scannet_sv_18cls_val` here.


**Step 3.** Generate .pkl files by:
```
python tools/create_data.py scannet --root-path ./data/scannet-sv --out-dir ./data/scannet-sv --extra-tag scannet_sv
```

**Final folder structure:**

```
scannet-sv
├── README.md
├── scannet_frames_25k/
├── meta_data/
│   ├── scannet_train.txt
│   ├── scannetv2_train.txt
│   ├── scannetv2_val.txt
├── scans/
├── 2D_info/
├── 2D
│   ├── scenexxxx_xx
│   │   ├── instance
│   │   │   ├── xxxxxx.png
├── scannet_sv_18cls_train/
├── scannet_sv_18cls_val/
│   ├── scenexxxx_xx_xxxxxx_bbox.npy
│   ├── scenexxxx_xx_xxxxxx_ins_label.npy
│   ├── scenexxxx_xx_xxxxxx_pc.npy
│   ├── scenexxxx_xx_xxxxxx_pose.txt
│   ├── scenexxxx_xx_xxxxxx_sem_label.npy
│   ├── scenexxxx_xx_xxxxxx.jpg
│   ├── scenexxxx_xx_depth_intrinsic.txt
│   ├── scenexxxx_xx_image_intrinsic.txt
├── prepare_2d_ins.py
├── generate_18cls_gt_data.py
├── load_scannet_data.py
├── scannet_utils.py
├── scannet_sv_infos_train.pkl
└── scannet_sv_infos_val.pkl

```

