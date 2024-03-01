### Processed ScanNet 2D Data
The processed 2D data can be downloaded from [HERE](https://cloud.tsinghua.edu.cn/library/5127338d-074c-4e1e-b4e5-a8c9f18e94bf/Online3D/). Run `cat mmdet_xxx.tar.* > mmdet_xxx.tar` to merge the files.  
Then skip the following step 2 and step 3. 


### Prepare ScanNet-SV data for training monocular RGB-D perception model 

**Step 1.** Download `scannet_frames_25k.zip` [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scannet_frames_25k' folder to this level of directory.
Follow [votenet](https://github.com/facebookresearch/votenet/tree/main/scannet) to download 3D data. 
Link or move the 'scans' and 'meta_data' folder to this level of directory.

**Step 2.** 
For 2D instance source data, run:
```
python download-scannet.py -o <ScanNet root> --type _2d-instance.zip
``` 

Then extract `_2d-instance.zip` into `2D_info` folder, whose structure follows: 

```
2D_info
└── scenexxxx_xx_2d-instance/instance/xxxxxx.png
```

Link or move the '2D_info' folder to this level of directory. 

**Step 3.** 

Process 2D instance data by:
```
python prepare_2d_ins.py --scannet_path ./2D_info --output_path ./2D --scene_index_file ./meta_data/scannet_train.txt
```

**Step 4.**
Then process SV data by running `python generate_18cls_gt_data.py`, which will create two folders named `scannet_sv_18cls_train` and `scannet_sv_18cls_val` here.


**Step 5.** Generate .pkl files by:
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

