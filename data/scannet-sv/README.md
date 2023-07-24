### Processed Data
The processed data can be downloaded from xxx.

### Prepare ScanNet-SV data for training monocular RGB-D perception model 

**Step 1.** Download `scannet_frames_25k.zip` [HERE](https://github.com/ScanNet/ScanNet). Link or move the 'scannet_frames_25k' folder to this level of directory.

**Step 2.** Process SV data by running `python Test_GT_Maker/make_scannet_18cls_multi_thread.py`, which will create two folders named `scannet_sv_18cls_train` and `scannet_sv_18cls_val` here.


**Step 3.** Generate .pkl files by:
```
python tools/create_data.py scannet --root-path ./data/scannet-sv --out-dir ./data/scannet-sv --extra-tag scannet_sv
```

**Final folder structure:**

```
scannet-sv
├── README.md
├── scannet_frames_25k/
├── scannet_sv_18cls_train/
├── scannet_sv_18cls_val/
│   ├── scenexxxx_xx_xxxxxx_2d_bbox.npy
│   ├── scenexxxx_xx_xxxxxx_bbox.npy
│   ├── scenexxxx_xx_xxxxxx_ins_label.npy
│   ├── scenexxxx_xx_xxxxxx_pc.npy
│   ├── scenexxxx_xx_xxxxxx_pose.txt
│   ├── scenexxxx_xx_xxxxxx_sem_label.npy
│   ├── scenexxxx_xx_xxxxxx.jpg
│   ├── scenexxxx_xx_depth_intrinsic.txt
│   ├── scenexxxx_xx_image_intrinsic.txt
├── Test_GT_Maker/make_scannet_18cls_multi_thread.py
├── scannet_sv_infos_train.pkl
└── scannet_sv_infos_val.pkl

```

