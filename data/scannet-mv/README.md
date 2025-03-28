### Prepare ScanNet-MV data for online tuning

**Step 1.** 
Prepare ScanNet 2D data. The processed 2D data can be downloaded from [HERE](https://cloud.tsinghua.edu.cn/d/f9dc0ae45f2f4666a209/). Run `cat 2D.tar.* > 2D.tar` to merge the files. Then skip to Step 2.

Or you can process ScanNet 2D data yourself by following the steps below.

First acquire `download-scannet.py` from [HERE](https://github.com/ScanNet/ScanNet). You should fill out an agreement to the ScanNet Terms of Use.

Then download 2D data, run:
```
python download-scannet.py -o <ScanNet root> --type .sens
python download-scannet.py -o <ScanNet root> --type _2d-label.zip
python download-scannet.py -o <ScanNet root> --type _2d-instance.zip
```

Extract `_2d-label.zip` and `_2d-instance.zip` into `2D_info` folder, whose structure follows: 

```
2D_info
├── scenexxxx_xx.sens
├── scenexxxx_xx_2d-label/label/xxxxxx.png
└── scenexxxx_xx_2d-instance/instance/xxxxxx.png
```

Link the `2D_info` folder to this level of directory. 

Process 2D data by:
```
python prepare_2d_data.py --scannet_path ./2D_info --output_path ./2D --label_map_file ./meta_data/scannetv2-labels.combined.tsv --scene_index_file ./meta_data/scannet_train.txt
```


**Step 2.** Prepare ScanNet 3D data. Follow [votenet](https://github.com/facebookresearch/votenet/tree/main/scannet) to download and process the 3D data. 
Link or move the `scans` folder to this level of directory. 

Generate online data by:
```
python generate_online_data.py
```

**Step 3.** Generate .pkl files by:
```
python tools/create_data.py scannet --root-path ./data/scannet-mv --out-dir ./data/scannet-mv --extra-tag scannet_mv
```


**Final folder structure:**

```
scannet-mv
├── meta_data/
├── batch_load_scannet_data.py
├── load_scannet_data.py
├── scannet_utils.py
├── README.md
├── scans/
├── 2D_info/
├── 2D
│   ├── scenexxxx_xx
│   │   ├── color
│   │   │   ├── xxxxxx.jpg
│   │   ├── depth
│   │   │   ├── xxxxxx.png
│   │   ├── label
│   │   │   ├── xxxxxx.png
│   │   ├── instance
│   │   │   ├── xxxxxx.png
│   │   ├── pose
│   │       ├── xxxxxx.txt
├── generate_online_data.py
├── instance_mask
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.npy
├── semantic_mask
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.npy
├── point
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.npy
├── modal_box
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.npy
├── amodal_box_mask
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.npy
├── scene_amodal_box
│   ├── scenexxxx_xx.npy
├── scannet_mv_infos_train.pkl
└── scannet_mv_infos_val.pkl

```
