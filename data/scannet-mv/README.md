### Processed Data
The processed data can be downloaded from xxx.


### Prepare ScanNet-MV data for online tuning

**Step 1.** Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Follow [votenet](https://github.com/facebookresearch/votenet/tree/main/scannet) to download 3D data. 
Link or move the 'scans' folder to this level of directory. 

For 2D data, run:
```
python download-scannet.py -o <ScanNet root> --type .sens
python download-scannet.py -o <ScanNet root> --type _2d-label.zip
python download-scannet.py -o <ScanNet root> --type _2d-instance.zip
```

**Step 2.** Process 3D data by running `python batch_load_scannet_data.py`, which will create a folder named `scannet_train_detection_data` here.

**Step 3.** Extract the 2D label and instance zip file using its original zip name. 
The 2D info structure follows:

```
├── 2D_info
│   ├── scenexxxx_xx.sens
│   ├── scenexxxx_xx_2d-label/label/xxxxxx.png
│   ├── scenexxxx_xx_2d-instance/instance/xxxxxx.png
```


Link or move the '2D_info' folder to this level of directory. 
 
Process 2D data from 2D info by:
```
python prepare_2d_data.py --scannet_path ./2D_info --output_path ./2D --label_map_file ./meta_data/scannetv2-labels.combined.tsv --scene_index_file ./meta_data/scannet_train.txt
```

**Step 4.** Generate online data by:
```
python generate_online_data.py
```

**Step 5.** Generate .pkl files by:
```
python tools/create_data.py scannet --root-path ./data/scannet-mv --out-dir ./data/scannet-mv --extra-tag scannet_mv
```


**Final folder structure:**

```
scannet
├── meta_data/
├── batch_load_scannet_data.py
├── load_scannet_data.py
├── scannet_utils.py
├── README.md
├── scans/
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
│   │   │   ├── xxxxxx.txt
│   │   ├── point
│   │   │   ├── xxxxxx.npy
│   │   └── box_mask
│   │       ├── xxxxxx.npy
├── scannet_train_detection_data
│   ├── scenexxxx_xx_bbox.npy
├── instance_mask
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.npy
├── semantic_mask
│   ├── scenexxxx_xx
│   │   ├── xxxxxx.npy
├── scannet_mv_infos_train.pkl
├── scannet_mv_infos_val.pkl
└── scannet_mv_infos_test.pkl

```
