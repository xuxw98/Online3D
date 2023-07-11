### Processed Data
The processed data can be downloaded from xxx.


### Prepare ScanNet-MV data for online tuning

**Step 1.** Download ScanNet v2 data [HERE](https://github.com/ScanNet/ScanNet). Follow [votenet](https://github.com/facebookresearch/votenet/tree/main/scannet) to download 3D data. For 2D data, run:
```
python download-scannet.py -o <ScanNet root> --type .sens
python download-scannet.py -o <ScanNet root> --type _2d-label.zip
python download-scannet.py -o <ScanNet root> --type _2d-instance.zip
```
Link or move the 'scans' folder to this level of directory. 

**Step 2.** Process 3D data by running `python batch_load_scannet_data.py`, which will create a folder named `scannet_train_detection_data` here.

**Step 3.** Process 2D data from *.sens by:
```
python prepare_2d_data.py --scannet_path ./scans --output_path ./2D
```
Then unzip the 2D semantic and instance label by:
```
```
**Step 4.** Generate online data by:
```
python generate_online_data.py
```

**Step 5.** Generate .pkl files by:
```
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
├── scannet_mv_infos_train.pkl
├── scannet_mv_infos_val.pkl
└── scannet_mv_infos_test.pkl

```
