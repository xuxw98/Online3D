### Processed SceneNN Data
The processed SceneNN data can be downloaded from [HERE](https://cloud.tsinghua.edu.cn/d/dab5f9ea7f0c42f38364/).Run `cat SceneNN.tar.* > SceneNN.tar` to merge the files. Then skip to Step 4.


### Prepare SceneNN-MV data for semantic segmentation test

**Step 1.** Download SceneNN `oni` data from [HERE](https://hkust-vgd.ust.hk/scenenn/main/oni/). Then download `.ply` data, `.xml` data and `trajectory.log` data from [HERE](https://drive.google.com/drive/folders/0B2BQi-ql8CzecGxSeXNzYWNZQUk?resourcekey=0-0zdk0kE0OD1Vp848__ZTdQ).
The data of each scene is structured as follows:

```
SceneNN
├── 005
│   ├── 005.ply                 /* the reconstructed triangle mesh  */
│   ├── 005.xml                 /* the annotation                   */
├── trajectory
│   ├── 005_trajectory.log      /* camera pose (local to world)     */
└── 005.oni                     /* the raw RGBD video               */
```

For simplicity and representativeness, we select 12 highly comprehensive and object-rich scenes , namely ['015', '005', '030', '054', '322', '263', '243', '080', '089', '093', '096', '011'].


**Step 2.** 
Process `oni` data by using the tool in the playback folder [HERE](https://github.com/hkust-vgd/scenenn).


**Step 3.** Process online data by:
```
python data_process.py
```

**Step 4.** Generate .pkl files by:
```
python tools/create_data.py scenenn --root-path ./data/scenenn-mv --out-dir ./data/scenenn-mv --extra-tag scenenn_mv
```


**Final folder structure:**

```
scenenn-mv
├── data_process.py
├── quality_check.py
├── scannet_utils.py
├── README.md
├── scannetv2-labels.combined.tsv
├── 005.oni
├── trajectory
│   ├── 005_trajectory.log
│   ├── 011_trajectory.log
├── 005
│   ├── depth
│   │   ├── depthxxxxx.png
│   ├── image
│   │   ├── imagexxxxx.png
│   ├── label
│   │   ├── xxxxx.npy
│   ├── point
│   │   ├── xxxxx.npy
│   ├── pose
│   |   ├── xxxxx.npy
|   ├── 005.ply 
|   ├── 005.xml 
|   ├── timestamp.txt
├── SceneNN_validate.pkl
└── scenenn_mv_infos_val.pkl

```
