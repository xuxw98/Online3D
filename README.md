# Online3D

Train and evaluate FCAF3D on ScanNet-SV:
```
```

Train and evaluate TD3D on ScanNet-SV:
```
```

Train and evaluate FCAF3D-Online on ScanNet-MV from scratch:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py 2 --work-dir work_dirs/online3d_v3FF
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py work_dirs/online3d_v3FF/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Train and evaluate TD3D-Online on ScanNet-MV from scratch:
```
```

Online-tuning: TODO.
