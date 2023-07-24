# Online3D

Train and evaluate FCAF3D on ScanNet-SV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/fcaf3d_sv/fcaf3d_scannet-3d-18class.py 2 --work-dir work_dirs/fcaf3d_sv
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/fcaf3d_sv/fcaf3d_scannet-3d-18class.py work_dirs/fcaf3d_sv/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Train and evaluate TD3D on ScanNet-SV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/td3d_sv/td3d_is_scannet-3d-18class.py 2 --work-dir work_dirs/td3d_sv
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/td3d_sv/td3d_is_scannet-3d-18class.py work_dirs/td3d_sv/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Train and evaluate FCAF3D-Online on ScanNet-MV from scratch:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py 2 --work-dir work_dirs/online3d_v3FF
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py work_dirs/online3d_v3FF/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Train and evaluate TD3D-Online on ScanNet-MV from scratch:
```
```

Finetune and evaluate FCAF3D (trained with SVFF) on ScanNet-MV:
```
```

Finetune and evaluate TD3D (trained with SVFF) on ScanNet-MV:
```
```


以上为工程部分（1、2、5、6）和之前做的方法部分（3、4），都弄完之后就可以做一下结合，把之前设计的方法用到SV-train MV-tuning的setting中。例如在SV上训练FCAF3D-FF（1），但是不直接在MV上finetune（5），而是给训好的FCAF3D-FF加上online的模块（如把conv模块当成0初始化的adapter插进网络，以及新增amodal head预测完整物体，设计帧间预测结果融合策略）再finetune
