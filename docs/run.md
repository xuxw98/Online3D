
Train and evaluate MinkUNet on ScanNet-SV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/minkunet_sv/minkunetFF_scannet_seg-3d-20class.py 2 --work-dir work_dirs/minkunet_svFF
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/minkunet_sv/minkunetFF_scannet_seg-3d-20class.py work_dirs/minkunet_svFF/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Train and evaluate FCAF3D on ScanNet-SV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/fcaf3d_sv/fcaf3dFF_scannet-3d-18class.py 2 --work-dir work_dirs/fcaf3d_svFF
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/fcaf3d_sv/fcaf3dFF_scannet-3d-18class.py work_dirs/fcaf3d_svFF/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Train and evaluate TD3D on ScanNet-SV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/td3d_sv/td3dFF_is_scannet-3d-18class.py 2 --work-dir work_dirs/td3d_svFF
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/td3d_sv/td3dFF_is_scannet-3d-18class.py work_dirs/td3d_svFF/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Finetune and evaluate MinkUNet (trained with SVFF) on ScanNet-MV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/minkunet_online/minkunetFF_online_scannet_seg-3d-20class.py 2 --work-dir work_dirs/minkunet_online
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/minkunet_online/minkunetFF_online_scannet_seg-3d-20class.py work_dirs/minkunet_online/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Evaluate MinkUNet (fintuned on ScanNet-MV) on SceneNN-MV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/minkunet_online/minkunetFF_scenenn_seg-3d-16class.py work_dirs/minkunet_online/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```


Finetune and evaluate FCAF3D (trained with SVFF) on ScanNet-MV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py 2 --work-dir work_dirs/online3d_v3FF
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py work_dirs/online3d_v3FF/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Finetune and evaluate TD3D (trained with SVFF) on ScanNet-MV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/td3d_online/td3dFF_online_is_scannet-3d-18class.py 2 --work-dir work_dirs/td3d_online
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/td3d_online/td3dFF_online_is_scannet-3d-18class.py work_dirs/td3d_online/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```
