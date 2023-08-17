# Online3D

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

Finetune and evaluate FCAF3D (trained with SVFF) on ScanNet-MV:
```
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_train.sh configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py 2 --work-dir work_dirs/online3d_v3FF
CUDA_VISIBLE_DEVICES=0,1 PORT=29544 bash ./tools/dist_test.sh configs/fcaf3d_online/online3dv3FF_8x2_scannet-3d-18class.py work_dirs/online3d_v3FF/latest.pth 2 --show-dir work_dirs/debug/ --eval x
```

Finetune and evaluate TD3D (trained with SVFF) on ScanNet-MV:
```
```


目前只维护FF版本，纯Depth版本不进行更新

本次更新确保了FCAF3D的SV和Online都可以跑通并得到正确结果
Method | mAP@0.25 | mAP@0.5 |
:--: |:--: | :--: |
Rec | 70.7 | 56.0
SV-->finetune | | 
Depth spatial from scratch | 67.7 | 46.7
SV-->Depth spatial finetume | 69.1 | 48.5
SV-->RGB temporal + Depth spatial finetune | |
SV-->temporal & spatial finetune + consist. loss | | 
SV-->temporal & spatial finetune + consist. loss + merge | |

TD3D的dataset/pipeline和model还需要进一步修改：
1. 兼容Det的基础上开发
2. backbone用MinkFFResNet，去掉NN（Det是MEFFResNet3D）
3. 各种数据增强时modal和amodal box也要一并操作