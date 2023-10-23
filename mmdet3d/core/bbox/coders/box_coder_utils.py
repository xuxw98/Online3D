import numpy as np
import torch


class ResidualCoder(object):
    def __init__(self, code_size=6, **kwargs):
        super().__init__()
        self.code_size = code_size

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 6) [x, y, z, dx, dy, dz]
            anchors: (N, 6) [x, y, z, dx, dy, dz]

        Returns:

        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)

        return torch.cat([xt, yt, zt, dxt, dyt, dzt], dim=-1)

    def decode_torch(self, box_encodings, anchors):
    
        xa, ya, za, dxa, dya, dza = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, dxt, dyt, dzt = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa # xt: (xg - xa)/diagonal
        yg = yt * diagonal + ya
        zg = zt * dza + za # zt: (zg - za)/dza

        dxg = torch.exp(dxt) * dxa # dxt : log(dxg/dxa)
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        return torch.cat([xg, yg, zg, dxg, dyg, dzg], dim=-1)
