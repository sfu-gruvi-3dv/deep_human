import numpy as np
import params

class PointCloud():
    def __init__(self, depth, mask, fx = params.fx, fy = params.fy, cx = params.cx, cy = params.cy):
        self.data = np.zeros(depth.shape)
        self.mask = mask
        nx = depth.shape[2]
        ny = depth.shape[1]
        u, v = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        # x = u / 128
        # y = v / 128
        self.data = np.stack((x, y, depth), axis=-1)


    def get_normal(self, beta = 3, gamma = 30):
        normal = np.zeros(self.data.shape)
        for b in range(normal.shape[0]):
            for i in range(normal.shape[1]):
                for j in range(normal.shape[2]):
                    if self.mask[b, i, j]:
                        normal[b, i, j, :] = self.calc_normal(b, i, j, beta, gamma)
        return normal

    def calc_normal(self, b, i, j, beta, gamma):
        nx = self.data.shape[2]
        ny = self.data.shape[1]

        A = []
        for y in range(max(0, i - beta), min(i + beta + 1, ny)):
            for x in range(max(0, j - beta), min(j + beta + 1, nx)):
                if self.mask[b, y, x] and abs(self.data[b, y, x, 2] - self.data[b, i, j, 2]) < gamma:
                    A.append(self.data[b, y, x, :])
        A = np.array(A)
        if A.shape[0] < 3:
            return [0, 0, 0]
        tmp = np.matmul(np.linalg.pinv(A), np.ones((A.shape[0], 1)))
        tmp = np.transpose(tmp).squeeze()
        normal = tmp / np.linalg.norm(tmp)
        if normal[2] > 0:
            normal = -normal
        return normal