import numpy as np
import math
import tensorflow as tf

def transform_normal(normal):
    rgbnormal = np.zeros(normal.shape)
    rgbnormal[:, :, :, 0] = (normal[:, :, :, 0] + 1) / 2 * 255
    rgbnormal[:, :, :, 1] = (normal[:, :, :, 1] + 1) / 2 * 255
    rgbnormal[:, :, :, 2] = (-normal[:, :, :, 2] + 1) / 2 * 255
    return rgbnormal

def draw_normal_sphere(batchsize, r):
    square = -np.ones([2*r, 2*r, 3])
    centerx = r
    centery = r
    for i in range(2*r):
        for j in range(2*r):
            x = (j - centerx) / r
            y = (i - centery) / r
            if x**2 + y**2 < 1:
                z = -math.sqrt(1 - x**2 - y**2)
                square[i, j, :] = [x, y, z]
    square = np.expand_dims(square, 0)
    square = np.repeat(square, batchsize, axis=0)
    square = transform_normal(square)

    return square

def normalize(t):
    norm = np.linalg.norm(t, axis=-1)
    norm[norm == 0] = 10e-6
    norm = np.expand_dims(norm, -1)
    normed_t = t / norm
    return normed_t

def apply_mask(t, mask):
    canvas = np.zeros(t.shape)
    canvas[mask] = t[mask]
    return canvas

def naive_pcd(depth_batch):
    z = depth_batch
    batchsize = depth_batch.shape[0]
    nx = depth_batch.shape[2]
    ny = depth_batch.shape[1]
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    x = (x - nx/2) / (nx/2)
    y = (y - ny/2) / (ny/2)
    x = np.expand_dims(x, 0)
    y = np.expand_dims(y, 0)
    x = np.repeat(x, batchsize, axis=0)
    y = np.repeat(y, batchsize, axis=0)
    return np.stack((x, y, z), axis=-1)

def calc_normal(pcd, depth_batch, b, i, j, beta, gamma):
    nx = depth_batch.shape[2]
    ny = depth_batch.shape[1]

    A = []
    for y in range(max(0, i - beta), min(i + beta+1, ny)):
        for x in range(max(0, j - beta), min(j + beta+1, nx)):
            if abs(depth_batch[b, y, x] - depth_batch[b, i, j]) < gamma:
                A.append(pcd[b, y, x, :])
    A = np.array(A)
    if A.shape[0] < 3:
        return [0, 0, 0]
    tmp = np.matmul(np.linalg.pinv(A), np.ones((A.shape[0], 1)))
    tmp = np.transpose(tmp).squeeze()
    normal =  tmp / np.linalg.norm(tmp)
    if normal[2]> 0:
        normal = -normal
    return normal

def dump_pointcloud(pointcloud, name):
    f = open(name + ".ply", "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex " + str(pointcloud.shape[1]*pointcloud.shape[2]) + "\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    # img2 = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
    for i in range(pointcloud.shape[1]):
        for j in range(pointcloud.shape[2]):
            x = pointcloud[0, i, j, 0]
            y = pointcloud[0, i, j, 1]
            z = pointcloud[0, i, j, 2]

            f.write(str(x) + "  " + str(y) + "  " + str(z) + " " +
                    str(255) + " " +
                    str(255) + " " +
                    str(255) + "\n")

    f.close()


def depth_to_normal(depth_batch, beta=1, gamma=0.05):
    pcd = naive_pcd(depth_batch)
    dump_pointcloud(pcd, 'pointCloud_raw')
    normal = np.zeros(pcd.shape)
    for b in range(normal.shape[0]):
        for i in range(normal.shape[1]):
            for j in range(normal.shape[2]):
                normal[b, i, j, :] = calc_normal(pcd, depth_batch, b, i, j, beta, gamma)
    return normal

def smooth_depth(depth):
    bs, h, w = depth.get_shape().as_list()
    depth_ext = tf.expand_dims(depth, axis=3)
    smoothed = tf.image.resize_nearest_neighbor(depth_ext, [h // 8, w // 8])
    smoothed = tf.image.resize_nearest_neighbor(smoothed, [h, w])
    smoothed = tf.squeeze(smoothed, -1)
    return smoothed
