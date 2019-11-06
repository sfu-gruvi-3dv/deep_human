import numpy as np
import math
import scipy.misc
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def getTightBox(label):
    # tightest bounding box covering the joint positions
    tBox = {}
    tBox['x_min'] = min(label[0, :])
    tBox['y_min'] = min(label[1, :])
    tBox['x_max'] = max(label[0, :])
    tBox['y_max'] = max(label[1, :])
    tBox['humWidth'] = tBox['x_max'] - tBox['x_min']
    tBox['humHeight'] = tBox['y_max'] - tBox['y_min']
    # slightly larger area to cover the head/feat of the human

    tBox['x_min'] = tBox['x_min'] - 0.25 * tBox['humWidth']
    tBox['y_min'] = tBox['y_min'] - 0.35 * tBox['humHeight']
    tBox['x_max'] = tBox['x_max'] + 0.25 * tBox['humWidth']
    tBox['y_max'] = tBox['y_max'] + 0.25 * tBox['humHeight']
    tBox['humWidth'] = tBox['x_max'] - tBox['x_min'] + 1
    tBox['humHeight'] = tBox['y_max'] - tBox['y_min'] + 1
    return tBox


def getTightmask(label):
    # tightest bounding box covering the joint positions
    maxdis = np.max(label)
    fg = np.where(label > 100)
    tBox = {}
    tBox['x_min'] = np.min(fg[1])
    tBox['y_min'] = np.min(fg[0])
    tBox['x_max'] = np.max(fg[1])
    tBox['y_max'] = np.max(fg[0])
    tBox['humWidth'] = tBox['x_max'] - tBox['x_min']
    tBox['humHeight'] = tBox['y_max'] - tBox['y_min']
    # slightly larger area to cover the head/feat of the human

    tBox['x_min'] = tBox['x_min'] - 0.25 * tBox['humWidth']
    tBox['y_min'] = tBox['y_min'] - 0.1 * tBox['humHeight']
    tBox['x_max'] = tBox['x_max'] + 0.25 * tBox['humWidth']
    tBox['y_max'] = tBox['y_max'] + 0.1 * tBox['humHeight']
    tBox['humWidth'] = tBox['x_max'] - tBox['x_min'] + 1
    tBox['humHeight'] = tBox['y_max'] - tBox['y_min'] + 1
    return tBox


def getScale(label):
    tBox = getTightBox(label)
    return max(tBox['humHeight'] / 200, tBox['humWidth'] / 200)


def getCenter(label):
    tBox = getTightBox(label)
    center = np.zeros(2)
    center[0] = tBox['x_min'] + tBox['humWidth'] / 2
    center[1] = tBox['y_min'] + tBox['humHeight'] / 2
    return center


def getScale_detail(label):
    tBox = getTightmask(label)
    return max(tBox['humHeight'] / 200, tBox['humWidth'] / 200)


def getCenter_detail(label):
    tBox = getTightmask(label)
    center = np.zeros(2)
    center[0] = tBox['x_min'] + tBox['humWidth'] / 2
    center[1] = tBox['y_min'] + tBox['humHeight'] / 2
    return center


def rnd(x):
    return max(-2 * x, min(2 * x, np.random.normal(0, 1) * x))


def getTransform(center, scale, rot, res):
    h = 200 * scale
    t = np.eye(3)
    t[0, 0] = res / h
    t[1, 1] = res / h

    t[0, 2] = res * (-center[0] / h + 0.5)
    t[1, 2] = res * (-center[1] / h + 0.5)

    if rot != 0:
        rot = -rot
        r = np.eye(3)
        ang = rot * math.pi / 180
        s = math.sin(ang)
        c = math.cos(ang)
        r[0, 0] = c
        r[0, 1] = -s
        r[1, 0] = s
        r[1, 1] = c

        t_ = np.eye(3)
        t_[0, 2] = -res / 2
        t_[1, 2] = -res / 2
        t_inv = np.eye(3)
        t_inv[0, 2] = res / 2
        t_inv[1, 2] = res / 2
        t = t_inv * r * t_ * t
    return t


def transform(pt, center, scale, rot, res, invert):
    pt_ = np.ones(3)
    pt_[0] = pt[0] - 1
    pt_[1] = pt[1] - 1

    t = getTransform(center, scale, rot, res)
    # print(t)
    if invert:
        t = np.linalg.inv(t)
    # print(t)
    new_point = (t.dot(pt_))[0:2]
    new_point = new_point + 1e-4
    # print(new_point)
    return new_point.astype(np.int64)


def reverse_transform(pt, center, scale, rot, res, invert):
    pt_ = np.ones(3)
    pt_[0] = pt[0] - 1
    pt_[1] = pt[1] - 1

    t = getTransform(center, scale, rot, res)
    # print(t)
    if invert:
        t = np.linalg.inv(t)
    # print(t)
    new_point = (t.dot(pt_))[0:2]
    new_point = new_point + 1e-4
    # print(new_point)
    return new_point.astype(np.int64)


def crop(img, center, scale, rot, res, method):
    ndim = len(img.shape)
    if ndim == 2:  # if grayscale(depth)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    ht, wd = img.shape[0], img.shape[1]
    tmpImg = img
    newImg = np.zeros([res, res, img.shape[2]])

    scalefactor = (200 * scale) / res
    if scalefactor < 2.0:
        scalefactor = 1.0
    else:
        newsize = math.floor(max(ht, wd) / scalefactor)
        if newsize < 2:
            if ndim == 2:
                newImg = np.reshape(newImg, (newImg.shape[0], newImg.shape[1]))
                return newImg
            else:
                sf = newsize / max(img.shape[0], img.shape[1])
                tmpImg = scipy.misc.imresize(tmpImg, sf, method)
                ht = tmpImg.shape[0]
                wd = tmpImg.shape[1]
    c, s = center / scalefactor, scale / scalefactor
    ul = transform([1, 1], c, s, 0, res, True)
    br = transform([res + 1, res + 1], c, s, 0, res, True)
    if scalefactor >= 2:
        br = br - (br + ul - res)
    pad = np.ceil(np.linalg.norm(ul - br) / 2 - (br[0] - ul[1]) / 2).astype(np.int8)
    if rot != 0:
        ul = ul - pad
        br = br + pad
    if (br[1] <= ul[1] or br[0] <= ul[0]):
        return
    newImg = np.zeros([br[1] - ul[1], br[0] - ul[0], img.shape[2]])

    newImg = tmpImg[max(0, ul[1]):min(br[1], ht + 1), max(0, ul[0]): min(br[0], wd + 1) - 1, :]
    newImg = newImg[max(0, 2 - ul[1]):min(br[1], ht + 1) - ul[1], max(1, 2 - ul[0]):min(br[0], wd + 1) - ul[0], :]

    if rot != 0:
        # newImg = scipy.misc.imrotate(newImg, rot * math.pi / 180.0, interp=method)
        newImg = scipy.ndimage.rotate(newImg, rot, reshape=False, order=0).astype(np.uint8)
        newImg = newImg[pad + 1:newImg.shape[0] - pad, pad + 1:newImg.shape[1] - pad, :]

    if scalefactor < 2:
        newImg = newImg
    if ndim == 2:
        newImg = newImg.reshape(newImg.shape[0], newImg.shape[1])
    return newImg.astype(np.uint8)


def cropfor3d(img, center, scale, rot, res, method):
    ul = transform([1, 1], center, scale, 0, res, True)
    br = transform([res, res], center, scale, 0, res, True)
    pad = np.floor(np.linalg.norm(ul - br) / 2 - (br[0] - ul[1]) / 2).astype(np.int64)

    if rot != 0:
        ul = ul - pad
        br = br + pad

    # if(br[1]<=ul[1] or br[0]<=ul[0]):
    #     return


    if len(img.shape) > 2:
        newDim = [br[1] - ul[1], br[0] - ul[0], img.shape[2]]
        newImg = np.zeros([newDim[0], newDim[1], newDim[2]])
        ht = img.shape[0]
        wd = img.shape[1]
    else:
        newDim = [br[1] - ul[1], br[0] - ul[0]]
        newImg = np.zeros([newDim[0], newDim[1]])
        ht = img.shape[0]
        wd = img.shape[1]

    newX = [max(1, -ul[0] + 2), min(br[0], wd + 1) - ul[0]]
    newY = [max(1, -ul[1] + 2), min(br[1], ht + 1) - ul[1]]
    oldX = [max(1, ul[0]), min(br[0], wd + 1) - 1]
    oldY = [max(1, ul[1]), min(br[1], ht + 1) - 1]

    if len(newDim) > 2:
        newImg[newY[0] - 1:newY[1] - 1, newX[0] - 1:newX[1] - 1, :] = img[oldY[0] - 1: oldY[1] - 1,
                                                                      oldX[0] - 1: oldX[1] - 1, :]
        newImg = scipy.misc.imresize(newImg, [res, res], interp=method)
    else:
        newImg[newY[0] - 1:newY[1] - 1, newX[0] - 1:newX[1] - 1] = img[oldY[0] - 1: oldY[1] - 1,
                                                                   oldX[0] - 1: oldX[1] - 1]
        newImg = newImg.astype(np.int32)
        # print(np.max(newImg))
        newImg = cv2.resize(newImg, (res, res), interpolation=cv2.INTER_NEAREST)
        # print(np.max(newImg))
    return newImg


def cropdepth(img, center, scale, rot, res, paddingvalue=0.45):
    ul = transform([1, 1], center, scale, 0, res, True)
    br = transform([res, res], center, scale, 0, res, True)
    pad = np.floor(np.linalg.norm(ul - br) / 2 - (br[0] - ul[1]) / 2).astype(np.int64)

    if rot != 0:
        ul = ul - pad
        br = br + pad

    # if(br[1]<=ul[1] or br[0]<=ul[0]):
    #     return


    newDim = [br[1] - ul[1], br[0] - ul[0]]
    newImg = np.ones([newDim[0], newDim[1]]) * paddingvalue
    # newImg = np.zeros([newDim[0],newDim[1]])
    ht = img.shape[0]
    wd = img.shape[1]

    newX = [max(1, -ul[0] + 2), min(br[0], wd + 1) - ul[0]]
    newY = [max(1, -ul[1] + 2), min(br[1], ht + 1) - ul[1]]
    oldX = [max(1, ul[0]), min(br[0], wd + 1) - 1]
    oldY = [max(1, ul[1]), min(br[1], ht + 1) - 1]

    newImg[newY[0] - 1:newY[1] - 1, newX[0] - 1:newX[1] - 1] = img[oldY[0] - 1: oldY[1] - 1, oldX[0] - 1: oldX[1] - 1]
    newImg = newImg.astype(np.float32)
    newImg = cv2.resize(newImg, (res, res),
                        interpolation=cv2.INTER_NEAREST)  # scipy.misc.imresize(newImg, [res, res], interp='nearest')
    return newImg


def changeSegmIx(segm, s):
    out = np.zeros(segm.shape)
    for i in range(len(s)):
        out[np.where(segm == i + 1)] = s[i]
    return out


def changeSegmIxup3d(segm, s):
    out = np.zeros(segm.shape)
    for i in range(len(s)):
        out[np.where(segm == i)] = s[i]
    return out


def Drawgaussian2D(img, pt, sigma_2d):
    res2D = img.shape[0]
    temp = np.zeros([res2D, res2D])
    temp = drawgaussian2d(temp, pt, sigma_2d)
    return temp


def Drawguassian3D(vol, pt, z, sigma_2d, size_z):
    # print('pt: ', pt)
    resZ, res2D = vol.shape[2], vol.shape[1]
    temp = np.zeros([res2D, res2D])
    temp = drawgaussian2d(temp, pt, sigma_2d)
    # sum_2d = np.sum(temp)
    # if sum_2d >= 1e-2:
    #     temp /= sum_2d
    # else:
    #     temp *= 0
    # plt.figure()
    # plt.imshow(temp, aspect='auto', cmap=plt.get_cmap('jet'))
    # plt.show()
    # print(np.sum(temp))
    zun = gaussian1D(size_z)
    count = 0
    offset = 9
    z = z + offset
    zmin = np.int_(z - int(size_z / 2))
    zmax = np.int_(z + int(size_z / 2)) + 1

    for i in range(zmin, zmax):

        if i >= 0 and i < resZ:
            vol[:, :, i] = zun[count] * temp

        count = count + 1
    # print(np.sum(vol))
    # sum_3d = np.sum(vol)
    # if sum_3d > 1e-2:
    #     vol /= sum_3d
    # else:
    #     vol *= 0
    return vol


def gaussian1D(size):
    gauss = np.zeros(size)
    center = math.ceil(size / 2)
    amplitude = 1.0
    sigma = 0.25
    for i in range(size):
        gauss[i] = amplitude * math.exp(-(math.pow((i + 1 - center) / (sigma * size), 2) / 2))
    # gauss_sum = np.sum(gauss)
    # gauss /= gauss_sum
    return gauss


def gaussian2D(size):
    gauss = np.zeros([size, size])
    center = math.ceil(size / 2)
    amplitude = 1.0
    sigma = 0.25
    for i in range(size):
        for j in range(size):
            distance = np.linalg.norm([i + 1 - center, j + 1 - center])
            gauss[i, j] = amplitude * math.exp(-(math.pow(distance / (sigma * size), 2) / 2))
    return gauss


def drawgaussian2d(img, pt, sigma):
    ul = [math.floor(pt[0] - 3 * sigma), math.floor(pt[1] - 3 * sigma)]
    br = [math.floor(pt[0] + 3 * sigma), math.floor(pt[1] + 3 * sigma)]
    # print('img.shape:',img.shape)
    # print('ul , br', ul, br)
    if (ul[1] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # print('goes here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return img
    ## change 6 to 4 if we want small kel
    size = 6 * sigma + 1

    g = gaussian2D(size)

    g_x = [max(0, -ul[0]), min(br[0], img.shape[1] - 1) - max(0, ul[0]) + max(0, -ul[0]) + 1]
    g_y = [max(0, -ul[1]), min(br[1], img.shape[0] - 1) - max(0, ul[1]) + max(0, -ul[1]) + 1]

    img_x = [max(0, ul[0]), min(br[0], img.shape[1] - 1) + 1]
    img_y = [max(0, ul[1]), min(br[1], img.shape[0] - 1) + 1]

    assert g_x[0] >= 0 and g_y[0] >= 0

    # print('pt: ', pt)
    # print('img: ', img_y[0], img_y[1], img_x[0], img_x[1], ' g: ', g_y[0], g_y[1], g_x[0], g_x[1])
    # print('tosize: ',img[img_y[0] : img_y[1], img_x[0] : img_x[1]].shape, 'fromsize: ',g[g_y[0]: g_y[1], g_x[0] : g_x[1]].shape)
    img[img_y[0]: img_y[1], img_x[0]: img_x[1]] += g[g_y[0]: g_y[1], g_x[0]: g_x[1]]

    return img


def save3dheat(trunk, index):
    w, h, d = trunk.shape
    print('trunk shape:', w, h, d)
    f = open("heat3d_" + str(index) + "gt.ply", "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex " + str(w * h * d) + "\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    # img2 = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
    for i in range(w):
        for j in range(h):
            for k in range(d):
                f.write(str(i) + "  " + str(j) + "  " + str(k * (w / d)) + " " +
                        str(int(trunk[i, j, k] * 255.0)) + " " +
                        str(int(trunk[i, j, k] * 255.0)) + " " +
                        str(int(trunk[i, j, k] * 255.0)) + "\n")
    f.close()


def quantize(depth, dPelvis, step, Zres):
    lowB = -(Zres - 1) / 2
    upB = (Zres - 1) / 2

    fgix = np.where(depth < 1e3)

    out = np.ones(depth.shape)
    outctn = np.zeros(depth.shape)
    out = out * (upB + 1)

    out[fgix] = np.clip(np.ceil((depth[fgix] - dPelvis) / step).astype(np.int8), lowB, upB)
    outctn[fgix] = np.clip(depth[fgix] - dPelvis, -0.6, 0.6)

    out = out * -1
    outctn = outctn * -1
    outctn = (outctn + 0.6) / 1.2 * 19
    return out, outctn


def relative_up3d(depth, dPelvis, step, Zres):  # halfrange
    backdepth = max(depth[0, 0], depth[depth.shape[0] - 1, depth.shape[1] - 1])
    lowB = -(Zres + 1) / 2 * step
    upB = (Zres + 1) / 2 * step
    fgix = np.where(depth < (backdepth - 1e-2))

    nForeground = len(fgix)
    out = np.ones(depth.shape, dtype=np.float32)

    out = out * ((Zres - 1) / 2 + 1) * step

    out[fgix] = np.clip(depth[fgix] - dPelvis, lowB, upB)
    return out, nForeground


def relative(depth, dPelvis, step, Zres):  # halfrange
    lowB = -(Zres - 1) / 2 * step
    upB = (Zres - 1) / 2 * step
    fgix = np.where(depth < 1e3)

    nForeground = len(fgix)
    out = np.ones(depth.shape, dtype=np.float32)

    out = out * ((Zres - 1) / 2 + 1) * step
    # print(np.max(depth[fgix]))
    # print(np.min(depth[fgix]))


    out[fgix] = np.clip(depth[fgix] - dPelvis, lowB, upB)
    # print(np.max(out[fgix]))
    # print(np.min(out[fgix]))
    return out, nForeground


def extent(segm, seg_num):
    h, w = segm.shape
    out = np.zeros([h, w, seg_num])
    for i in range(0, seg_num):
        mask = np.zeros([h, w])
        mask[np.where(segm == i + 1)] = 1
        out[:, :, i] = mask
        # plt.figure()
        # fig = plt.imshow(mask, aspect='auto', cmap=plt.get_cmap('Set2'))
        # plt.show()
    return out


def up3dtosurreal(segm_raw):
    colortable = \
        [[84, 1, 68], [36, 223, 191], [51, 219, 167], [26, 225, 212],
         [25, 228, 233], [36, 231, 253], [137, 74, 61], [124, 178, 46],
         [139, 84, 57], [118, 185, 57], [134, 65, 65], [129, 55, 68],
         [124, 45, 70], [115, 34, 72], [106, 24, 71], [95, 12, 70],
         [130, 170, 36], [134, 163, 31], [137, 155, 30], [139, 148, 31],
         [141, 139, 34], [141, 132, 36], [140, 92, 53], [141, 100, 49],
         [142, 108, 46], [142, 117, 42], [142, 124, 39], [110, 192, 71],
         [101, 198, 87], [89, 205, 107], [78, 210, 126], [65, 215, 146]
         ]

    colorm = np.array(colortable, dtype=np.int32)
    colorhash = colorm[:, 0] * 1000000 + colorm[:, 1] * 1000 + colorm[:, 2]
    colordic = {}
    for i in range(colorhash.shape[0]):
        colordic[colorhash[i]] = i
    segm_raw = segm_raw.astype(np.int32)
    segmhash = segm_raw[:, :, 0] * 1000000 + segm_raw[:, :, 1] * 1000 + segm_raw[:, :, 2]

    segm_full = np.empty(segmhash.shape)
    for h in range(segmhash.shape[0]):
        for w in range(segmhash.shape[1]):
            # print(segm_raw[h,w,:], segmhash[h,w], w, h)
            segm_full[h, w] = colordic[segmhash[h, w]]

    # print(np.unique(segm_full))
    segm_full = changeSegmIxup3d(segm_full,
                                 [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 6, 6, 7, 7, 7, 8, 3, 3, 4, 4, 4, 5, 12, 13, 13, 13,
                                  14, 9, 10, 10, 10, 11])

    return segm_full


def estimate3D(W, Zrel, intrinsic, zroot):
    Z = Zrel + zroot
    S = computeSkel(W, Z, intrinsic)
    return S


def maxLocation(joints, center, scale, Res):
    H = scale * 200
    W = np.zeros(joints.shape, dtype=np.float32)
    W[0, :] = joints[1, :] / Res[0] * H + center[0] - 0.5 * H + 1
    W[1, :] = joints[0, :] / Res[1] * H + center[1] - 0.5 * H + 1
    return W


def computeSkel(W, Z, intrinsic):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    X = np.multiply(W[0, :] - cx, Z) / fx  # ((W[0,:] - cx). * Z)/ fx;
    Y = np.multiply(W[1, :] - cy, Z) / fy  # ((W[1,:] - cy). * Z)/ fy;
    X = X.reshape(1, 16)
    Y = Y.reshape(1, 16)
    Z = Z.reshape(1, 16)
    S = np.concatenate([X, Y, Z], axis=0)
    return S


def computeError(S, Sgt):
    diff = S - Sgt
    dist = np.mean(np.linalg.norm(diff, axis=0))
    return dist


def denormalize(jointsz):
    denormalizejoints = -1 * (jointsz / (19.0 / 1.2) - 0.6)
    # out = out * -1
    # outctn = outctn * -1
    # outctn = (outctn + 0.6) / 1.2 * 19
    return denormalizejoints


def getExtrinsicBlender(cam_loc):
    R_world2bcam = np.asarray([[0, 0, -1],
                               [0, -1, 0],
                               [1, 0, 0]], dtype=np.float32)
    T_world2bcam = -1 * R_world2bcam.dot(cam_loc)
    # print('R_world2bcam:\n',R_world2bcam)
    # print('T_world2bcam:\n',T_world2bcam)
    R_bcam2cv = np.asarray([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]], dtype=np.float32)
    # print('R_bcam2cv:\n',R_bcam2cv)
    R_world2cv = R_bcam2cv.dot(R_world2bcam)
    # print('R_world2cv:\n',R_world2cv)
    T_world2cv = R_bcam2cv.dot(T_world2bcam)
    # print('T_world2cv.shape: ', T_world2cv.shape)
    RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
    # print('RT:\n', RT)
    return RT


def eval_joints3d(expected_joints, joints_3d, intrinsic, zroot, center, scale, res, cam_loc):
    extrinsic = getExtrinsicBlender(cam_loc)
    W = maxLocation(expected_joints[0:2, :], center, scale, res)
    Zrel = denormalize(expected_joints[2, :])
    S = estimate3D(W, Zrel, intrinsic, zroot)
    rot_joints = extrinsic.dot(np.concatenate([joints_3d, np.ones([1, 16], dtype=np.float32)], axis=0))
    err = computeError(S, rot_joints)
    return err


def eval_joints3d_human36m(expected_joints, joints_3d, intrinsic, zroot, center, scale, res):
    W = maxLocation(expected_joints[0:2, :], center, scale, res)
    Zrel = denormalize(expected_joints[2, :])
    S = estimate3D(W, Zrel, intrinsic, zroot)
    err = computeError(S, joints_3d)
    return err


def Kinect_DepthNormalization(depthImage):
    row, col = depthImage.shape
    widthBound = row - 1
    heightBound = col - 1
    filledDepth = np.copy(depthImage)

    filterBlock5x5 = np.zeros([5, 5], dtype=np.int32)
    zeroPixels = 0
    for x in range(row):
        for y in range(col):
            if filledDepth[x, y] == 0:
                zeroPixels += 1
                p = 0
                for xi in range(-2, 3, 1):
                    q = 0
                    for yi in range(-2, 3, 1):
                        xSearch = x + xi
                        ySearch = y + yi
                        if (xSearch > 0 and xSearch < widthBound and ySearch > 0 and ySearch < heightBound):
                            filterBlock5x5[p, q] = filledDepth[xSearch, ySearch]
                        q = q + 1
                    p = p + 1

                X = np.sort(filterBlock5x5.reshape([25]))
                v = X[X > 0]
                if (len(v) == 0):
                    filledDepth[x, y] = 0
                else:
                    # print('np.diff(np.append(v,9999999))')
                    # print(np.diff(np.append(v,9999999)))
                    indices = np.where(np.diff(np.append(v, 9999999)) > 0)[0]
                    # print('indices:',indices)
                    # finding longest persistent length of repeated values
                    i = np.argmax(np.diff(np.append(0, indices)))
                    # print('indices:')
                    # print(indices)
                    # print('i:')
                    # print(i)
                    # print('indices[i]:')
                    # print(indices[i])
                    # The value that is repeated is the mode
                    mode = v[indices[i]]
                    # fill in the x,y value with the statistical mode of the values
                    # print('mode:')
                    # print(mode)
                    filledDepth[x, y] = mode
    return filledDepth


def compute_depth_err(batchindex, pred_depth_, depth_batch, mask_batch, datatype):
    res = 160
    batch_num = pred_depth_.shape[0]
    err = np.zeros([batch_num, res], dtype=np.float32)
    histmap = np.zeros([batch_num, res, 20], dtype=np.float32)
    histout = np.zeros([20], np.float32)
    bestoffset = np.zeros([batch_num], np.int32)
    output = {}

    for i in range(batch_num):
        mask = mask_batch[i].astype(np.float32)
        gtdepth = np.multiply(np.copy(depth_batch[i]), mask)
        pnum = np.sum(mask)

        median_offset = np.median(gtdepth[mask>0.5]) - np.median(pred_depth_[i][mask>0.5])


        for j in range(res):
            offset = (j - (res / 2)) * (0.40 / res)
            depth_shift = np.multiply((np.copy(pred_depth_[i]) + offset + median_offset).reshape([256, 256]), mask)
            errmap = np.absolute(depth_shift - gtdepth)
            histmap[i, j, :] = \
            np.histogram(errmap.reshape(256 * 256), bins=20, range=(0.0, 0.5), weights=mask.reshape(256 * 256),
                         density=False)[0]
            err[i, j] = np.sum(errmap) / pnum

    for i in range(batch_num):
        mask = mask_batch[i].astype(np.float32)
        gtdepth = np.multiply(np.copy(depth_batch[i]), mask)
        minloc = np.argmin(err[i, :])
        offset = (minloc - (res / 2)) * (0.2 / res)

        depth_shift = np.multiply((np.copy(pred_depth_[i]) + offset + median_offset).reshape([256, 256]), mask)
        errmap = np.absolute(depth_shift - gtdepth)

        norm = plt.Normalize(0.0, 0.2)
        colors = plt.cm.jet(norm(errmap))
        colors[mask<0.5] = 0

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.2)
        im = ax.imshow(errmap, cmap = 'jet', vmin = 0.0, vmax = 0.2)
        fig.colorbar(im, cax=cax, orientation='vertical')
        im.set_data(colors)
        plt.savefig('./heat_map/' + str(batchindex) + '_' + str(i) + ' ' + datatype + '.png')

        bestoffset[i] = minloc
        histout += histmap[i, minloc, :]

    output['err'] = np.mean(np.amin(err, axis=1))
    output['histmap'] = histout
    output['bestoffset'] = bestoffset
    return output



def depth2mesh(depth, mask, filename):
    h = depth.shape[0]
    w = depth.shape[1]
    f = open(filename + ".obj", "w")
    for i in range(h):
        for j in range(w):
            f.write('v '+str(float(2.0*i/h))+' '+str(float(2.0*j/w))+' '+str(float(depth[i,j]))+'\n')

    for i in range(h-1):
        for j in range(w-1):
            localpatch= np.copy(depth[i:i+2,j:j+2])
            dy = localpatch[0,:] - localpatch[1,:]
            dx = localpatch[:,0] - localpatch[:,1]
            dy = np.abs(dy)
            dx = np.abs(dx)
            if np.max(dx)<0.05 and np.max(dy) < 0.05 and mask[i,j]:
                f.write('f ' + str(int(i*h+j))+' '+ str(int(i*h+j+1))+' '+ str(int((i+1)*h+j))+'\n')
                f.write('f ' + str(int((i+1)*h+j+1))+' '+ str(int((i+1)*h+j))+' '+ str(int(i*h+j+1))+'\n')
    f.close()
    return