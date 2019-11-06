import numpy as np
import math
import scipy.misc
import cv2
import matplotlib.pyplot as plt

def getTightBox(label):
    # tightest bounding box covering the joint positions
    tBox = {}
    tBox['x_min'] = min(label[0,:])
    tBox['y_min'] = min(label[1,:])
    tBox['x_max'] = max(label[0,:])
    tBox['y_max'] = max(label[1,:])
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
    fg = np.where(label>100)
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
    tBox = getTightmask(label)
    return max(tBox['humHeight'] / 400, tBox['humWidth'] / 400)

def getCenter(label):
    tBox = getTightmask(label)
    center = np.zeros(2)
    center[0] = tBox['x_min'] + tBox['humWidth']/2
    center[1] = tBox['y_min'] + tBox['humHeight']/2
    return center

def rnd(x):
    return max(-2*x, min(2*x, np.random.normal(0,1)*x))

def getTransform(center,scale,rot,res):
    h = 200 * scale
    t = np.eye(3)
    t[0,0] = res/h
    t[1,1] = res/h

    t[0,2] = res * (-center[0] / h + 0.5)
    t[1,2] = res * (-center[1] / h + 0.5)

    if rot != 0:
        rot = -rot
        r = np.eye(3)
        ang = rot * math.pi / 180
        s = math.sin(ang)
        c = math.cos(ang)
        r[0,0] = c
        r[0,1] = -s
        r[1,0] = s
        r[1,1] = c

        t_ = np.eye(3)
        t_[0,2] = -res/2
        t_[1,2] = -res/2
        t_inv = np.eye(3)
        t_inv[0,2] = res/2
        t_inv[1,2] = res/2
        t = t_inv * r * t_ * t
    return t

def transform(pt,center,scale,rot,res,invert):
    pt_ = np.ones(3)
    pt_[0] = pt[0]-1
    pt_[1] = pt[1]-1

    t = getTransform(center,scale,rot,res)
    #print(t)
    if invert:
        t = np.linalg.inv(t)
    #print(t)
    new_point = (t.dot(pt_))[0:2]
    new_point = new_point + 1e-4
    #print(new_point)
    return new_point.astype(np.int64)


def crop(img, center, scale, rot, res, method):
    ndim = len(img.shape)
    if ndim == 2:   #if grayscale(depth)
        img = np.reshape(img,(img.shape[0],img.shape[1],1))
    ht,wd = img.shape[0],img.shape[1]
    tmpImg = img
    newImg = np.zeros([res,res,img.shape[2]])

    scalefactor = (200 * scale)/res
    if scalefactor < 2.0:
        scalefactor = 1.0
    else:
        newsize = math.floor(max(ht,wd)/scalefactor)
        if newsize < 2:
            if ndim == 2:
                newImg = np.reshape(newImg,(newImg.shape[0],newImg.shape[1]))
                return newImg
            else:
                sf = newsize / max(img.shape[0],img.shape[1])
                tmpImg = scipy.misc.imresize(tmpImg,sf,method)
                ht = tmpImg.shape[0]
                wd = tmpImg.shape[1]
    c,s = center/scalefactor , scale/scalefactor
    ul = transform([1,1],c,s,0,res,True)
    br = transform([res+1,res+1],c,s,0,res,True)
    if scalefactor >= 2:
        br = br - (br + ul - res)
    pad = np.ceil(np.linalg.norm(ul-br)/2-(br[0]-ul[1])/2).astype(np.int8)
    if rot != 0 :
        ul = ul-pad
        br = br+pad
    if(br[1]<=ul[1] or br[0]<=ul[0]):
        return
    newImg = np.zeros([br[1]-ul[1],br[0]-ul[0],img.shape[2]])

    newImg = tmpImg[max(0,ul[1]):min(br[1], ht+1), max(0,ul[0]) : min(br[0],wd+1)-1,:]
    newImg = newImg[max(0,2-ul[1]):min(br[1],ht+1)-ul[1], max(1,2-ul[0]):min(br[0],wd+1)-ul[0],:]

    if rot != 0:
        #newImg = scipy.misc.imrotate(newImg, rot * math.pi / 180.0, interp=method)
        newImg = scipy.ndimage.rotate(newImg, rot, reshape=False, order=0).astype(np.uint8)
        newImg = newImg[pad+1:newImg.shape[0]-pad,pad+1:newImg.shape[1]-pad,:]

    if scalefactor < 2:
        newImg = newImg
    if ndim == 2:
        newImg = newImg.reshape(newImg.shape[0],newImg.shape[1])
    return newImg.astype(np.uint8)

def cropfor3d(img, center, scale, rot, res, method):

    ul = transform([1,1],center,scale,0,res,True)
    br = transform([res,res],center,scale,0,res,True)
    pad = np.floor(np.linalg.norm(ul-br)/2-(br[0]-ul[1])/2).astype(np.int64)

    if rot != 0 :
        ul = ul-pad
        br = br+pad

    # if(br[1]<=ul[1] or br[0]<=ul[0]):
    #     return


    if len(img.shape) > 2:
        newDim = [br[1] - ul[1], br[0] - ul[0], img.shape[2]]
        newImg = np.zeros([newDim[0],newDim[1],newDim[2]])
        ht = img.shape[0]
        wd = img.shape[1]
    else:
        newDim = [br[1] - ul[1], br[0] - ul[0]]
        newImg = np.zeros([newDim[0],newDim[1]])
        ht = img.shape[0]
        wd = img.shape[1]



    newX = [max(1, -ul[0] + 2), min(br[0], wd+1) - ul[0]]
    newY = [max(1, -ul[1] + 2), min(br[1], ht+1) - ul[1]]
    oldX = [max(1, ul[0]), min(br[0], wd+1) - 1]
    oldY = [max(1, ul[1]), min(br[1], ht+1) - 1]


    if len(newDim) > 2:
        newImg[newY[0]-1:newY[1]-1,newX[0]-1:newX[1]-1,:] = img[oldY[0] -1 : oldY[1] -1, oldX[0] -1 : oldX[1] -1, : ]
        newImg = scipy.misc.imresize(newImg, [res, res], interp='bilinear')
    else:
        newImg[newY[0]-1:newY[1]-1,newX[0]-1:newX[1]-1] = img[oldY[0] - 1: oldY[1] - 1, oldX[0] - 1: oldX[1] - 1]
        newImg = newImg.astype(np.int32)
        #print(np.max(newImg))
        newImg = cv2.resize(newImg,(res,res),interpolation=cv2.INTER_NEAREST)
        #print(np.max(newImg))
    return newImg

def cropdepth(img, center, scale, rot, res):

    ul = transform([1,1],center,scale,0,res,True)
    br = transform([res,res],center,scale,0,res,True)
    pad = np.floor(np.linalg.norm(ul-br)/2-(br[0]-ul[1])/2).astype(np.int64)

    if rot != 0 :
        ul = ul-pad
        br = br+pad

    # if(br[1]<=ul[1] or br[0]<=ul[0]):
    #     return


    newDim = [br[1] - ul[1], br[0] - ul[0]]
    newImg = np.zeros([newDim[0],newDim[1]])
    ht = img.shape[0]
    wd = img.shape[1]



    newX = [max(1, -ul[0] + 2), min(br[0], wd+1) - ul[0]]
    newY = [max(1, -ul[1] + 2), min(br[1], ht+1) - ul[1]]
    oldX = [max(1, ul[0]), min(br[0], wd+1) - 1]
    oldY = [max(1, ul[1]), min(br[1], ht+1) - 1]



    newImg[newY[0]-1:newY[1]-1,newX[0]-1:newX[1]-1] = img[oldY[0] - 1: oldY[1] - 1, oldX[0] - 1: oldX[1] - 1]
    newImg = newImg.astype(np.float32)
    newImg = cv2.resize(newImg,(res,res),interpolation=cv2.INTER_NEAREST)#scipy.misc.imresize(newImg, [res, res], interp='nearest')
    return newImg

def changeSegmIx(segm, s):
    out = np.zeros(segm.shape)
    for i in range(len(s)):
        out[np.where(segm == i+1)] = s[i]
    return out


def Drawgaussian2D(img,pt,sigma_2d):
    res2D = img.shape[0]
    temp = np.zeros([res2D,res2D])
    temp = drawgaussian2d(temp,pt,sigma_2d)
    return temp



def Drawguassian3D(vol, pt, z, sigma_2d, size_z):

    resZ , res2D = vol.shape[2], vol.shape[1]
    temp = np.zeros([res2D, res2D])
    temp = drawgaussian2d(temp, pt, sigma_2d)
    zun = gaussian1D(size_z)
    count = 0
    offset = 9
    z = z + offset
    zmin = np.int_(z - int(size_z / 2))
    zmax = np.int_(z + int(size_z / 2))

    for i in range(zmin,zmax):
        
        if i >= 0 and i < resZ :
            vol[:,:,i] = zun[count] * temp
        
        count = count + 1
    #print(np.sum(vol))
    return vol




def gaussian1D(size):
    gauss = np.zeros(size)
    center = math.ceil(size/2)
    amplitude = 1.0
    sigma = 0.25
    for i in range(size):
        gauss[i] = amplitude * math.exp(-(math.pow((i+1-center)/(sigma * size),2)/2))
    return gauss

def gaussian2D(size):
    gauss = np.zeros([size,size])
    center = math.ceil(size/2)
    amplitude = 1.0
    sigma = 0.25
    for i in range(size):
        for j in range(size):
            distance = np.linalg.norm([i+1 - center , j+1 - center])
            gauss[i,j] = amplitude * math.exp(-(math.pow( distance / (sigma * size), 2) / 2))
    return gauss

def drawgaussian2d(img,pt,sigma):
    ul = [math.floor(pt[0] - 3*sigma),math.floor(pt[1] - 3*sigma)]
    br = [math.floor(pt[0] + 3*sigma),math.floor(pt[1] + 3*sigma)]

    if(ul[1] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        return img
    size = 6 * sigma + 1

    g = gaussian2D(size)

    g_x = [max(0,-ul[0]),min(br[0],img.shape[1])-max(0,ul[0])+max(0,-ul[0]) + 1 ]
    g_y = [max(0,-ul[1]),min(br[1],img.shape[0])-max(0,ul[1])+max(0,-ul[1]) + 1 ]

    img_x = [max(0,ul[0]),min(br[0],img.shape[1]-1)+1]
    img_y = [max(0,ul[1]),min(br[1],img.shape[0]-1)+1]

    assert g_x[0]>=0 and g_y[0]>=0

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0]: g_y[1], g_x[0] : g_x[1]]

    return img

def quantize(depth, dPelvis, step, Zres):
    lowB = -(Zres - 1)/2
    upB = (Zres - 1)/2


    #offset = -lowB + 1

    fgix = np.where(depth < 1e3)

    nForeground = len(fgix)
    out = np.ones(depth.shape)
    out = out * (upB + 1)
    
    out[fgix] = np.clip(np.ceil((depth[fgix] - dPelvis)/step).astype(np.int8),lowB,upB)

    #out[fgix] = out[fgix] + offset

    return  out, nForeground

def relative(depth,dPelvis, step, Zres): #halfrange
    lowB = -(Zres - 1)/2 * step
    upB = (Zres - 1)/2 * step
    fgix = np.where(depth < 1e3)

    nForeground = len(fgix)
    out = np.ones(depth.shape,dtype=np.float32)

    out = out * ((Zres - 1)/2 + 1) * step
    #print(np.max(depth[fgix]))
    #print(np.min(depth[fgix]))


    out[fgix] = np.clip(depth[fgix]-dPelvis,lowB,upB)
    #print(np.max(out[fgix]))
    #print(np.min(out[fgix]))
    return out, nForeground

def extent(segm,seg_num):
    h,w = segm.shape
    out = np.zeros([h,w,seg_num])
    for i in range(0,seg_num):
        mask = np.zeros([h,w])
        mask[np.where(segm == i+1)] = 1
        out[:,:,i] = mask
        # plt.figure()
        # fig = plt.imshow(mask, aspect='auto', cmap=plt.get_cmap('Set2'))
        # plt.show()
    return out