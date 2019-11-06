import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from PIL import Image

def draw3dskeleton(numjoints,Resz,heatmap3d):
    joints_link = a=np.asarray([range(16),[1,3,0,6,2,0,-1,4,5,8,8,8,10,11,12,13]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_xlim(0,20)
    ax.set_xlim(0,64)
    ax.set_ylim(0,64)
    #ax.set_zlim(0,64)
    ax.set_zlim(0,20)
    x = np.zeros(numjoints)
    y = np.zeros(numjoints)
    z = np.zeros(numjoints)

    #draw all the joints
    for i in range(numjoints):
        #jointtruck = heatmap3d[i*Resz:(i+1)*Resz,:,:]
        jointtruck = heatmap3d[:,:,i*Resz:(i+1)*Resz]
        x[i],y[i],z[i] = np.unravel_index(np.argmax(jointtruck, axis=None), jointtruck.shape)
        print(x[i],y[i],z[i])
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    #draw all the links
    for i in range(numjoints):
        start = a[0,i]
        end   = a[1,i]
        if end >= 0:
            ax.plot([x[start],x[end]], [y[start],y[end]], [z[start],z[end]])
    plt.show()


def draw3dskeleton_joints(numjoints, Resz, joints3d):
    joints_link = a = np.asarray([range(16), [1, 3, 0, 6, 2, 0, -1, 4, 5, 8, 8, 8, 10, 11, 12, 13]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim(0,20)
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    # ax.set_zlim(0,64)
    ax.set_zlim(0, 20)
    x = np.zeros(numjoints)
    y = np.zeros(numjoints)
    z = np.zeros(numjoints)

    # draw all the joints
    for i in range(numjoints):
        x[i] = int(joints3d[0,i])
        y[i] = int(joints3d[1,i])
        z[i] = int(joints3d[2,i])
        print(x[i], y[i], z[i])
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # draw all the links
    for i in range(numjoints):
        start = a[0, i]
        end = a[1, i]
        if end >= 0:
            ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]])
    plt.show()

def draw2dskeleton(img, heatmap2d):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    a = np.asarray([range(16),[1,3,0,6,2,0,-1,4,5,8,8,8,10,11,12,13]])
    
    x = np.zeros(16)
    y = np.zeros(16)
    
    for i in range(16):
        jointtruck = heatmap2d[:,:,i]
        x_, y_ = np.unravel_index(np.argmax(jointtruck, axis=None), jointtruck.shape)
        x[i] = int(x_/64*255)
        y[i] = int(y_/64*255)
        print(x[i],' ',y[i])
        img = cv2.circle(img, (int(y[i]),int(x[i])), 3, (0,0,255))
        
    for i in range(16):
        start = a[0,i]
        end   = a[1,i]
        if end >= 0:
            img = cv2.line(img,(int(y[start]), int(x[start])),
                                    (int(y[end]), int(x[end])),
                                         (255,0,0),5)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out = Image.fromarray(img)
    img_out.show()

def draw2dskeleton_joints(img, joints):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    a = np.asarray([range(16), [1, 3, 0, 6, 2, 0, -1, 4, 5, 8, 8, 8, 10, 11, 12, 13]])

    x = np.zeros(16)
    y = np.zeros(16)

    for i in range(16):
        x[i] = int(joints[0, i] * 4)
        y[i] = int(joints[1, i] * 4)
        print(x[i], ' ', y[i])
        img = cv2.circle(img, (int(y[i]), int(x[i])), 3, (0, 0, 255))

    for i in range(16):
        start = a[0, i]
        end = a[1, i]
        if end >= 0:
            img = cv2.line(img, (int(y[start]), int(x[start])),
                           (int(y[end]), int(x[end])),
                           (255, 0, 0), 5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out = Image.fromarray(img)
    img_out.show()

def draw2dskeleton_2(img, numjoints, Resz, heatmap2d):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    a = np.asarray([range(16), [1, 3, 0, 6, 2, 0, -1, 4, 5, 8, 8, 8, 10, 11, 12, 13]])

    x = np.zeros(16)
    y = np.zeros(16)

    for i in range(16):
        jointtruck = heatmap2d[:, :, i]
        x_, y_ = np.unravel_index(np.argmax(jointtruck, axis=None), jointtruck.shape)
        x[i] = int(x_ / 64 * 255)
        y[i] = int(y_ / 64 * 255)
        img = cv2.circle(img, (int(y[i]), int(x[i])), 3, (0, 0, 255))

    for i in range(16):
        start = a[0, i]
        end = a[1, i]
        if end >= 0:
            img = cv2.line(img, (int(y[start]), int(x[start])),
                           (int(y[end]), int(x[end])),
                           (255, 0, 0), 5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_out = Image.fromarray(img)
    img_out.show()

def show_prections(img, predictions):
    i = 0
    jointsnum = predictions.shape[0]
    for coord in range(jointsnum):
        if(True):
            keypt = (int(predictions[coord,0]), int(predictions[coord,1]))
            print(keypt)
            text_loc = (keypt[0]+5, keypt[1]+7)
            cv2.circle(img, keypt, 3, (55,255,155), -1)
            cv2.putText(img, str(coord), text_loc, cv2.FONT_HERSHEY_DUPLEX, 0.5, (55,255,155), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test():
    i=1