from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import matplotlib.pyplot as plt
import math

imread = plt.imread
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    # from io import StringIO
    from io import BytesIO
    import PIL.Image
    img = PIL.Image.open(BytesIO(res))
    return np.asarray(img)

def read_npy(res):
    # from io import StringIO
    from io import BytesIO
    # import PIL.Image
    return np.load(BytesIO(res),encoding='bytes')
    # import StringIO
    # from io import StringIO
    # return np.load(StringIO(res))

def write_pose(x, y, z, roll, pitch, yaw, num=2, filename='pose.txt'):

    f = open(filename, 'w+')
    for i in range(len(x)):
        # f.write(filename,'w+')
        f.write('%6.4f %6.4f %6.4f\n' % (x[i], y[i], z[i]))
        f.write('%6.4f %6.4f %6.4f\n' % (roll[i], pitch[i], yaw[i]))

    f.close()

def eulerAnglesToRotationMatrix(theta):
    print('theta 0 ', theta[0], theta[1], theta[2])

    R_x = np.array([[1.,         0.,                  0.                   ],
                    [0.,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0.,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0.,      math.sin(theta[1])  ],
                    [0.,                     1.,      0.                   ],
                    [-math.sin(theta[1]),   0.,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0.],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0.],
                    [0.,                     0.,                      1.]
                    ])


    # R = np.dot(R_z, np.dot( R_y, R_x ))
    R = np.matmul(np.matmul( R_z, R_y ), R_x)

    return R

def DepthConversion(PointDepth, f):
    H = PointDepth.shape[0]
    W = PointDepth.shape[1]
    i_c = np.float(H) / 2 - 1
    j_c = np.float(W) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
    DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
    # PlaneDepth = (PointDepth-1) / (1 + (DistanceFromCenter / f)**2)**(0.5)
    PlaneDepth = (PointDepth) / (1 + (DistanceFromCenter / f)**2)**(0.5)
    return PlaneDepth

from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
print('res')
print(res)

'''Load a camera trajectory '''
# traj_file = './camera_traj2.json' # Relative to this python script
# traj_file = './camera_traj36.json' # Relative to this python script
traj_file = './camera_trans4.json' # Relative to this python script
import json
camera_trajectory = json.load(open(traj_file))
# We will show how to record a camera trajectory in another tutorial
print('Length Camera Trajectory.')
print(len(camera_trajectory))
''' Render Images '''
# for idx in range(len(camera_trajectory)):
# for idx in range(10):
idx = 0
# while True:
# for idx in range(len(camera_trajectory)):
# num_images = 18
# num_images = 18
num_images = 4
# x = []
# y = []
# z = []
# roll = []
# pitch = []
# yaw = []
for idx in range(num_images):
    loc, rot = camera_trajectory[idx]
    # print(loc)
    # print(rot)
    # print(type(loc))
    # x.append(loc['x'])
    # y.append(loc['y'])
    # z.append(loc['z'])
    # roll.append(rot['roll'])
    # pitch.append(rot['pitch'])
    # yaw.append(rot['yaw'])
    # rot['yaw'] = rot['yaw'] -360
    # rot_mat = eulerAnglesToRotationMatrix([math.radians(rot['roll']),math.radians(rot['pitch']),math.radians(rot['yaw'])])
    rot_mat = eulerAnglesToRotationMatrix([math.radians(rot['pitch']),math.radians(rot['roll']),math.radians(rot['yaw'])]) # worked on pure elevation trans3
    # rot_mat = eulerAnglesToRotationMatrix([math.radians(rot['pitch']),math.radians(rot['yaw']),math.radians(rot['roll'])])
    # rot_mat = eulerAnglesToRotationMatrix([math.radians(rot['pitch']),math.radians(rot['roll']),math.radians(rot['yaw'])])
    # rot_mat = np.linalg.inv(rot_mat)
    # trans_1 = np.array(loc['x'],loc['y'],loc['z'])
    trans = np.zeros((3,1))

    # loc['y'] = -1 * loc['y']
    print('loc',loc['x'],loc['y'],loc['z'] )
    trans[:,0] = np.transpose(np.array((loc['x'],loc['y'],loc['z'])))
    trans[:,0] = np.transpose(np.array((loc['y'],loc['z'],loc['x'])))
    # trans[:,0] = np.transpose(np.array((loc['y'],loc['x'],loc['z'])))

    print('rotation')
    print(rot['roll'])
    print(rot['pitch'])
    print(rot['yaw'])
    print(rot_mat)
    print(rot_mat.shape)
    print(trans)
    trans = trans/100 # 1/1 meters
    print(trans.shape)
    pose = np.hstack((rot_mat,trans))
    last_row = np.zeros((1,4))
    last_row[:,3] = 1
    pose = np.vstack((pose,last_row))
    # pose = np.linalg.inv(pose)
    print('pose')
    print(pose)
    # np.savetxt('pose_out.txt', pose)
    pose_filename = '/home/daryl/depth-fusion-unreal/tsdf-fusion-python/data_house/rgbd-frames/frame-{:06}.pose.txt'.format(idx)
    np.savetxt(pose_filename, pose)
    # x = loc['x']
    # y = loc['y']
    # z = loc['z']
    # roll = rot['roll']
    # pitch = rot['pitch']
    # yaw = rot['yaw']

    # print(**loc)
    # print('idx: ', idx)

    # Set position of the first camera
    client.request('vset /camera/0/location {x} {y} {z}'.format(**loc))
    client.request('vset /camera/0/rotation {pitch} {yaw} {roll}'.format(**rot))

    # Get image
    # res = client.request('vget /camera/0/lit lit.png')
    res = client.request('vget /camera/0/lit /home/daryl/depth-fusion-unreal/tsdf-fusion-python/data_house/rgbd-frames/frame-{:06d}.color.png'.format(idx))
    res_depth = client.request('vget /camera/0/depth /home/daryl/depth-fusion-unreal/tsdf-fusion-python/data_house/rgbd-frames/frame-{:06}.depth.png'.format(idx))
    res_depth2 = client.request('vget /camera/0/depth npy')
    print('type ', type(res_depth2))
    depth_pt = read_npy(res_depth2)
    depth = DepthConversion(depth_pt, 320)
    # depth = depth_pt
    print('max and min depth', np.max(depth), np.min(depth))
    print(depth.shape)
    depth_np_filename = '/home/daryl/depth-fusion-unreal/tsdf-fusion-python/data_house/rgbd-frames/frame-{:06}.depth.npy'.format(idx)

    np.save(depth_np_filename, depth)

    print('The image is saved to %s' % res)
    print('Depth is saved to %s' % res_depth)
    # idx = idx +1
    # delay
# write_pose(x,y,z, roll,pitch, yaw)
