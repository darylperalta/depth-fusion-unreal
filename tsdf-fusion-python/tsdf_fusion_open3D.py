# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

from py3d import *
#from trajectory_io import *
import numpy as np
from utils_withcolor import LoadMatrixFromFile
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSDF-fusion implementation")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("--data",  dest='data_path',default= './data_house/rgbd-frames', help='data path')
    # parser.add_argument("--data",  dest='data_path',default= '/home/daryl/datasets/rgbd-datsets/pumpkin/seq-01', help='data path')
    # parser.add_argument("--data",  dest='data_path', help='data path')
    # parser.add_argument("--cam", dest='cam_K_file',default='/home/daryl/tsdf-fusion-python/data/camera-intrinsics.txt', help = 'camera intrinsics')
    parser.add_argument("--cam", dest='cam_K_file',default='/home/daryl/tsdf-fusion-python/data/camera-intrinsics.txt', help = 'camera intrinsics')
    parser.add_argument("--out_pts",  dest='output_pts',default= 'tsdf.ply', help = 'output point cloud')
    parser.add_argument("--out_bin", dest='output_bin',default='mesh.ply', help = 'output bin')

    args = parser.parse_args()

    data_path = args.data_path
    cam_K_file = args.cam_K_file
    output_pts = args.output_pts
    output_bin = args.output_bin

    #camera_poses = read_trajectory("../../TestData/RGBD/odometry.log")
    voxel_size = 0.006
    volume = ScalableTSDFVolume(voxel_length = voxel_size,
            sdf_trunc = voxel_size*5, with_color = True)


    first_frame_idx = 150
    num_frames = 2

    '''load intrinsic matrix'''
    cam_K = LoadMatrixFromFile(cam_K_file,3,3)
    #print('camk')
    #print(cam_K)
    #cam_K_np = np.array(cam_K).reshape(3,3)
    cam_K_np = np.array(cam_K,dtype='float32')
    print('intriinsic matrix',cam_K_np.reshape(3,3))

    intrinsic_open3d = PinholeCameraIntrinsic.get_prime_sense_default()
    intrinsic_open3d.intrinsic_matrix = cam_K_np.reshape(3,3)




    #print('length: ',len(camera_poses))
    for frame_idx in range(first_frame_idx, first_frame_idx+num_frames):

        curr_frame_prefix = '{:06}'.format(frame_idx)
        depth_im_file = data_path +'/frame-' + curr_frame_prefix + '.depth.png'
        color_im_file = data_path + '/frame-' + curr_frame_prefix + '.color.png'

        ''' read base frame camera pose '''
        cam2world_file = data_path+'/frame-' + curr_frame_prefix + '.pose.txt'
        cam2world = LoadMatrixFromFile(cam2world_file)
        cam2world_np = np.array(cam2world, dtype='float32').reshape(4,4)

        print("Integrate {:d}-th image into the volume.".format(frame_idx))
        color = read_image(color_im_file)
        depth = read_image(depth_im_file)

        rgbd = create_rgbd_image_from_color_and_depth(color, depth,
                depth_trunc = 4.0, convert_rgb_to_intensity = False)
        volume.integrate(rgbd, intrinsic_open3d,
                np.linalg.inv(cam2world_np))

    print('volume', volume)
    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    draw_geometries([mesh])
    write_triangle_mesh(output_bin,mesh)

    point_cloud = volume.extract_point_cloud()
    write_point_cloud(output_pts, point_cloud)
    # draw_geometries([point_cloud])
