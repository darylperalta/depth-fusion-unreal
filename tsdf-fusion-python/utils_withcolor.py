import numpy as np
import cv2
from math import floor
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import argparse


mod = SourceModule("""
    __global__ void Integrate(float * cam_K, float * cam2base, float * depth_im,
                   int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
                   float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
                   float * voxel_grid_TSDF, float * voxel_grid_weight) {

      int pt_grid_z = blockIdx.x;
      int pt_grid_y = threadIdx.x;

      for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

        // Convert voxel center from grid coordinates to base frame camera coordinates
        float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
        float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
        float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

        // Convert from base frame camera coordinates to current frame camera coordinates
        float tmp_pt[3] = {0};
        tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
        tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
        tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
        float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
        float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
        float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

        if (pt_cam_z <= 0)
          continue;

        int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
        int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
        if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
          continue;

        float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

        if (depth_val <= 0 || depth_val > 6)
          continue;

        float diff = depth_val - pt_cam_z;

        if (diff <= -trunc_margin)
          continue;

        // Integrate
        int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
        float dist = fmin(1.0f, diff / trunc_margin);
        float weight_old = voxel_grid_weight[volume_idx];
        float weight_new = weight_old + 1.0f;
        voxel_grid_weight[volume_idx] = weight_new;
        voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
      }
    }

    """)


def LoadMatrixFromFile(filename,M=3,N=3):
    #tmp = np.array((1,2), dtype=np.float32)
    #i = 0;
    tmp = []
    with open(filename) as f:
        for line in f:
            line_str = line.split()
            #line_str = float(line_str)
            #tmp.append(x)
            #tmp.append(y)
            #tmp.append(z)
            #print("type of line")
            #print(type(line_str))
            #print(line_str)
            #for i in range(N):
            #    line_str.pop()
            line_str.reverse()
            while(line_str):
                tmp.append(float(line_str.pop()))

            #element = line_str.pop()
            #print
    return tmp

def SaveVoxelGrid2SurfacePointCloud(filename,voxel_grid_dim_x,voxel_grid_dim_y,voxel_grid_dim_z, voxel_size,voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
    voxel_grid_TSDF,voxel_grid_weight,tsdf_thresh,weight_thresh):

    '''count total number of points in point cloud'''

    ply_header = '''ply
format ascii 1.0
element vertex %(num_pts)d
property float x
property float y
property float z
end_header
'''

    num_pts = 0
    '''
    for i in range (0, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z):
        if (abs(voxel_grid_TSDF[i])< tsdf_thresh and voxel_grid_weight[i] > weight_thresh):
            num_pts= num_pts+1
    print("Array addition     ")
    print(num_pts)
    '''
    mask = (abs(voxel_grid_TSDF)<tsdf_thresh) & (voxel_grid_weight>weight_thresh)
    num_pts = np.sum(mask)
    print("SDFasdf NUMPTS DDFS     ")
    print(num_pts)
    with open(filename, 'wb') as f:
        f.write((ply_header % dict(num_pts=num_pts)).encode('utf-8'))
        #np.savetxt(f, coordinate, fmt='%f %f %f ')


        mask = (abs(voxel_grid_TSDF)<tsdf_thresh) & (voxel_grid_weight>weight_thresh)
        num_pts = np.sum(mask)

        i = np.arange(voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z)

        z = np.zeros((num_pts),dtype='int32')
        y = np.zeros((num_pts),dtype='int32')
        x = np.zeros((num_pts),dtype='int32')

        z = np.floor(i[mask]/(voxel_grid_dim_x*voxel_grid_dim_y))
        y = np.floor((i[mask]-(z*voxel_grid_dim_z*voxel_grid_dim_y))/voxel_grid_dim_x)
        x = np.int32(i[mask] -(z*voxel_grid_dim_x*voxel_grid_dim_y)-(y*voxel_grid_dim_x))

        pt_base_x = np.float32(voxel_grid_origin_x + np.float32(x) * voxel_size).reshape(num_pts,1)
        pt_base_y = np.float32(voxel_grid_origin_y + np.float32(y) * voxel_size).reshape(num_pts,1)
        pt_base_z = np.float32(voxel_grid_origin_z + np.float32(z) * voxel_size).reshape(num_pts,1)

        #coordinates = np.zeros((num_pts,3),dtype='float32')
        coordinates = np.hstack((pt_base_x,pt_base_y,pt_base_z))
        np.savetxt(f, coordinates, fmt='%f %f %f ')
                #f.write(pt_base_x)
                #f.write(pt_base_y)
                #f.write(pt_base_z)


def main():
    '''
    cam_K_file = "data/camera-intrinsics.txt"
    data_path = "data/rgbd-frames"

    output_pts = 'tsdf.ply'
    output_bin = 'tsdf.bin'
    '''
    parser = argparse.ArgumentParser(description="TSDF-fusion implementation")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("--data",  dest='data_path',default= 'data_house/rgbd-frames')
    parser.add_argument("--cam", dest='cam_K_file',default='data_house/camera-intrinsics.txt')
    parser.add_argument("--out_pts",  dest='output_pts',default= 'tsdf_house.ply')
    parser.add_argument("--out_bin", dest='output_bin',default='tsdf_house.bin')
    parser.add_argument("--first_frame", dest='first_frame_idx', type=int, default=0)
    parser.add_argument("--base_frame", dest='base_frame_idx', type=int, default=0)
    parser.add_argument("--num_frames", dest='num_frames',type=int, default=2)
    parser.add_argument("--npy",action='store_true',default=False)

    args = parser.parse_args()


    data_path = args.data_path
    cam_K_file = args.cam_K_file
    output_pts = args.output_pts
    output_bin = args.output_bin
    first_frame_idx = args.first_frame_idx
    base_frame_idx = args.base_frame_idx
    num_frames = args.num_frames
    npy = args.npy
    print('base frame', base_frame_idx)
    #data_path = "data1/rgbd-frames"
    ''' chess dataset '''
    #data_path = "/home/daryl/datasets/chess/seq-01"
    #data_path = "data1/rgbd-frames"
    # base_frame_idx = 150
    # first_frame_idx = 150

    # num_frames = 2

    im_width = 640
    im_height = 480

    voxel_grid_origin_x = -1.5
    voxel_grid_origin_y = -1.5
    voxel_grid_origin_z = -0.5
    voxel_size = 0.006
    voxel_size = 0.03
    # voxel_size = 0.12

    trunc_margin = voxel_size * 5
    # voxel_grid_dim_x = 500
    # voxel_grid_dim_y = 500
    # voxel_grid_dim_z = 500
    voxel_grid_dim_x = 600
    voxel_grid_dim_y = 600
    voxel_grid_dim_z = 600


    # voxel_size = 0.006
    # voxel_grid_dim_x = 500
    # voxel_grid_dim_y = 500
    # voxel_grid_dim_z = 500


    '''read camera intrinsics'''
    cam_K = LoadMatrixFromFile(cam_K_file,3,3)
    print('camk')
    print(cam_K)
    #cam_K_np = np.array(cam_K).reshape(3,3)
    cam_K_np = np.array(cam_K,dtype='float32')
    #cam_K_np.reshape(3,3)
    print(cam_K_np)

    '''read base frame camera pose'''
    base2world_file = data_path+'/frame-'+'{:06}'.format(base_frame_idx)+'.pose.txt'
    base2world = LoadMatrixFromFile(base2world_file)
    base2world_np = np.array(base2world,dtype='float32').reshape(4,4)
    print(base2world_np)
    print(base2world_np.dtype)
    print("asdfasfd")

    '''invert base frame camera pose to get world-to-base frame transform'''
    base2world_inv = np.linalg.inv(base2world_np)
    print('inverse')
    print(base2world_inv)

    '''flatten again the camera poses'''
    base2world_np_flat = base2world_np.flatten()
    base2world_inv_flat = base2world_inv.flatten()

    '''initialize voxel grid'''
    voxel_grid_TSDF = np.ones((voxel_grid_dim_x*voxel_grid_dim_y*voxel_grid_dim_z),dtype='float32')
    voxel_grid_weight = np.zeros((voxel_grid_dim_x*voxel_grid_dim_y*voxel_grid_dim_z),dtype='float32')


    print('voxel_grid_TSDF')
    print(voxel_grid_TSDF)
    print(voxel_grid_TSDF.shape)

    print('voxel_grid_TSDF')
    print(voxel_grid_weight)
    print(voxel_grid_weight.shape)

    '''Load variables to GPU memory'''
    gpu_voxel_grid_TSDF = cuda.mem_alloc(voxel_grid_TSDF.nbytes)
    gpu_voxel_grid_weight = cuda.mem_alloc(voxel_grid_weight.nbytes)
    cuda.memcpy_htod(gpu_voxel_grid_TSDF,voxel_grid_TSDF)
    cuda.memcpy_htod(gpu_voxel_grid_weight,voxel_grid_weight)

    '''add CUDA error check'''

    gpu_cam_K = cuda.mem_alloc(cam_K_np.nbytes)
    cuda.memcpy_htod(gpu_cam_K,cam_K_np)
    gpu_cam2base = cuda.mem_alloc(base2world_np_flat.nbytes)
    '''initialize depth image'''
    depth_im = np.zeros((im_height,im_width),dtype='float32')
    depth_norm_flat = depth_im.flatten()
    gpu_depth_im = cuda.mem_alloc(depth_norm_flat.nbytes)

    '''add CUDA error check'''

    '''Loop through each depth frame and integrate TSDF voxel grid'''
    print('Loop through each depth frame and integrate TSDF voxel grid')
    for frame_idx in range(first_frame_idx,first_frame_idx+num_frames):

        curr_frame_prefix = '{:06}'.format(frame_idx)
        #print(curr_frame_prefix)
        print('current frame prefix', curr_frame_prefix)
        '''read current frame depth'''

        color_im_file = data_path + '/frame-' + curr_frame_prefix + '.color.png'
        #print(depth_im_file)
        if npy:
            depth_im_file = data_path +'/frame-' + curr_frame_prefix + '.depth.npy'
            depth_im = np.load(depth_im_file)
            print(depth_im_file)
        else:
            depth_im_file = data_path +'/frame-' + curr_frame_prefix + '.depth.png'
            depth_im = cv2.imread(depth_im_file,cv2.IMREAD_UNCHANGED)
        # print('max and min depth', np.max(depth_im))
        print('max and min depth', np.max(depth_im), np.min(depth_im))
        # depth_norm = depth_im/1000.0
        depth_norm = depth_im/1.0
        mask_depth = depth_norm>10.0
        depth_norm[mask_depth]= 10.0
        depth_norm_flat = depth_norm.flatten().astype(np.float32)
        #print(type(depth_im))
        #print(depth_norm_flat.dtype)
        #print(depth_norm_flat[0:10])
        color_im = cv2.imread(color_im_file)

        print('type', color_im.dtype, color_im.shape)
        print('max and min color', np.max(color_im), np.min(depth_im))


        color_b = color_im[0].flatten()
        color_g = color_im[1].flatten()
        color_r = color_im[2].flatten()
        del(color_im)


        #print(depth_im_file)

        ''' read base frame camera pose '''
        cam2world_file = data_path+'/frame-' + curr_frame_prefix + '.pose.txt'
        cam2world = LoadMatrixFromFile(cam2world_file)
        cam2world_np = np.array(cam2world, dtype='float32').reshape(4,4)
        print('cam2world', cam2world_np)
        '''Compute relative camera pose (camera-to-base frame)'''
        cam2base = np.dot(base2world_inv,cam2world_np)
        print('cam2base', cam2base)
        cam2base_flat = cam2base.flatten()
        cuda.memcpy_htod(gpu_cam2base,cam2base_flat)
        cuda.memcpy_htod(gpu_depth_im,depth_norm_flat)
        #print('fusing')
        integrate_func = mod.get_function('Integrate')
        integrate_func(gpu_cam_K,gpu_cam2base,gpu_depth_im,np.int32(im_height),np.int32(im_width), np.int32(voxel_grid_dim_x), np.int32(voxel_grid_dim_y), np.int32(voxel_grid_dim_z),
            np.float32(voxel_grid_origin_x), np.float32(voxel_grid_origin_y), np.float32(voxel_grid_origin_z), np.float32(voxel_size), np.float32(trunc_margin), gpu_voxel_grid_TSDF, gpu_voxel_grid_weight,
            block=(voxel_grid_dim_y,1,1), grid=(voxel_grid_dim_y,1))


    #print('cam2base')
    #print(cam2base)

    '''Load TSDF voxel grid from GPU to CPU memory'''

    cuda.memcpy_dtoh(voxel_grid_TSDF, gpu_voxel_grid_TSDF)
    cuda.memcpy_dtoh(voxel_grid_weight, gpu_voxel_grid_weight)

    tsdf_thresh = 0.2
    weight_thresh =0.0
    #mask = (abs(voxel_grid_TSDF)<tsdf_thresh) and (voxel_grid_weight>weight_thresh)
    mask = (abs(voxel_grid_TSDF)<tsdf_thresh)
    #print('voxel grid')
    #mask2 =voxel_grid_TSDF)>tsdf_thresh
    #print(voxel_grid_TSDF[0:20])
    #print(np.sum(mask))

    print("Saving surface point cloud (tsdf.ply)...")




    SaveVoxelGrid2SurfacePointCloud(output_pts, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                  voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                  voxel_grid_TSDF, voxel_grid_weight, 0.2, 0.0);
    print("Depth fusion finished. Point cloud saved in 'tsdf.ply'")

    '''Saving TSDF voxel grid values to disk (tsdf.bin)...'''
    # print('Saving TSDF voxel grid values to disk (tsdf.bin)...')
    #
    # with open(output_bin, 'wb') as f:
    #     '''
    #     f.write(float(voxel_grid_dim_x))
    #     f.write(float(voxel_grid_dim_y))
    #     f.write(float(voxel_grid_dim_z))
    #     f.write(float(voxel_grid_origin_x))
    #     f.write(float(voxel_grid_origin_y))
    #     f.write(float(voxel_grid_origin_z))
    #     f.write(float(voxel_size))
    #     f.write(float(trunc_margin))
    #     for i in range(voxel_grid_TSDF.shape[0]):
    #         f.write(float(voxel_grid_TSDF[i]))
    #     '''
    #     np.float32(voxel_grid_dim_x).tofile(f)
    #     np.float32(voxel_grid_dim_y).tofile(f)
    #     np.float32(voxel_grid_dim_z).tofile(f)
    #     np.float32(voxel_grid_origin_x).tofile(f)
    #     np.float32(voxel_grid_origin_y).tofile(f)
    #     np.float32(voxel_grid_origin_z).tofile(f)
    #     np.float32(voxel_size).tofile(f)
    #     np.float32(trunc_margin).tofile(f)
    #     #for i in range(voxel_grid_TSDF.shape[0]):
    #         #f.write(float(voxel_grid_TSDF[i]))
    #     voxel_grid_TSDF.tofile(f)

if __name__ == '__main__':
    main()
