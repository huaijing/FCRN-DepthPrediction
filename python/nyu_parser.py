import argparse
import os
import scipy.misc
import numpy as np
import h5py
  
  
def parse_NYU(data_path, result_dir):
    print(data_path, result_dir)
    with h5py.File(data_path, 'r') as f:
        print(list(f.keys()))
    
        n = f['depths'].shape[0]
        print(n)
        for i in range(n):
            image0 = f['images'][i,:,:,:]
            depth = f['depths'][i,:,:]
            image = np.transpose(image0, (1, 2, 0))  
 
            print(image0.shape, image.shape, depth.shape)

            image = np.transpose(image, (1, 0, 2))
            depth = np.transpose(depth)

            rip = result_dir + str(i) + "_image.jpg"
            scipy.misc.imsave(rip, image)

            rdp = result_dir + str(i) + "_depth.jpg"
            scipy.misc.imsave(rdp, depth)


def main():
    # parse 
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help = 'nyu_depth_v2')
    parser.add_argument('result_dir', help = 'nyu_depth_v2 result path')
   
    args = parser.parse_args()
    parse_NYU(args.data_path, args.result_dir)


if __name__ == '__main__':
    main()
