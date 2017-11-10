import argparse
import os
import scipy.misc
import numpy as np
import h5py
import scipy.io as sio

  
def parse_make3d(data_dir, result_dir):
    print(data_dir, result_dir)

    for file in os.listdir(data_dir):
        data_path = data_dir + file
    #    with h5py.File(data_path, 'r') as f:
    #        print(list(f.keys()))
        mat = sio.loadmat(data_path)

        array = mat['Position3DGrid']
        print(array.shape)
        
        print(array)

        break

    '''
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
    '''

def main():
    # parse 
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help = 'make3d dir')
    parser.add_argument('result_dir', help = 'make3d result path')
   
    args = parser.parse_args()
    parse_make3d(args.data_dir, args.result_dir)


if __name__ == '__main__':
    main()
