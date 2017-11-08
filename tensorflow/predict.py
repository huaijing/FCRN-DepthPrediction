import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import models
import scipy.ndimage

def predict(model_data_path, image_dir, result_dir):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
 
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:
        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        for img_name in os.listdir(image_dir):
            image_path = image_dir + img_name
            print(image_path, image_dir, img_name)

            # Read image
            img = Image.open(image_path)
            img = img.resize([width,height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis = 0)
      
            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
            depthMap = pred[0,:,:,0]
            alignedDepthMap = depthMap[4:124,:]
            print(alignedDepthMap.max(), alignedDepthMap.min(), alignedDepthMap.shape)

            
            print('Resampled by a factor of 4 with bilinear interpolation:')
            #finalDepthMap = scipy.ndimage.zoom(alignedDepthMap, 4, order=1)
            finalDepthMap = scipy.ndimage.zoom(alignedDepthMap, 1.6, order=1)
            
            finalDepthMap = scipy.ndimage.filters.median_filter(finalDepthMap, size=(5, 5))
            finalDepthMap = np.transpose(finalDepthMap)
            finalDepthMap = np.flip(finalDepthMap, 1)

            result_pre = result_dir + img_name[:-4]
            # Plot result
            fig = plt.figure()
            ii = plt.imshow(finalDepthMap, interpolation='nearest')
            fig.colorbar(ii)
            plt.savefig(result_pre + '_figure.png', dpi=100)
            plt.imsave(result_pre + '_result.png', finalDepthMap)
            plt.close()

            # save txt
            np.savetxt(result_pre + '_result.txt', finalDepthMap, fmt=["%.3f",]*finalDepthMap.shape[1])

                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_dir', help='Directory of images to predict')
    parser.add_argument('result_dir', help='Directory of result')
    args = parser.parse_args()

    # Predict the image
    predict(args.model_path, args.image_dir, args.result_dir)
    
    os._exit(0)

if __name__ == '__main__':
    main()
