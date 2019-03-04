import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time

from predictor import PosPrediction

class LandmarkDetector:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, batch, prefix = '.', gpu_memory_fraction=0.8):

        # resolution of input and output image size.
        self.resolution_inp = 256 
        self.resolution_op = 160
        self.prefix = prefix

        #---- load PRN 
        self.pos_predictor = PosPrediction(
            batch, 
            self.resolution_inp, self.resolution_op, 
            gpu_memory_fraction=gpu_memory_fraction
        )

        # uv file
        # TODO: Convert to tensorflow
        self.uv_kpt_ind = np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt').astype(np.int32) # 2 x 68 get kpt

    def restore(self, sess):
        prn_path = os.path.join(self.prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(sess, prn_path)

    def get_landmarks(self):
        '''
        Returns:
            kpt: 68 3D landmarks. shape = (?, 68, 3).
        '''
        pos = self.pos_predictor.predict_batch()

        # TODO: Implement without using tf.transpose
        pos = tf.transpose(pos, perm=[2,1,0,3]) # (256,256,?,3)
        kpt = tf.gather_nd(pos, np.transpose(self.uv_kpt_ind))
        kpt = tf.transpose(kpt, perm=[1,0,2]) # (?,68,3)
        return kpt

if __name__ == '__main__':
    import cv2
    import tensorflow as tf
    img1 = cv2.imread(
        os.path.expanduser(
            '~/datasets/vggface2/train_cropped_no_pan_headwear_expr_160x160/n000002/0012_01.jpg'
        )
    )
    img2 = cv2.imread(
        os.path.expanduser(
            '~/datasets/vggface2/train_cropped_no_pan_headwear_expr_160x160/n000002/0024_01.jpg'
        )
    )
    stacked_imgs = np.stack([img1, img2], 0) / 255.
    imgs = tf.convert_to_tensor(stacked_imgs, dtype=tf.float32)

    detector = LandmarkDetector(imgs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        detector.restore(sess)
        landmarks = detector.get_landmarks()
        L = np.asarray(sess.run(landmarks))
        print(L.shape)
        for i in range(68):
            cv2.circle(img1, tuple(L[0,i,:2]), 2, (255,0,0), -1)
            cv2.circle(img2, tuple(L[1,i,:2]), 2, (255,0,0), -1)

        cv2.imwrite(os.path.expanduser('~/Downloads/img1.jpg'), img1)
        cv2.imwrite(os.path.expanduser('~/Downloads/img2.jpg'), img2)
        
        











