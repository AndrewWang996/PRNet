import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
import tensorflow as tf

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
        self.uv_kpt_ind = tf.convert_to_tensor(
            np.transpose(
                np.loadtxt(prefix + '/Data/uv-data/uv_kpt_ind.txt')
            ), # 68 x 2 
            dtype=tf.int32
        )

    def restore(self, sess, filter_vars=None):
        prn_path = os.path.join(self.prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(sess, prn_path, filter_vars)

    def get_landmarks(self):
        '''
        Returns:
            kpt: 68 3D landmarks. shape = (?, 68, 3).
        '''
        pos = self.pos_predictor.predict_batch()

        # TODO: Implement without using tf.transpose
        pos = tf.transpose(pos, perm=[2,1,0,3]) # (256,256,?,3)
        kpt = tf.gather_nd(pos, self.uv_kpt_ind)
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
        landmarks = detector.get_landmarks()
        detector.restore(sess)

        pairs = [(1,17), (2,16), (3,15), (4,14), (5,13), (6,12), (7,11), (8,10)] # jawline
        pairs.extend([(18,27), (19,26), (20,25), (21,24), (22,23)]) # eyebrows
        pairs.extend([(37,46), (38,45), (39,44), (40,43), (41,48), (42,47)])  # eyes
        pairs.extend([(32,36), (33,35)]) # mouth
        pairs.extend([(49,55), (50,54), (51,53), (56,60), (57,59)]) # outer lip
        pairs.extend([(61,65), (62,64), (66,68)]) # inner lip
        pairs = np.asarray(pairs) - 1
        left = tf.convert_to_tensor(pairs[:,0])
        right = tf.convert_to_tensor(pairs[:,1])
 

        L = np.asarray(sess.run(landmarks))
        Lp = tf.transpose(landmarks, perm=[1,0,2])
        Lp = tf.gather(Lp, left)
        Lp = tf.transpose(Lp, perm=[1,0,2])
        for i in range(48,60):
            cv2.circle(img1, tuple(L[0,i,:2]), 2, (255,0,0), -1)
            cv2.circle(img2, tuple(L[1,i,:2]), 2, (255,0,0), -1)

        cv2.imwrite(os.path.expanduser('~/Downloads/img1.jpg'), img1)
        cv2.imwrite(os.path.expanduser('~/Downloads/img2.jpg'), img2)
        
        











