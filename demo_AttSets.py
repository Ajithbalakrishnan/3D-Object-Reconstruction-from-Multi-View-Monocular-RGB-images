import tensorflow as tf
import os
import sys
sys.path.append('..')
import tools as tools
import numpy as np
import matplotlib.pyplot as plt

GPU='0'

def load_real_rgbs(test_mv=3):
    obj_rgbs_folder ='./Data_sample/amazon_real_rgbs/edited/'
    rgbs = []
    rgbs_views = sorted(os.listdir(obj_rgbs_folder))
    for v in rgbs_views:
        if not v.endswith('png'): continue
        print('start')
        rgbs.append(tools.Data.load_single_X_rgb_r2n2(obj_rgbs_folder + v, train=False))
    print('end')
    rgbs = np.asarray(rgbs)
    x_sample = rgbs[0:test_mv, :, :, :].reshape(1, test_mv, 127, 127, 3)
    return x_sample, None

def load_shapenet_rgbs(test_mv=3):
    obj_rgbs_folder = './Data_sample/ShapeNetRendering/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/rendering/'
    obj_gt_vox_path ='./Data_sample/ShapeNetVox32/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/model.binvox'
    rgbs=[]
    rgbs_views = sorted(os.listdir(obj_rgbs_folder))
    for v in rgbs_views:
        if not v.endswith('png'): continue
        rgbs.append(tools.Data.load_single_X_rgb_r2n2(obj_rgbs_folder + v, train=False))
    rgbs = np.asarray(rgbs)
    x_sample = rgbs[0:test_mv, :, :, :].reshape(1, test_mv, 127, 127, 3)
    y_true = tools.Data.load_single_Y_vox(obj_gt_vox_path)
    return x_sample, y_true

def ttest_demo():
    model_path = './Model_released/'
    if not os.path.isfile(model_path + 'model.cptk.data-00000-of-00001'):
        print ('please download our released model first!')
        return

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = GPU
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(model_path + 'model.cptk.meta', clear_devices=True)
        saver.restore(sess, model_path + 'model.cptk')
        print ('model restored!')

        X = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        Y_pred = tf.get_default_graph().get_tensor_by_name("r2n/Reshape_9:0")

        x_sample, gt_vox = load_real_rgbs()
        #x_sample, gt_vox = load_shapenet_rgbs()
        y_pred = sess.run(Y_pred, feed_dict={X: x_sample})

    ###### to visualize
    th = 0.25
    y_pred[y_pred>=th]=1
    y_pred[y_pred<th]=0
    tools.Data.plotFromVoxels(np.reshape(y_pred,[32,32,32]), title='y_pred')
    if gt_vox is not None:
        tools.Data.plotFromVoxels(np.reshape(gt_vox,[32,32,32]), title='y_true')
    from matplotlib.pyplot import show
    show()

#########################
if __name__ == '__main__':
	print ('enterd')    
	ttest_demo()
    
