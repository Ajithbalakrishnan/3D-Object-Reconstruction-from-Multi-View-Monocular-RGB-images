import tensorflow as tf
import os
import sys
sys.path.append('..')
import tools as tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
GPU='0'

vox_res = 32

def load_real_rgbs(test_mv=3):
    obj_rgbs_folder ='./Data_sample/amazon_real_rgbs/lamp/'
    rgbs = []
    rgbs_views = sorted(os.listdir(obj_rgbs_folder))
    for v in rgbs_views:
        if not v.endswith('png'): continue
        rgbs.append(tools.Data.load_single_X_rgb_r2n2(obj_rgbs_folder + v, train=False))
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
    #########################################
    Y_true_vox = []
    Y_true_vox.append(y_true)
    Y_true_vox = np.asarray(Y_true_vox)
    return x_sample, Y_true_vox
    #########################################
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

#        x_sample, gt_vox = load_real_rgbs()
        x_sample, gt_vox = load_shapenet_rgbs()
        
	### reconstruction loss #########################################################
   
        gt_vox1=gt_vox.astype(np.float32)
        Y_vox_ = tf.reshape(gt_vox1, shape=[-1, vox_res ** 3])
        Y_pred_ = tf.reshape(Y_pred, shape=[-1, vox_res ** 3])
        rec_loss = tf.reduce_mean(-tf.reduce_mean(Y_vox_ * tf.log(Y_pred_ + 1e-8), reduction_indices=[1])-tf.reduce_mean((1 - Y_vox_) * tf.log(1 - Y_pred_ + 1e-8),reduction_indices=[1]))
        
        gt_vox=gt_vox.astype(np.float64)
#        Y_pred=Y_pred.astype(np.int32)

        Y_vox_ = tf.reshape(gt_vox, shape=[-1, vox_res ** 3,1])
        Y_pred_ = tf.reshape(Y_pred, shape=[-1, vox_res ** 3,1])
        iou = tf.metrics.mean_iou(labels=Y_vox_,predictions=Y_pred_,num_classes=1)
        sess.run(tf.local_variables_initializer())


                                #########################################################
        ## session run
        y_pred,recon_loss,iou_value = sess.run([Y_pred, rec_loss,iou], feed_dict={X: x_sample})			                     
        print("Cross entropy loss : ",	recon_loss)
        print("IOU :",iou_value)		                     
        
#        y_pred= sess.run(Y_pred, feed_dict={X: x_sample})             
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
    ttest_demo()
