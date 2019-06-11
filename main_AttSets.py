import tensorflow as tf
import os
import shutil
import sys
import scipy.io
sys.path.append('..')
import tools as tools
import numpy as np
import time
from keras.layers import BatchNormalization,Conv3D,MaxPooling3D,Dense,Reshape,Add,LeakyReLU
from keras.activations import relu,sigmoid
from keras import models
import copy 
###################
#import tflearn
###################

batch_size = 1
img_res = 127
vox_res32 = 32
total_mv = 24   
GPU0 = '0'
#re_train=False
re_train=True
single_view_train = False
multi_view_train = False

#####################################

config={}                                 # python dictionary
config['batch_size'] = batch_size
config['total_mv'] = total_mv
#config['cat_names'] = ['02691156','02828884','02933112','02958343','03001627','03211117',
#            '03636649','03691459','04090263','04256520','04379243','04401088','04530566']
config['cat_names'] = ['03001627']
for name in config['cat_names']:
    config['X_rgb_'+name] = './Data_sample/ShapeNetRendering/'+name+'/'
    config['Y_vox_'+name] = './Data_sample/ShapeNetVox32/'+name+'/'

# output : {'batch_size': 1, 'total_mv': 24, 'cat_names': ['03001627'], 'Y_vox_03001627': '/home/wiproec4/3d reconstruction/attsets/Data_sample/#ShapeNetVox32/03001627/', 'X_rgb_03001627': '/home/wiproec4/3d reconstruction/attsets/Data_sample/ShapeNetRendering/03001627/'}
def metric_IoU(batch_voxel_occup_pred, batch_voxel_occup_true):
    batch_voxel_occup_pred_ = copy.deepcopy(batch_voxel_occup_pred)
    batch_voxel_occup_pred_[batch_voxel_occup_pred_ >= 0.5] = 1
    batch_voxel_occup_pred_[batch_voxel_occup_pred_ < 0.5] = 0
	
    I = batch_voxel_occup_pred_ * batch_voxel_occup_true
    U = batch_voxel_occup_pred_ + batch_voxel_occup_true			
    U[U < 1] = 0
    U[U >= 1] = 1
    iou = np.sum(I) * 1.0 / np.sum(U) * 1.0
    return iou

#####################################
def refiner_network(volumes_in):
	
	input_volumes_32 = tf.reshape(volumes_in, [-1, vox_res32, vox_res32, vox_res32, 1])
	
	print("input_volumes_32_shape" , input_volumes_32.shape)   #input_volumes_32_shape (?,32,32,32,1)
	
	rn1=Conv3D(filters=32, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_c1')(input_volumes_32)
	rn2=BatchNormalization()(rn1)
	rn3=LeakyReLU(alpha=.2)(rn2)
	volumes_16_l =MaxPooling3D(pool_size=(2, 2, 2),name='ref_m1')(rn3)
	
	print("volumes_16_l_shape" , volumes_16_l.shape)      #volumes_16_l_shape (?,16,16,16,32)
	
	rn5=Conv3D(filters=64, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_c2')(volumes_16_l)
	rn6=BatchNormalization()(rn5)
	rn7=LeakyReLU(alpha=.2)(rn6)
	volumes_8_l =MaxPooling3D(pool_size=(2, 2, 2),name='ref_m2')(rn7)
	
	print("volumes_8_l_shape" ,volumes_8_l.shape)
	
	rn9=Conv3D(filters=128, kernel_size=(4, 4, 4), padding='same',data_format="channels_last",name='ref_c3')(volumes_8_l)
	rn10=BatchNormalization()(rn9)
	rn11=LeakyReLU(alpha=.2)(rn10)
	volumes_4_l =MaxPooling3D(pool_size=(2, 2, 2),name='ref_m3')(rn11)
	
	print("volumes_4_l_shape" , volumes_4_l.shape)
	
	flatten_features=tf.reshape(volumes_4_l , [-1,8192])   
	
	fc1=Dense(units=2048, activation='relu',name='ref_fc1')(flatten_features)
	fc1=relu(fc1, alpha=0.0, max_value=None, threshold=0.0)
	
	print("fc1_shape",fc1.shape)
	
	fc2=Dense(units=8192, activation='relu',name='ref_fc2')(fc1)
	fc2=relu(fc2, alpha=0.0, max_value=None, threshold=0.0)
	
	print("fc2_shape",fc2.shape)
	
	fc2=tf.reshape(fc2, [-1, 4,4,4,128])        
	reshaped_1=Add()([fc2,volumes_4_l]) 
	
	print("reshaped_1.shape",reshaped_1.shape)
	
	rn13= tools.Ops.deconv3d(reshaped_1, k=4, out_c=64, str=2, name='ref_c4')
	rn14=BatchNormalization()(rn13)
	volumes_4_r=relu(rn14, alpha=0.0, max_value=None, threshold=0.0)
	
	print("volumes_4_r_shape",volumes_4_r.shape)
	
	reshaped_2=Add() ([volumes_4_r,volumes_8_l]) 
	
	print("reshaped_2_shape",reshaped_2.shape)

	rn16= tools.Ops.deconv3d(reshaped_2, k=4, out_c=32, str=2, name='ref_c5')
	rn17=BatchNormalization()(rn16)
	volumes_8_r =relu(rn17, alpha=0.0, max_value=None, threshold=0.0)
 
	reshaped_3=Add()([volumes_8_r,volumes_16_l])
    
	print("reshaped_3_shape",reshaped_3.shape)
	
	rn19= tools.Ops.deconv3d(volumes_8_r, k=4, out_c=1, str=2, name='ref_c6')
	volumes_16_r= sigmoid(rn19)

	reshape_4=Add()([volumes_16_r,input_volumes_32])
	reshape_4=(reshape_4*0.5)
	
	print("reshape_4_shape",reshape_4.shape)
	
	reshape_5=tf.reshape(reshape_4, [-1, vox_res32, vox_res32, vox_res32])

	return reshape_5
	
def attsets_fc(x, out_ele_num,  name):
    in_ele_num = tf.shape(x)[1]
    in_ele_len = int(x.get_shape()[2])
    out_ele_len = in_ele_len    
    print("out_ele_len ", out_ele_len)
	####################
    x_1st = x
    x_1st_tp = tf.reshape(x_1st, [-1, in_ele_len])
    weights_1st = tools.Ops.fc(x_1st_tp, out_d=out_ele_num*out_ele_len, name=name+'_1st')
	
	########## option 1
    weights_1st = weights_1st
	########## option 2
	#weights_1st = tf.nn.tanh(weights_1st)

	####################
    weights_1st = tf.reshape(weights_1st, [-1, in_ele_num, out_ele_num, out_ele_len])
    weights_1st = tf.nn.softmax(weights_1st, 1)
    x_1st = tf.tile(x_1st[:,:,None,:], [1,1,out_ele_num,1])
    x_1st = x_1st*weights_1st
    x_1st = tf.reduce_sum(x_1st, axis=1)
    x_1st = tf.reshape(x_1st, [-1, out_ele_num*out_ele_len])       
    return x_1st, weights_1st

#####################################
class Network:
	def __init__(self):
		self.train_mod_dir = './train_mod/'
		self.train_sum_dir = './train_sum/'
		self.test_res_dir = './test_res/'
		self.test_sum_dir = './test_sum/'
		
		print ('re_train : ', re_train)
		if os.path.exists(self.test_res_dir):
			if re_train:
				print ('test_res_dir and files kept!')
			else:
				shutil.rmtree(self.test_res_dir)
				os.makedirs(self.test_res_dir)
				print ('test_res_dir: deleted and then created!')
		else:
			os.makedirs(self.test_res_dir)
			print ('test_res_dir: created!')
		
		if os.path.exists(self.train_mod_dir):
			if re_train:
				if os.path.exists(self.train_mod_dir + 'model.cptk.data-00000-of-00001'):
					print ('model found! will be reused!')
				else:
					print ('model not found! error!')
					#exit()
			else:
				shutil.rmtree(self.train_mod_dir)
				os.makedirs(self.train_mod_dir)
				print ('train_mod_dir: deleted and then created!')
		else:
			os.makedirs(self.train_mod_dir)
			print ('train_mod_dir: created!')
		
		if os.path.exists(self.train_sum_dir):
			if re_train:
				print ('train_sum_dir and files kept!')
			else:
				shutil.rmtree(self.train_sum_dir)
				os.makedirs(self.train_sum_dir)
				print ('train_sum_dir: deleted and then created!')
		else:
			os.makedirs(self.train_sum_dir)
			print ('train_sum_dir: created!')
		
		if os.path.exists(self.test_sum_dir):
			if re_train:
				print ('test_sum_dir and files kept!')
			else:
				shutil.rmtree(self.test_sum_dir)
				os.makedirs(self.test_sum_dir)
				print ('test_sum_dir: deleted and then created!')
		else:
			os.makedirs(self.test_sum_dir)
			print ('test_sum_dir: created!')

	def base_r2n2(self, X_rgb):
		im_num = tf.shape(X_rgb)[1]
		[_, _, d1, d2, cc] = X_rgb.get_shape()
		X_rgb = tf.reshape(X_rgb, [-1, int(d1), int(d2), int(cc)])
		print("Network Structure")
		print("base_r2n2",X_rgb.shape)
 
		en_c = [96, 128, 256, 256, 256, 256]
		l1 = tools.Ops.xxlu(tools.Ops.conv2d(X_rgb, k=7, out_c=en_c[0], str=1, name='l1'), label='lrelu')
#		tf.summary.image(name="2D_c1",tensor=l1,max_outputs=1)
		print("l1_r2n",l1.shape)
		l2 = tools.Ops.xxlu(tools.Ops.conv2d(l1, k=3, out_c=en_c[0], str=1, name='l2'), label='lrelu')
#		tf.summary.image(name="2D_c2",tensor=l2,max_outputs=1)
		l2 = tools.Ops.maxpool2d(l2, k=2, s=2, name='l2_p')
#		tf.summary.image(name="2D_MaxP",tensor=l2,max_outputs=1)
		print("l2_r2n",l2.shape)

		l3 = tools.Ops.xxlu(tools.Ops.conv2d(l2, k=3, out_c=en_c[1], str=1, name='l3'), label='lrelu')
		print("l3_r2n",l3.shape)
		l4 = tools.Ops.xxlu(tools.Ops.conv2d(l3, k=3, out_c=en_c[1], str=1, name='l4'), label='lrelu')
		print("l4_r2n",l4.shape)
		l22 = tools.Ops.conv2d(l2, k=1, out_c=en_c[1], str=1, name='l22')
		print("l22_r2n",l22.shape)
		l4 = l4 + l22
		l4 = tools.Ops.maxpool2d(l4, k=2, s=2, name='l4_p')
		print("l4+l22_r2n",l4.shape)

		l5 = tools.Ops.xxlu(tools.Ops.conv2d(l4, k=3, out_c=en_c[2], str=1, name='l5'), label='lrelu')
		print("l5_r2n",l5.shape)
		l6 = tools.Ops.xxlu(tools.Ops.conv2d(l5, k=3, out_c=en_c[2], str=1, name='l6'), label='lrelu')
		print("l6_r2n",l6.shape)
		l44 = tools.Ops.conv2d(l4, k=1, out_c=en_c[2], str=1, name='l44')
		print("l44_r2n",l44.shape)
		l6 = l6 + l44
		l6 = tools.Ops.maxpool2d(l6, k=2, s=2, name='l6_p')
		print("l6+l44_r2n",l6.shape)

		l7 = tools.Ops.xxlu(tools.Ops.conv2d(l6, k=3, out_c=en_c[3], str=1, name='l7'), label='lrelu')
		print("l7_r2n",l7.shape)
		l8 = tools.Ops.xxlu(tools.Ops.conv2d(l7, k=3, out_c=en_c[3], str=1, name='l8'), label='lrelu')
		l8 = tools.Ops.maxpool2d(l8, k=2, s=2, name='l8_p')
		print("l8_r2n",l8.shape)

		l9 = tools.Ops.xxlu(tools.Ops.conv2d(l8, k=3, out_c=en_c[4], str=1, name='l9'), label='lrelu')
		print("l9_r2n",l9.shape)
		l10 = tools.Ops.xxlu(tools.Ops.conv2d(l9, k=3, out_c=en_c[4], str=1, name='l10'), label='lrelu')
		print("l10_r2n",l10.shape)
		l88 = tools.Ops.conv2d(l8, k=1, out_c=en_c[4], str=1, name='l88')
		print("l88_r2n",l88.shape)
		l10 = l10 + l88
		l10 = tools.Ops.maxpool2d(l10, k=2, s=2, name='l10_p')
		print("l10_r2n",l10.shape)

		l11 = tools.Ops.xxlu(tools.Ops.conv2d(l10, k=3, out_c=en_c[5], str=1, name='l11'), label='lrelu')
		print("l11_r2n",l11.shape)
		l12 = tools.Ops.xxlu(tools.Ops.conv2d(l11, k=3, out_c=en_c[5], str=1, name='l12'), label='lrelu')
		print("l12_r2n",l12.shape)
		l1010 = tools.Ops.conv2d(l10, k=1, out_c=en_c[5], str=1, name='l1010_p')
		print("l1010_r2n",l1010.shape)
		l12 = l12 + l1010
		l12 = tools.Ops.maxpool2d(l12, k=2, s=2, name='l12_p')
		print("l12_r2n",l12.shape)

		[_, d1, d2, cc] = l12.get_shape()
		l12 = tf.reshape(l12, [-1, int(d1) * int(d2) * int(cc)])
		print("fc1_input_r2n",l12.shape)
		fc = tools.Ops.xxlu(tools.Ops.fc(l12, out_d=1024, name='lfc1'), label='lrelu')
		print("fc1_output_r2n",fc.shape)

		#### use fc attention
		input = tf.reshape(fc, [-1, im_num, 1024])
		print("att_fc_in_r2n",input.shape)
		latent_3d, weights = attsets_fc(input, out_ele_num=1, name='att')
		print("att_fc_out_r2n",latent_3d.shape)

		####
		latent_3d = tools.Ops.xxlu(tools.Ops.fc(latent_3d, out_d=4*4*4*128, name='lfc2'), label='lrelu')
		print("fc3_out_r2n",latent_3d.shape)
		latent_3d = tf.reshape(latent_3d, [-1, 4, 4, 4, 128])

		####
		de_c = [128, 128, 128, 64, 32, 1]
		
		print("d1_in_r2n",latent_3d.shape)
		d1 = tools.Ops.xxlu(tools.Ops.deconv3d(latent_3d, k=3, out_c=de_c[1], str=2, name='ld1'), label='lrelu')
		print("d1_out_r2n",d1.shape)
		d2 = tools.Ops.xxlu(tools.Ops.deconv3d(d1, k=3, out_c=de_c[1], str=1, name='ld2'), label='lrelu')
		print("d2_out_r2n",d2.shape)
		d00 = tools.Ops.deconv3d(latent_3d, k=1, out_c=de_c[1], str=2, name='ld00')
		print("d00_out_r2n",d00.shape)
		d2 = d2 + d00
		print("d2+d00_out_r2n",d2.shape)

		d3 = tools.Ops.xxlu(tools.Ops.deconv3d(d2, k=3, out_c=de_c[2], str=2, name='ld3'), label='lrelu')
		print("d3_out_r2n",d3.shape)
		d4 = tools.Ops.xxlu(tools.Ops.deconv3d(d3, k=3, out_c=de_c[2], str=1, name='ld4'), label='lrelu')
		print("d4_out_r2n",d4.shape)
		d22 = tools.Ops.deconv3d(d2, k=1, out_c=de_c[2], str=2, name='ld22')
		print("d22_out_r2n",d22.shape)
		d4 = d4 + d22
		print("d4+d22_out_r2n",d4.shape)

		d5 = tools.Ops.xxlu(tools.Ops.deconv3d(d4, k=3, out_c=de_c[3], str=2, name='ld5'), label='lrelu')
		print("d5_out_r2n",d5.shape)
		d6 = tools.Ops.xxlu(tools.Ops.deconv3d(d5, k=3, out_c=de_c[3], str=1, name='ld6'), label='lrelu')
		print("d6_out_r2n",d6.shape)
		d44 = tools.Ops.deconv3d(d4, k=1, out_c=de_c[3], str=2, name='ld44')
		print("d44_out_r2n",d44.shape)
		d6 = d6 + d44
		print("d6+d44_out_r2n",d6.shape)

		d7 = tools.Ops.xxlu(tools.Ops.deconv3d(d6, k=3, out_c=de_c[4], str=1, name='ld7'), label='lrelu')
		print("d7_out_r2n",d7.shape)
		d8 = tools.Ops.xxlu(tools.Ops.deconv3d(d7, k=3, out_c=de_c[4], str=1, name='ld8'), label='lrelu')
		print("d8_out_r2n",d8.shape)
		d77 = tools.Ops.xxlu(tools.Ops.deconv3d(d7, k=3, out_c=de_c[4], str=1, name='ld77'), label='lrelu')
		print("d77_out_r2n",d77.shape)
		d8 = d8 + d77
		print("d8+d77_out_r2n",d8.shape)

		d11 = tools.Ops.deconv3d(d8, k=3, out_c=de_c[5], str=1, name='ld11')
		print("d11_out_r2n",d11.shape)
		y = tf.nn.sigmoid(d11)

		y = tf.reshape(y, [-1, vox_res32, vox_res32, vox_res32])
		print("y_out_r2n",y.shape)
                
		y=refiner_network(y)
		return y, weights

	def build_graph(self):
		img_res = 127
		vox_res = 32
		self.X_rgb = tf.placeholder(shape=[None, None, img_res, img_res, 3], dtype=tf.float32)
		self.Y_vox = tf.placeholder(shape=[None, vox_res, vox_res, vox_res], dtype=tf.float32)
		self.lr = tf.placeholder(tf.float32)
		
		with tf.variable_scope('r2n'):
			self.Y_pred, self.weights = self.base_r2n2(self.X_rgb)
			tf.summary.histogram('Attsets_Weights', self.weights)
		
		with tf.device('/gpu:' + GPU0):
			### rec loss
			print ("reached")
			Y_vox_ = tf.reshape(self.Y_vox, shape=[-1, vox_res ** 3])
			Y_pred_ = tf.reshape(self.Y_pred, shape=[-1, vox_res ** 3])
			self.rec_loss = tf.reduce_mean(-tf.reduce_mean(Y_vox_ * tf.log(Y_pred_ + 1e-8), reduction_indices=[1]) -
			                     tf.reduce_mean((1 - Y_vox_) * tf.log(1 - Y_pred_ + 1e-8),reduction_indices=[1]))
			sum_rec_loss = tf.summary.scalar('rec_loss', self.rec_loss)
			self.sum_merged = sum_rec_loss
			tf.summary.histogram('rec_loss', self.rec_loss)
			
			# Y_vox__=Y_vox_.astype(np.float32)
			# iou_value= metric_IoU( Y_pred_,Y_vox__)
			# tf.summary.histogram('iou_value', iou_value) 
			# tf.summary.scalar('iou_value', iou_value)
			
#			self.sum_histo=tf.summary.histogram('rec_loss', self.rec_loss)
 
             

			base_var = [var for var in tf.trainable_variables() if var.name.startswith('r2n/l')]
			att_var = [var for var in tf.trainable_variables() if var.name.startswith('r2n/att')]
			refine_var = [var for var in tf.trainable_variables() if var.name.startswith('r2n/ref')]
			self.base_optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.rec_loss, var_list=base_var)
			self.att_optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.rec_loss, var_list=att_var)
			self.refine_optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.rec_loss, var_list=refine_var)
		
		print ("total weights:",tools.Ops.variable_count())
		self.saver = tf.train.Saver(max_to_keep=1)
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.visible_device_list = GPU0
		self.sess = tf.Session(config=config)
		self.merged = tf.summary.merge_all()
		self.sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, self.sess.graph)
		self.sum_writer_test = tf.summary.FileWriter(self.test_sum_dir, self.sess.graph)

		#######################
		path = self.train_mod_dir
		#path = './Model_released/'  # retrain the released model

		if os.path.isfile(path + 'model.cptk.data-00000-of-00001'):
			print ("restoring saved model!")
			self.saver.restore(self.sess, path + 'model.cptk')
		else:
			self.sess.run(tf.global_variables_initializer())
		return 0
    
	def train(self, data):
		for epoch in range(0, 3, 1):
			train_view_num = 24  ##!!!!!!!!!!!
			data.shuffle_train_files(epoch, train_mv=train_view_num)
			total_train_batch_num = data.total_train_batch_num  #int(len(self.X_rgb_train_files)/(self.batch_size*train_mv))
			print ('total_train_batch_num:', total_train_batch_num)
			for i in range(total_train_batch_num):
				#### training
				X_rgb_bat, Y_vox_bat = data.load_X_Y_train_next_batch(train_mv=train_view_num)
				print("multi_view_train_X_rgb_bat : ",X_rgb_bat.shape)#np.asarray(X.append(X_rgb[b*train_mv:(b+1)*train_mv,:]))
				

				print(time.ctime())
				
				##### option 1: seperate train, seperate optimize
				#if epoch<=30:
				#	single_view_train=True
				#	multi_view_train=False
				#else:
				#	single_view_train=False
				#	multi_view_train=True

				##### optiion 2: joint train, seperate optimize
				single_view_train = True
				multi_view_train = True

				###########  single view train
				if single_view_train:
					
					rgb = np.reshape(X_rgb_bat,[batch_size*train_view_num, 1, 127,127,3])
					print("single_view_train_rgb_input_shape ",rgb.shape)
					vox = np.tile(Y_vox_bat[:,None,:,:,:],[1,train_view_num,1,1,1])
					vox = np.reshape(vox, [batch_size*train_view_num, 32,32,32])
#					_, rec_loss_c, sum_train,xxx,sum_train_histo = self.sess.run([self.base_optim,self.rec_loss,self.sum_merged,self.refine_optim,self.sum_histo],
					_, rec_loss_c, sum_train,xxx = self.sess.run([self.base_optim,self.rec_loss,self.merged,self.refine_optim],feed_dict={self.X_rgb: rgb, self.Y_vox: vox, self.lr: 0.0001})
					print ('ep:', epoch, 'i:', i, 'train single rec loss:', rec_loss_c)
                                        									
				########## multi view train
				if multi_view_train:
#					rec_loss_c, _, sum_train,xxx,sum_train_histo = self.sess.run([self.rec_loss, self.att_optim, self.sum_merged,self.refine_optim,self.sum_histo],
					rec_loss_c, _, sum_train,xxx = self.sess.run([self.rec_loss, self.att_optim, self.merged,self.refine_optim],feed_dict={self.X_rgb: X_rgb_bat, self.Y_vox: Y_vox_bat,self.lr: 0.0001})
					print ('ep:', epoch, 'i:', i, 'train multi rec loss:', rec_loss_c)
                                        				
				############
				if epoch % 1 == 0:
					self.sum_writer_train.add_summary(sum_train, epoch * total_train_batch_num + i)
#					self.sum_writer_train.add_summary(sum_train_histo, epoch * total_train_batch_num + i)
				
				#### testing
				if epoch > 150 :
					X_rgb_batch, Y_vox_batch = data.load_X_Y_test_next_batch(test_mv=1)
					rec_loss_te, Y_vox_test_pred, att_pred, sum_test = \
						self.sess.run([self.rec_loss, self.Y_pred,self.weights, self.sum_merged],
					    feed_dict={self.X_rgb: X_rgb_batch, self.Y_vox: Y_vox_batch})
					X_rgb_batch = X_rgb_batch.astype(np.float16)
					Y_vox_batch = Y_vox_batch.astype(np.float16)
					Y_vox_test_pred = Y_vox_test_pred.astype(np.float16)
					att_pred = att_pred.astype(np.float16)
					to_save = {'X_test':X_rgb_batch,'Y_test_pred':Y_vox_test_pred,'att_pred':att_pred,'Y_test_true':Y_vox_batch}
					scipy.io.savemat(self.test_res_dir+'X_Y_pred_'+str(epoch).zfill(2)+'_'+str(i).zfill(5)+'.mat',to_save,do_compression=True)
					self.sum_writer_test.add_summary(sum_test, epoch * total_train_batch_num + i)
					print ('ep:', epoch, 'i:', i, 'test rec loss:', rec_loss_te)
					
				#### model saving
				if epoch % 10 == 0 and epoch > 180:
					self.saver.save( self.sess, save_path=self.train_mod_dir + 'model.cptk' )
					print ( 'epoch:', epoch, 'i:', i, 'model saved!' )
  
#				summary = self.sess.run(self.merged)                          
#				self.sum_writer_test.add_summary(summary,i)				

##########
if __name__ =='__main__':

		net = Network()          #net=object to create instance

		print("network compleated")   ###

		net.build_graph()
		print("graph compleated")
                
#               sys.exit(). sys.exit()        ###
		
		data = tools.Data(config)
		print("tools.data compleated")

                
		print('trianing data')
		
		net.train(data)

##########################################----TensorBoard----###########################################
#		with tf.Session() as sess:
#			merge = tf.summary.merge_all()            
#			writer = tf.summary.FileWriter('./graphs/train', sess.graph)
#			for step in range(3):
#				summary=sess.run(merge)
#				writer.add_summary(summary,step)
			
           


	

	
