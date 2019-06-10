def refiner_network(volumes):
	
	input_volumes_32 = tf.reshape(volumes, [-1, vox_res32, vox_res32, vox_res32, 1])
	
	rn1=tools.Ops.conv3d(input_volumes_32, k=4, out_c=32, str=1, name='ref_c1')
	rn2=tf.nn.batch_normalization(rn1, name='ref_b1')
	rn3=tools.Ops.xxlu(rn2,label='lrelu')
	volumes_16_l =tools.Ops.maxpool3d(rn3, k=2, s=1, name='ref_m1')
	
	rn5=tools.Ops.conv3d(rn4, k=4, out_c=64, str=1, name='ref_c2')
	rn6=tf.nn.batch_normalization(rn5, name='ref_b2')
	rn7=tools.Ops.xxlu(rn6,label='lrelu')
	volumes_8_l =tools.Ops.maxpool3d(rn7, k=2, s=1, name='ref_m2')
		
	rn9=tools.Ops.conv3d(rn8, k=4, out_c=128, str=1, name='ref_c3')
	rn10=tf.nn.batch_normalization(rn9, name='ref_b3')
	rn11=tools.Ops.xxlu(rn10,label='lrelu')
	volumes_4_l =tools.Ops.maxpool3d(rn11, k=2, s=1, name='ref_m3')


	fc1 = tools.Ops.xxlu(tools.Ops.fc(rn12, out_d=2048, name='fc1'), label='relu')
	
	fc2 = tools.Ops.xxlu(tools.Ops.fc(fc1, out_d=8192, name='fc2'), label='relu')
	        
        reshaped_1 = volumes_4_l+tf.reshape(fc2, [-1, 4, 4, 4, 128])

        rn13= tools.Ops.deconv3d(reshaped_1, k=4, out_c=64, str=2, name='ref_c4')
        rn14=tf.nn.batch_normalization(rn13, name='ref_b4')
	volumes_4_r=tools.Ops.xxlu(rn14,label='lrelu')

        reshaped_2=volumes_4_r+volumes_8_l

        rn16= tools.Ops.deconv3d(reshaped_2, k=4, out_c=32, str=2, name='ref_c5')
        rn17=tf.nn.batch_normalization(rn16, name='ref_b5')
	volumes_8_r =tools.Ops.xxlu(rn17,label='lrelu')
 
        reshaped_3=volumes_8_r+volumes_16_l
        
        rn19= tools.Ops.deconv3d(reshaped_3, k=4, out_c=1, str=2, name='ref_c6')
        volumes_16_r= tf.nn.sigmoid(rn19)

        reshape_4=(volumes_16_r+input_volumes_32)*0.5
        
        reshape_5=tf.reshape(reshape_4, [-1, vox_res32, vox_res32, vox_res32])

        return reshape_5

	
	
