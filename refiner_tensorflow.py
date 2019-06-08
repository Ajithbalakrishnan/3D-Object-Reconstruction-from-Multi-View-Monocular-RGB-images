def refiner_network(volumes)
	
	input_volumes_32 = tf.reshape(volumes, [-1, vox_res32, vox_res32, vox_res32,1])
	
	l1=tools.Ops.conv3d(input_volumes_32, k=4, out_c=32, str=1, name='ref_c1')
	l2=tf.nn.batch_normalization(l1, name='ref_b1')
	l3=tools.Ops.xxlu(l2,label='lrelu')
	l4=tools.Ops.maxpool3d(l3, k=2, s=1, name='ref_m1')
	
	l5=tools.Ops.conv3d(l4, k=4, out_c=32, str=1, name='ref_c1')
	l6=tf.nn.batch_normalization(l1, name='ref_b1')
	l7=tools.Ops.xxlu(l2,label='lrelu')
	l8=tools.Ops.maxpool3d(l3, k=2, s=1, name='ref_m1')
	
	
	l1=tools.Ops.conv3d(input_volumes_32, k=4, out_c=32, str=1, name='ref_c1')
	l2=tf.nn.batch_normalization(l1, name='ref_b1')
	l3=tools.Ops.xxlu(l2,label='lrelu')
	l4=tools.Ops.maxpool3d(l3, k=2, s=1, name='ref_m1')
	
	
	
	
