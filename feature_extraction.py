import caffe
import numpy as np


extract_from_layer_alex_net = "fc7"

extract_from_layer_google_net = "pool5/7x7_s1"

model_folder = "CaffeModel"

model_def_alex_net= "deploy_alexnet.prototxt.txt"

model_def_google_net= "deploy_googlenet.prototxt.txt"

pretrained_model_alex_net="bvlc_alexnet.caffemodel" 

pretrained_model_google_net="bvlc_googlenet.caffemodel" 


def extract_features(UPLOAD_FOLDER,input_images_file):
	# change based on your deploy.prototxt file 
	batch_size = 10

	images_loaded_by_caffe= caffe.io.load_image(UPLOAD_FOLDER+"/"+input_images_file)

	# Create a net object 

	net_alex_net = caffe.Net(model_folder+"/"+model_def_alex_net, model_folder+"/"+ pretrained_model_alex_net, caffe.TEST)

	net_google_net = caffe.Net(model_folder+"/"+model_def_google_net, model_folder+"/"+pretrained_model_google_net, caffe.TEST)



	# set up transformer - creates transformer object 

	transformer_alex_net = caffe.io.Transformer({'data': net_alex_net .blobs['data'].data.shape})

	transformer_google_net = caffe.io.Transformer({'data': net_google_net.blobs['data'].data.shape}) 


	# transpose image from HxWxC to CxHxW

	transformer_alex_net.set_transpose('data', (2,0,1)) 

	transformer_google_net.set_transpose('data', (2,0,1))
	# swap image channels from RGB to BGR

	transformer_alex_net.set_channel_swap('data', (2,1,0)) 

	transformer_google_net.set_channel_swap('data', (2,1,0)) 


	transformer_alex_net.set_raw_scale('data', 255)

	transformer_google_net.set_raw_scale('data', 255)

	images_loaded_by_caffe = np.array(images_loaded_by_caffe)  


	net_alex_net.blobs['data'].data[0] =  transformer_alex_net.preprocess('data', images_loaded_by_caffe)


	net_google_net.blobs['data'].data[0] = transformer_google_net.preprocess('data', images_loaded_by_caffe)


	features_for_alex_net = net_alex_net.blobs[extract_from_layer_alex_net].data[0].copy()

	features_for_alex_net=np.reshape(features_for_alex_net, (1, len(features_for_alex_net)))



	features_for_google_net = net_google_net.blobs[extract_from_layer_google_net].data[0].copy()

	features_for_google_net=np.reshape(features_for_google_net, (1, len(features_for_google_net)))


	features=np.append(features_for_alex_net,features_for_google_net) 

	np.savetxt("input_Dir/"+input_images_file+"output.csv", features, delimiter=",")


