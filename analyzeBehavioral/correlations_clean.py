import numpy as np
import os
from sklearn import svm
from sklearn import preprocessing
import pickle
import scipy.io
import pdb

# caffenary = cnn_base.CaffeTools(deploy='models/finetune_identity/deploy_100.prototxt', model='models/finetune_identity/100_elevations_lights_iter_270000.caffemodel')

base_dir = './'
# read the dataset file to get ground truth
tr = np.loadtxt(base_dir+'behavioral_face_set_stimuli.txt')
ground_truth = np.ndarray(shape=(96))
for item, i in zip(tr, range(96)):
	if item[0] == item[1]:
		ground_truth[i] = 1
	else:
		ground_truth[i] = -1


# Get the features
db_dir = base_dir+'images_wbg/'
prediction = np.ndarray(shape=(96))
distance = np.ndarray(shape=(96))
train_features = np.ndarray(shape=(96, 60*2))
test_features = np.ndarray(shape=(96, 60*2))

mean_features = scipy.io.loadmat('mean.mat')
mean_features = np.transpose(mean_features['x'])

sigma_features = scipy.io.loadmat('sigma.mat')
sigma_features =  np.transpose(sigma_features['x'])

counter=0
for i in range(96):
	train_features[i,:] = np.hstack((mean_features[counter,:], sigma_features[counter,:]))
	counter+=1
	test_features[i,:] = np.hstack((mean_features[counter,:], sigma_features[counter,:]))
	counter+=1
	# train_image = db_dir+'pair_'+str(i+1)+'_0.png'
	# test_image = db_dir+'pair_'+str(i+1)+'_1.png'
	# train_features[i, :] = caffenary.get_features(filename=train_image)
	# test_features[i, :] = caffenary.get_features(filename=test_image)

train_features = preprocessing.scale(train_features)
test_features = preprocessing.scale(test_features)
	
# get correlations b/w features for the train and test images
corrs = np.ndarray(shape=(96))
prediction = np.ndarray(shape=(96))
for i in range(96):
	corrs[i] = np.corrcoef(train_features[i,:], test_features[i,:])[0,1]

# search for the best threshold
results = np.ndarray(shape=(201))
for level, l in zip(range(202), np.linspace(-1, 1, 200)):
	prediction[corrs < l] = -1
	prediction[corrs >= l] = 1
	results[level] = sum(prediction == ground_truth)

print results.max()

# get actual responses from the model for the best threshold
l = np.linspace(-1, 1, 200)[results.argmax()]
prediction[corrs < l] = 0
prediction[corrs >= l] = 1








