#!/usr/bin/env python
from __future__ import print_function
from Features import *
from ConfMatrix import *

import argparse
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random
import matplotlib.pyplot as plt

N_FOLD = 1

cnf_matrix = np.zeros((10, 10))
correct = incorrect = 0

#==============================================================================#
#                         INITIALIZATIONS                                      #
#==============================================================================#


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", 
		required=False, help = "root directory of dataset",
		default="./dataset/")
args = vars(ap.parse_args())

left_dir, disp_dir, labeled_dir = [(args["path"]+x) for x in ("left/", "disp/", "labeled/")]
files = [f.split(".")[0] for f in listdir(labeled_dir) if isfile(join(labeled_dir, f))]

saved_segments = get_saved_segments([ (left_dir+x) for x in 
						listdir(left_dir) if isfile(join(left_dir, x))])

#############################################################

db = shelve.open('saved/features.data', 'cwb')
extract = True

if 'status' in db:
	if db['status']:
		extract = False

if extract:
	print ("Saved features not found. Features will be calculated\n")
else:
	print ("Saved features found, reusing them\n")

saved_features = {}
if extract:
	db['features'] = {}
	for (n, img_path) in enumerate(files):
		img_name = os.path.split(img_path)[1].split(".")[0]
		print (" "*40 + "\r", end='')
		print ("%.02f%%\t(%d/%d)\tExtracting features: [%s]"%((n+1.0)/len(files)*100.0, (n+1.0), len(files), img_path), end='')
		saved_features[img_name] = extract_alex_features(args["path"]+"/", img_name, saved_segments[img_name])
		sys.stdout.flush()
	print ("\n")
	db['features'] = saved_features
	db.sync()
	db['status'] = True
	db.sync()
	print ("Features saved")
else:
	saved_features = db['features']

db.close()
#############################################################

rand_ord = range(len(files))
random.seed(448)
random.shuffle(rand_ord)
# rand_ord is random permutation of [1...n]

for kfold in range(N_FOLD):

	print ("Validation #%d:"%(kfold+1))

	# PARTITION DATA

	# rotate rand_ord and extract train, test files
	# rand_ord = rand_ord[len(files)/4:] + rand_ord[:len(files)/4]
	# train_files = [files[i] for i in rand_ord[len(files)/4:]]
	# test_files = [files[i] for i in rand_ord[:len(files)/4]]

	train_files = files[:int(len(files)*0.8)]
	test_files = files[int(len(files)*0.8):]

	# EXTRACT FEATURES

	data_X, data_Y = [], []
	for img_name in train_files:
		[temp_X, temp_Y, _] = saved_features[img_name]
		data_X = data_X + temp_X
		data_Y = data_Y + temp_Y
	train_X = np.array(data_X)
	train_Y = np.array(data_Y)


	# TRAIN THE NEURAL NETS

	# NUMBER OF FEATURES
	print (train_X.shape)
	N_INPS = train_X.shape[1]

	model = Sequential()
	model.add(Dense(512, input_dim=N_INPS, init='uniform', activation='relu'))
	model.add(Dense(64, init='uniform', activation='relu'))
	# model.add(Dense(1024, init='uniform', activation='relu'))
	# model.add(Dense(256, init='uniform', activation='relu'))
	model.add(Dense(32, init='uniform', activation='relu'))
	model.add(Dense(10, init='uniform', activation='relu'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	print ("\nLooking for saved weights...")
	if os.path.isfile("saved/model.h5"):
		model.load_weights("saved/model.h5")
		print ("Using Saved weights\n")
	else:
		print ("Saved weights not found. Training model...")
		model.fit(train_X, train_Y, nb_epoch=100, batch_size=50)
		model.save_weights("saved/model.h5")

	# EVALUATE

	for (n, img_name) in enumerate(test_files):

		# pred_segment = predicted class for each labelled segment
		[temp_X, temp_Y, bitmap] = saved_features[img_name]
		temp_X = np.asarray(temp_X)
		temp_Y = np.asarray(temp_Y)
		print (temp_Y.shape)
		print (temp_X.shape)
		pred_segment = model.predict(temp_X)
		pred_segment = [np.argmax(b)+1 for b in pred_segment]

		combined_labels = []
		i = j = 0
		for i in range(len(bitmap)):
			if (bitmap[i]):
				combined_labels.append(pred_segment[j])
				j += 1
			else:
				combined_labels.append(0)

		# print ("==============")
		# temp_cpy = np.asarray(saved_segments[img_name])
		# minval = np.amin(temp_cpy)
		# maxval = np.amax(temp_cpy)
		# print (minval, maxval)
		# # print (combined_labels)
		# print (len(saved_segments[img_name]))
		# print (len(saved_segments[img_name][0]))
		# print (len(combined_labels))
		# print (len(bitmap))

		# pred_labels = [ [combined_labels[y] for y in x] for x in saved_segments[img_name]]
		pred_labels = saved_segments[img_name]
		for row in range(len(pred_labels)):
			for col in range(len(pred_labels[row])):
				# print ("******")
				# print (row, col)
				# print (row, col, pred_labels[row][col])
				# print (row, col, pred_labels[row][col], combined_labels[pred_labels[row][col]])
				pred_labels[row][col] = combined_labels[pred_labels[row][col]]
		ground_truth = io.imread(labeled_dir+img_name+".png")

		pred_labels = np.asarray(pred_labels)
		ground_truth = np.asarray(ground_truth)

		for i in range(10):
			img_obj = cv2.imread(left_dir+img_name+".jpg")
			b_channel, g_channel, r_channel = cv2.split(img_obj)
			alpha_channel = (pred_labels==(i+1)).astype(int)*255
			for row in range(len(r_channel)):
				for col in range(len(r_channel[0])):
					r_channel[row][col] = (pred_labels[row][col] == (i+1)).astype(int) * 255
			img_RGBA = cv2.merge((b_channel, g_channel, r_channel))
			cv2.imwrite(args["path"]+"pred/"+img_name+"_"+str(i+1)+".jpg", img_RGBA)

		pred_labels = [item for sublist in pred_labels for item in sublist]
		ground_truth = [item for sublist in ground_truth for item in sublist]

		for i in range(len(pred_labels)):
			if pred_labels[i]>0 and ground_truth[i]>0:
				cnf_matrix[pred_labels[i]-1][ground_truth[i]-1] += 1
				if pred_labels[i] == ground_truth[i]:
					correct += 1
				else:
					incorrect += 1

print ("\n\n Confusion Matrix \n\n")
print (cnf_matrix)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')
plt.show()

print ("Correct Predictions: %d\nIncorrect Predictions: %d\n"%(correct, incorrect))
print ("Pixel-wise Accuracy: %.02f%%"%(((correct+0.0)/(correct+incorrect))*100.0))
