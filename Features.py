from __future__ import print_function
import shelve, os, sys, cv2
import numpy as np
from skimage.util import img_as_float
from skimage import io, feature
from skimage.segmentation import slic, mark_boundaries
from collections import Counter
os.environ["GLOG_logtostderr"] = "true"
os.environ["GLOG_minloglevel"] = "1" #suppress caffe output
import caffe


EPS = 1e-7
LBP_NPTS = 8
RADIUS = 5

SLIC_SEGS = 100
SLIC_SIGMA = 1


def extract_features(base_path, img_name, segments):
	left_img = base_path+"left/"+img_name+".jpg"
	labeled_img = base_path+"labeled/"+img_name+".png"

	lbp = feature.local_binary_pattern(
			cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_RGB2GRAY), 
			LBP_NPTS,
			RADIUS, 
			method="uniform"
		)

	image = img_as_float(io.imread(left_img))
	# segments = saved_segments[img_name]

	labels = io.imread(labeled_img)

	data_X = []
	data_Y = []
	INC_BITMAP = []

	for (i, segVal) in enumerate(np.unique(segments)):
		(hist, _) = np.histogram(
						lbp[segments == segVal].ravel(),
						bins=np.arange(0, LBP_NPTS + 3),
						range=(0, LBP_NPTS + 2)
					)
		hist = hist.astype("float")
		hist /= (hist.sum() + EPS)

		rgb_average = np.average(image[segments==segVal], 0)

		X = (hist.tolist() + rgb_average.tolist())
		Y = int(Counter(labels[segments == segVal]).most_common(1)[0][0])

		if Y>0:
			data_X.append(X)
			data_Y.append(np.eye(1, 10, Y-1)[0])
			INC_BITMAP.append(True)
		else:
			INC_BITMAP.append(False)

	return (data_X, data_Y, INC_BITMAP)

def extract_alex_features(base_path, img_name, segments):
	segments = np.asarray(segments)

	left_img = base_path+"left/"+img_name+".jpg"
	labeled_img = base_path+"labeled/"+img_name+".png"

	CAFFE_ROOT = os.environ['CAFFE_ROOT']
	CAFFENET_ROOT = CAFFE_ROOT + "models/bvlc_reference_caffenet/"

	model_file = CAFFENET_ROOT + "bvlc_reference_caffenet.caffemodel"
	deploy_prototxt = CAFFENET_ROOT + "deploy.prototxt"

	net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)

	layer = 'fc7'
	if layer not in net.blobs:
		raise TypeError("Invalid layer name: " + layer)

	imagemean_file = CAFFE_ROOT + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255.0)

	net.blobs['data'].reshape(1,3,227,227)

	n_segs = np.amax(segments) + 1

	base_img = caffe.io.load_image(left_img)
	labels = io.imread(labeled_img)

	data_X = []
	data_Y = []
	INC_BITMAP = []

	for i in range(n_segs):
		mask = (segments == i)
		rows = np.any(mask, axis=1)
		cols = np.any(mask, axis=0)
		rmin, rmax = np.where(rows)[0][[0, -1]]
		cmin, cmax = np.where(cols)[0][[0, -1]]

		img = base_img[rmin:rmax+1, cmin:cmax+1]
		net.blobs['data'].data[...] = transformer.preprocess('data', img)
		output = net.forward()

		X = net.blobs[layer].data[0]
		Y = int(Counter(labels[segments == i]).most_common(1)[0][0])

		if Y>0:
			data_X.append(X)
			data_Y.append(np.eye(1, 10, Y-1)[0])
			INC_BITMAP.append(True)
		else:
			INC_BITMAP.append(False)

	return (data_X, data_Y, INC_BITMAP)

def get_saved_segments(files):
	print ("\nChecking for saved segments data")

	db = shelve.open('saved/segments.data', 'cwb')
	extract = True

	if 'status' in db:
		if db['status']:
			extract = False

	if extract:
		print ("Saved segments not found. Segments will be calculated\n")
	else:
		print ("Saved segments found, reusing them\n")

	saved_segments = {}
	if extract:
		for (n, img_path) in enumerate(files):
			image = img_as_float(io.imread(img_path))
			img_name = os.path.split(img_path)[1].split(".")[0]
			print (" "*40 + "\r", end='')
			print ("%.02f%%\t(%d/%d)\tSLIC: [%s]\r" % ((n+1.0)/len(files)*100.0, (n+1.0), len(files), img_path), end='')
			saved_segments[img_name] = slic(image, n_segments = SLIC_SEGS, sigma = SLIC_SIGMA)
			sys.stdout.flush()
		print ("\n")
		db['segments'] = saved_segments
		db.sync()
		db['status'] = True
		db.sync()
	else:
		saved_segments = db['segments']

	db.close()
	return saved_segments


def get_saved_features(files):
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
			print (" "*40 + "\r", end='')
			print ("%.02f%%\t(%d/%d)\tExtracting features: [%s]"%((n+1.0)/len(files)*100.0, (n+1.0), len(files), img_path), end='')
			[temp_X, temp_Y, _] = extract_alex_features(args["path"]+"/" ,img_name, saved_segments[img_name])
			data_X, data_Y = data_X + temp_X, data_Y + temp_Y
			sys.stdout.flush()
			saved_features[img]
		print ("\n")
		train_X, train_Y = np.array(data_X), np.array(data_Y)
		db['features'] = (train_X, train_Y)
		db.sync()
		db['status'] = True
		db.sync()
		print ("Features saved")
	else:
		(train_X, train_Y) = db['features']

	db.close()