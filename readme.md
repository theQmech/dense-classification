# Dense Image Classifiaction

Semantic Segmentation is a term used in Computer Vision which refers to 
partitioning an image into meaningful parts and at the same time classify 
each partition into one of the predetermined labels.

Traditionally, CNNs have been used to assign labels to image(Image Classification).
Here, we apply CNNs to the task of Semantic Segmentation by extracting features
using AlexNet and then applying a neural net on these features.

## Setting up

Use `pip install -r requirements.txt` to install necessary libraries.

Another dependency not listed is the [caffe](https://github.com/BVLC/caffe) framework.
Make sure that the system variable `CAFFE_ROOT` points to your installation of `caffe`.

Use script `get_data.sh` to download dataset.

Make a directory `saved/` where precomputed information of the pipeline will be
stored. `mkdir -p saved/`

The following command will run the pipeline: `python Main.py -p dataset/ 2>/dev/null`

Redirecting `stderr` to `/dev/null` isn't necessary, but saves cluttering by redirecting the logs.
