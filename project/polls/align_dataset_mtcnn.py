"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from polls import facenet
from polls import detect_face
import random
from time import sleep

def main(path):
    sleep(random.random())

    output_dir = path+'_out'
    path = path

    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = os.listdir(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    #bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_1sec.txt')


    nrof_images_total = 0
    nrof_successfully_aligned = 0

    random.shuffle(dataset)

    for image_path in dataset:
        image_path = os.path.join(path,image_path)
        nrof_images_total += 1
        filename = os.path.splitext(os.path.split(image_path)[1])[0]
        output_filename = os.path.join(output_dir, filename+'.png')
        print(output_filename)
        if not os.path.exists(output_filename):
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim<2:
                    print('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:,:,0:3]

                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces>0:
                    for i in range(nrof_faces) :
                        det = bounding_boxes[i,0:4]
                        img_size = np.asarray(img.shape)[0:2]

                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-25,0) #left
                        bb[1] = np.maximum(det[1]-25,0) #top
                        bb[2] = np.minimum(det[2]+25,img_size[1]) #right
                        bb[3] = np.minimum(det[3]+25,img_size[0]) #bottom
                        print (bb[0], bb[1], bb[2], bb[3])
                        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                        scaled = misc.imresize(cropped, (160,160), interp='bilinear')
                        misc.imsave(os.path.join(output_dir, filename+'_'+str(i)+'.png'), scaled)

                    # det = bounding_boxes[:,0:4]
                    # img_size = np.asarray(img.shape)[0:2]
                    # if nrof_faces>1:
                    #     bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    #     img_center = img_size / 2
                    #     offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    #     offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    #     index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    #     det = det[index,:]
                    # det = np.squeeze(det)
                    # bb = np.zeros(4, dtype=np.int32)
                    # bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    # bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    # bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    # bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                    # cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    # scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                    # nrof_successfully_aligned += 1
                    # misc.imsave(output_filename, scaled)
                    # text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
                else:
                    print('Unable to align "%s"' % image_path)
                    #text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    sess.close()