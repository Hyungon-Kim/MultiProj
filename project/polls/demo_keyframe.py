from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import pickle
import argparse
#from src import classifier
import dlib
import os
from polls import align_dlib
from polls import facenet
import math
from scipy import misc
import sys
from polls import align_dataset_mtcnn

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

INV_TEMPLATE = np.float32([
                            (-0.04099179660567834, -0.008425234314031194, 2.575498465013183),
                            (0.04062510634554352, -0.009678089746831375, -1.2534351452524177),
                            (0.0003666902601348179, 0.01810332406086298, -0.32206331976076663)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
landmarkIndices = [39, 42, 57]
AD = align_dlib.AlignDlib('C:/Users/mmlab/django/mysite/static/shape_predictor_68_face_landmarks.dat')

# Load the model

input_path = 'C:/Users/mmlab/django/mysite/media/'
classifier_filename = 'C:/Users/mmlab/django/mysite/static/celeb_self_1000.pkl'


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = 'C:/Users/mmlab/django/mysite/static/20171019-185655/20171019-185655.pb'
            facenet.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                # output_path = input_path+'_out'
                # align_dataset_mtcnn.main(input_path)


            images = os.listdir(input_path)
            #AD = align_dlib.AlignDlib('C:/Users/mmlab/django/mysite/static/shape_predictor_68_face_landmarks.dat')
            bb = dlib.rectangle(left=0,top=0,right=160,bottom=160)
                #----------------------------------------face align-------------------------------------------
                # with tf.Graph().as_default():
                   # model = 'C:/Users/mmlab/django/mysite/static/20171019-185655'
                   # # Load the model
                   # print('Loading feature extraction model')
                   # facenet.load_model(model)
                   # Get input and output tensors
            batch_size = 600
            face_images = []
            face_location = []
            frame_name = []
            # classifier_filename = 'C:/Users/mmlab/django/mysite/static/celeb_self_1000.pkl'
            # classifier_filename_exp = os.path.expanduser(classifier_filename)
            # Classify images
            print('Testing classifier')

            image= misc.imread(input_path+'/'+args)                # Run forward pass to calculate embeddings
            print ('Calculating features fo images')

                # if (bb.right()-bb.left() < 80 or bb.bottom()-bb.top()<80) :
                #     continue
                # landmarks = AD.findLandmarks(image,bb)
                # npLandmarks = np.float32(landmarks)
                # npLandmarkIndices = np.array(landmarkIndices)
                # H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices],160 * MINMAX_TEMPLATE[npLandmarkIndices]* float(110/160)+160*(1-float(110/160))/2)
                # thumbnail = cv2.warpAffine(image, H, (160, 160))

            face_images.append(image)
            frame_name.append(args)


            nrof_images = len(face_images)
            faces = np.zeros((nrof_images,160,160,3))
            for i in range(nrof_images) :
                if face_images[i].ndim == 2:
                    face_images[i] = facenet.to_rgb(face_images[i])
                img = facenet.prewhiten(face_images[i])
                faces[i,:,:,:] = img


            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                face_index = faces[start_index:end_index,:,:,:]
                feed_dict = {images_placeholder: face_index, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)

            #x,y,w,h

            #top right bottom left
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            #f = open('C:/Users/mmlab/Desktop/demo_roi/result_1sec3.txt','w')
            for i in range(len(best_class_indices)) :
                if (best_class_probabilities[i] > 0.02):
                    print('image name : %s , class : %s  %.3f' % (frame_name[i] , class_names[best_class_indices[i]], best_class_probabilities[i]))

    return class_names[best_class_indices[0]], frame_name[0]
                    #f.write('%s,%d,%d,%d,%d,%s,%.3f\n'%(frame_name[i],x,y,w,h,class_names[best_class_indices[i]],best_class_probabilities[i]))
                #print (face_location[i])


                    #f.close()
    cv2.destroyAllWindows()

