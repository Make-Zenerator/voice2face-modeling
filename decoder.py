import torch.nn as nn

import torchvision
from torch import nn
import psutil
from torchvision import datasets, models, transforms
import torch

PATH = "http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth"
vgg16 = models.vgg16(pretrained=PATH)
new_classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])
vgg16.classifier = new_classifier
vgg16.cuda()

import os
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from time import time
import face_recognition
from random import randint

class Decoder_Dataset(Dataset):
    def __init__(self, Folder, VggL, sample, dev = "cuda"):

        self.root     = Folder
        self.dev      = dev 
        self.names    = os.listdir(self.root) 
        self.sample   = sample if sample < 20 else 20
        self.a_length = len(os.listdir(self.root)) 
        self.length   = self.a_length * self.sample
        self.img_name = self.root + "{}/{}.jpg"
        self.vgg_features = VggL

    def image_out(self,path, mean =None , std = [0.229, 0.224, 0.225]):#[0.485, 0.456, 0.406]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        count = 0
        resultant = []
        try:
            faceLocation = face_recognition.face_locations(img)[count]
            x,y1,x1,y = faceLocation
            img = img[x:x1,y:y1]
            landmark = face_recognition.face_landmarks(img)
            for i in list(landmark[0].keys()):
                resultant += landmark[0][i]
            landmark = np.ravel(np.array(resultant))
            img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
            biden_encoding = face_recognition.face_encodings(img)[0]
        except IndexError:
            return None, None, None
        
        
        return  biden_encoding, img.reshape(1,224,224,3), np.ravel(np.array(landmark))



    def pseudo_idx(self,idx):
        if idx < self.a_length:
            return idx
        else:
            ## return self.pseudo_idx(idx - self.a_length) ## RECURSION
            return idx // self.sample 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        while True:

            index = idx
            index = self.pseudo_idx(index)
            name  = idx // self.a_length
            path =  self.img_name.format(self.names[index], name)

            if not os.path.exists(path):
                idx = randint(0, self.a_length) # IF FILE PATH DOESNOT EXISTS
                continue

            feature, image, landmark= self.image_out(path)
            if image is None:
                idx = randint(0, self.a_length) # IF FILE PATH DOESNOT EXISTS
                continue

            image = torch.from_numpy(image).view(3,224,224).float()
            return   torch.from_numpy(feature).float(), image,torch.from_numpy(landmark).float()


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.compat import dimension_value
from tensorflow.contrib.image import dense_image_warp
from tensorflow.contrib.image import interpolate_spline

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def _to_float32(value):
    return tf.cast(value, tf.float32)

def _to_int32(value):
    return tf.cast(value, tf.int32)

def _get_grid_locations(image_height, image_width):
    """Wrapper for np.meshgrid."""
    tfv1.assert_type(image_height, tf.int32)
    tfv1.assert_type(image_width, tf.int32)

    y_range = tf.range(image_height)
    x_range = tf.range(image_width)
    y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing='ij')
    return tf.stack((y_grid, x_grid), -1)


def _expand_to_minibatch(tensor, batch_size):
    """Tile arbitrarily-sized np_array to include new batch dimension."""
    ndim = tf.size(tf.shape(tensor))
    ones = tf.ones((ndim,), tf.int32)

    tiles = tf.concat(([batch_size], ones), 0)
    return tf.tile(tf.expand_dims(tensor, 0), tiles)


def _get_boundary_locations(image_height, image_width, num_points_per_edge):
    """Compute evenly-spaced indices along edge of image."""
    image_height_end = _to_float32(tf.math.subtract(image_height, 1))
    image_width_end = _to_float32(tf.math.subtract(image_width, 1))
    y_range = tf.linspace(0.0, image_height_end, num_points_per_edge + 2)
    x_range = tf.linspace(0.0, image_height_end, num_points_per_edge + 2)
    ys, xs = tf.meshgrid(y_range, x_range, indexing='ij')
    is_boundary = tf.logical_or(
        tf.logical_or(tf.equal(xs, 0.0), tf.equal(xs, image_width_end)),
        tf.logical_or(tf.equal(ys, 0.0), tf.equal(ys, image_height_end)))
    return tf.stack([tf.boolean_mask(ys, is_boundary), tf.boolean_mask(xs, is_boundary)], axis=-1)


def _add_zero_flow_controls_at_boundary(control_point_locations,
                                        control_point_flows, image_height,
                                        image_width, boundary_points_per_edge):
    """Add control points for zero-flow boundary conditions.
     Augment the set of control points with extra points on the
     boundary of the image that have zero flow.
    Args:
      control_point_locations: input control points
      control_point_flows: their flows
      image_height: image height
      image_width: image width
      boundary_points_per_edge: number of points to add in the middle of each
                             edge (not including the corners).
                             The total number of points added is
                             4 + 4*(boundary_points_per_edge).
    Returns:
      merged_control_point_locations: augmented set of control point locations
      merged_control_point_flows: augmented set of control point flows
    """

    batch_size = dimension_value(tf.shape(control_point_locations)[0])

    boundary_point_locations = _get_boundary_locations(image_height, image_width,
                                                       boundary_points_per_edge)
    boundary_point_shape = tf.shape(boundary_point_locations)
    boundary_point_flows = tf.zeros([boundary_point_shape[0], 2])

    minbatch_locations = _expand_to_minibatch(boundary_point_locations, batch_size)
    type_to_use = control_point_locations.dtype
    boundary_point_locations = tf.cast(minbatch_locations, type_to_use)

    minbatch_flows = _expand_to_minibatch(boundary_point_flows, batch_size)

    boundary_point_flows = tf.cast(minbatch_flows, type_to_use)

    merged_control_point_locations = tf.concat(
        [control_point_locations, boundary_point_locations], 1)

    merged_control_point_flows = tf.concat(
        [control_point_flows, boundary_point_flows], 1)

    return merged_control_point_locations, merged_control_point_flows


def sparse_image_warp(image,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundary_points=0,
                      name='sparse_image_warp'):
    """Image warping using correspondences between sparse control points.
    Apply a non-linear warp to the image, where the warp is specified by
    the source and destination locations of a (potentially small) number of
    control points. First, we use a polyharmonic spline
    (`tf.contrib.image.interpolate_spline`) to interpolate the displacements
    between the corresponding control points to a dense flow field.
    Then, we warp the image using this dense flow field
    (`tf.contrib.image.dense_image_warp`).
    Let t index our control points. For regularization_weight=0, we have:
    warped_image[b, dest_control_point_locations[b, t, 0],
                    dest_control_point_locations[b, t, 1], :] =
    image[b, source_control_point_locations[b, t, 0],
             source_control_point_locations[b, t, 1], :].
    For regularization_weight > 0, this condition is met approximately, since
    regularized interpolation trades off smoothness of the interpolant vs.
    reconstruction of the interpolant at the control points.
    See `tf.contrib.image.interpolate_spline` for further documentation of the
    interpolation_order and regularization_weight arguments.
    Args:
      image: `[batch, height, width, channels]` float `Tensor`
      source_control_point_locations: `[batch, num_control_points, 2]` float
        `Tensor`
      dest_control_point_locations: `[batch, num_control_points, 2]` float
        `Tensor`
      interpolation_order: polynomial order used by the spline interpolation
      regularization_weight: weight on smoothness regularizer in interpolation
      num_boundary_points: How many zero-flow boundary points to include at
        each image edge.Usage:
          num_boundary_points=0: don't add zero-flow points
          num_boundary_points=1: 4 corners of the image
          num_boundary_points=2: 4 corners and one in the middle of each edge
            (8 points total)
          num_boundary_points=n: 4 corners and n-1 along each edge
      name: A name for the operation (optional).
      Note that image and offsets can be of type tf.half, tf.float32, or
      tf.float64, and do not necessarily have to be the same type.
    Returns:
      warped_image: `[batch, height, width, channels]` float `Tensor` with same
        type as input image.
      flow_field: `[batch, height, width, 2]` float `Tensor` containing the dense
        flow field produced by the interpolation.
    """

    image = ops.convert_to_tensor(image)
    source_control_point_locations = ops.convert_to_tensor(
        source_control_point_locations)
    dest_control_point_locations = ops.convert_to_tensor(
        dest_control_point_locations)

    control_point_flows = (
        dest_control_point_locations - source_control_point_locations)

    clamp_boundaries = num_boundary_points > 0
    boundary_points_per_edge = num_boundary_points - 1

    with ops.name_scope(name):
        image_shape = tf.shape(image)
        batch_size, image_height, image_width = image_shape[0], image_shape[1], image_shape[2]

        # This generates the dense locations where the interpolant
        # will be evaluated.
        grid_locations = _get_grid_locations(image_height, image_width)

        flattened_grid_locations = tf.reshape(grid_locations,
                                              [tf.multiply(image_height, image_width), 2])

        # flattened_grid_locations = constant_op.constant(
        #     _expand_to_minibatch(flattened_grid_locations, batch_size), image.dtype)
        flattened_grid_locations = _expand_to_minibatch(flattened_grid_locations, batch_size)
        flattened_grid_locations = tf.cast(flattened_grid_locations, dtype=image.dtype)

        if clamp_boundaries:
            (dest_control_point_locations,
             control_point_flows) = _add_zero_flow_controls_at_boundary(
                 dest_control_point_locations, control_point_flows, image_height,
                 image_width, boundary_points_per_edge)

        flattened_flows = interpolate_spline(
            dest_control_point_locations, control_point_flows,
            flattened_grid_locations, interpolation_order, regularization_weight)

        dense_flows = array_ops.reshape(flattened_flows,
                                        [batch_size, image_height, image_width, 2])

        warped_image = dense_image_warp(image, dense_flows)

        return warped_image, dense_flows
    

def image_warping(src_img, src_landmarks, dest_landmarks):
    # expanded_src_landmarks = np.expand_dims(np.float32(src_landmarks), axis=0)
    # expanded_dest_landmarks = np.expand_dims(np.float32(dest_landmarks), axis=0)
    # expanded_src_img = np.expand_dims(np.float32(src_img) / 255, axis=0)

    warped_img, dense_flows = sparse_image_warp(src_img,
                          src_landmarks,
                          dest_landmarks,
                          interpolation_order=1,
                          regularization_weight=0.1,
                          num_boundary_points=2,
                          name='sparse_image_warp')

    with tf.Session() as sess:
        out_img = sess.run(warped_img)
        warp_img = np.uint8(out_img[:, :, :, :] * 255)
    
    return warp_img

def face_landmark(img):
    X = np.zeros((img.shape[0], 72 ,2))
    flag = []
    for i in range(img.shape[0]):

        landmark = face_recognition.face_landmarks(img[i].reshape(224,224,3))
        resultant = []
        try:
            for j in list(landmark[0].keys()):
                resultant += landmark[0][j] 
        except IndexError:
            flag.append(i)
            continue
        X[i] = np.array(resultant)
    return  X, flag


import torch 
from torch import nn
class DECODER(nn.Module):
    def __init__(self, phase):
        super(DECODER, self).__init__()
        self.phase = phase
        self.fc3 = nn.Linear(128, 1000)
        self.ReLU = nn.ReLU()
        #self.fc_bn3 = nn.BatchNorm1d(1000)

        self.fc4 = nn.Linear(1000, 14 * 14 * 64)
        self.fc_bn4 = nn.BatchNorm1d(14 * 14 * 64)
        def TransConv( i, kernal = 5, stride = 2, inp = None):
            if not inp:
                inp = max(256//2**(i-1), 32)

            layer =  nn.Sequential(
                nn.ConvTranspose2d(inp, max(256//2**i, 32), 
                                kernal, stride=stride, padding=2, output_padding=1, 
                                dilation=1, padding_mode='zeros'),
                nn.ReLU(),
                nn.BatchNorm2d(max(256//2**i, 32)))
            return layer
        self.T1_ = TransConv(1, inp = 64)
        self.T2_ = TransConv(2)
        self.T3_ = TransConv(3)
        self.T4_ = TransConv(4)
    
        self.ConvLast = nn.Sequential(
            nn.Conv2d(32, 3, (1,1), stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU())

        self.layerLandmark1 = nn.Linear(1000, 800)
        self.layerLandmark2 = nn.Linear(800, 600)
        self.layerLandmark3 = nn.Linear(600, 400)
        self.layerLandmark4 = nn.Linear(400, 200)
        self.layerLandmark5 = nn.Linear(200, 144)

    def forward(self, x):
        L1 = self.fc3(x)
        L1 = self.ReLU(L1)

        L2 = self.layerLandmark1(L1)
        L2 = self.ReLU(L2)

        L3 = self.layerLandmark2(L2)
        L3 = self.ReLU(L3)

        L4 = self.layerLandmark3(L3)
        L4 = self.ReLU(L4)

        L5 = self.layerLandmark4(L4)
        L5 = self.ReLU(L5)

        L6 = self.layerLandmark5(L5)
        outL = self.ReLU(L6)

        # B1 = self.fc_bn3(L1) 
        T0 = self.fc4(L1) 
        T0 = self.ReLU(T0)
        # T0 = self.fc_bn4(T0)
        T0 = T0.view(-1,64,14,14)

        T1 = self.T1_(T0)
        T2 = self.T2_(T1)
        T3 = self.T3_(T2)
        T4 = self.T4_(T3)

        outT = self.ConvLast(T4)
        if self.phase == "train":
            return outL,  outT 
        elif self.phase == "eval":
            img = outT.cpu().detach().numpy().reshape(-1, 224, 224, 3)*255
            outL = outL.cpu().detach().numpy()
            outL = np.dstack((outL[:,0::2],outL[:,1::2]))
            #print("land np img np ", outL_.shape, img_.shape)
            img = (img.reshape(-1,224,224,3)*255).astype(np.uint8)
            #print("img_t ",img_t.shape, img_t[0])
            src, flag = face_landmark(img)
            if flag:
                for r in flag:
                    src[r] = outL[r]

            return image_warping(img.astype(np.float32), src.astype(np.float32), outL.astype(np.float32))
