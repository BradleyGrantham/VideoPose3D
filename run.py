# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

args = parse_args()
print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):  # bgnote - this is where we are
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset(f"{args.keypoints}")
else:
    raise KeyError('Invalid dataset')

print('Loading 2D detections...')
keypoints = np.load(args.keypoints, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
keypoints = keypoints['positions_2d'].item()

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]

poses_valid_2d = keypoints["detectron2"]["custom"]

filter_widths = [int(x) for x in args.architecture.split(',')]

model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], num_joints_out=17,
                            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos.load_state_dict(checkpoint['model_pos'])

# bgnote - pad with 121 on the first axis
# bgnote - we will pass through our keypoints in one batch
batch_2d = np.expand_dims(np.pad(poses_valid_2d[0],
                            ((pad, pad), (0, 0), (0, 0)),
                            'edge'), axis=0)
print('INFO: Testing on {} frames'.format(poses_valid_2d[0].shape[0]))


# Evaluate
def evaluate(batch_2d, return_predictions=False):
    with torch.no_grad():
        model_pos.eval()
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()

        # Positional model
        predicted_3d_pos = model_pos(inputs_2d)

    return predicted_3d_pos.squeeze(0).cpu().numpy()


def rotate_about_z(predictions, theta):
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],
         [0, 0, 1]])

    return np.matmul(predictions, rot_matrix)


def rotate_about_y(predictions, theta):
    rot_matrix = np.array(
        [[np.cos(theta), 0, -np.sin(theta)],
         [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])

    return np.matmul(predictions, rot_matrix)


if args.render:
    print('Rendering...')
    
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None

    # bgnote - we want ground truth to be None as we don't know it
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')
        
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(batch_2d, return_predictions=True)
    
    if args.viz_output is not None:
        
        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]

        # The ground truth is not available, take the camera extrinsic params from a random subject.
        # They are almost the same, and anyway, we only need this for visualization purposes.
        for subject in dataset.cameras():
            if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                break

        # bgnote - we **need** this. I think because of how the model was trained
        prediction = camera_to_world(prediction, R=rot, t=0)

        # bgnote - all we are doing here is setting the minimum height to zero
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        # TODO - uncomment these and add command line arguments
        prediction = rotate_about_z(prediction, np.pi / 2)
        #bgnote - we need to rotate our prediction by 180 degrees in the y axis to use in d3 3d
        prediction = rotate_about_y(prediction, np.pi)

        with open("data.json", "w") as f:
            json.dump(prediction.tolist(), f)
        
        anim_output = {'Reconstruction': prediction}
        
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])  # bgnote - :2 cos it could be 3d
        
        from common.visualization import render_animation
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)