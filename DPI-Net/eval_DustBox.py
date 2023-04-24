import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py

import json

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data import load_data, prepare_input, normalize, denormalize
from models import DPINet
from utils import calc_box_init_FluidShake

parser = argparse.ArgumentParser()
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--env', default='')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--nf_relation', type=int, default=300)
parser.add_argument('--nf_particle', type=int, default=200)
parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)

parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--n_instances', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()

phases_dict = dict()

if args.env == 'DustBox':
    data_names = ['positions', 'velocities']
    args.n_his = 5

    # object states:
    # [x, y, xdot, ydot]
    args.state_dim = 2 + 2 + 2 * args.n_his
    args.position_dim = 2

    args.dt = 1./30.

    # object attr:
    # [dust, air_effector]
    args.attr_dim = 2

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 501
    # используется для генерации
    # args.time_step_clip = 0
    args.n_instance = 2
    
    # ???
    args.n_stages = 1

    args.neighbor_radius = 2.5

    # только в таком порядке
    phases_dict["instance_idx"] = [0, 199, 199+56]
    phases_dict["root_num"] = [[], []]
    phases_dict["instance"] = ['dust', 'air_rigid']
    phases_dict["material"] = ['dust', 'air_rigid']

    args.outf = 'dump_DustBox/' + args.outf
    args.evalf = 'dump_DustBox/' + args.evalf

else:
    raise AssertionError("Unsupported env")

args.outf = args.outf + '_' + args.env
args.evalf = args.evalf + '_' + args.env
args.dataf = 'data/' + args.dataf + '_' + args.env

print(args)

print("Loading stored stat from %s" % args.dataf)
stat_path = os.path.join(args.dataf, 'stat.h5')
stat = load_data(data_names[:2], stat_path)
for i in range(len(stat)):
    stat[i] = stat[i][-args.position_dim:, :]
    # print(data_names[i], stat[i].shape)

info_path = os.path.join(args.dataf, 'info.json')
with open(info_path) as f:
    info_dict = json.load(f)

use_gpu = torch.cuda.is_available()

model = DPINet(args, stat, phases_dict, residual=True, use_gpu=use_gpu)

if args.epoch == 0 and args.iter == 0:
    model_file = os.path.join(args.outf, 'net_best.pth')
else:
    model_file = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

print("Loading network from %s" % model_file)
model.load_state_dict(torch.load(model_file))
model.eval()

criterionMSE = nn.MSELoss()

if use_gpu:
    model.cuda()

infos = np.arange(5)

for idx in range(len(infos)):

    print("Rollout %d / %d" % (idx, len(infos)))
    des_dir = os.path.join(args.evalf, 'rollout_%d' % idx)
    os.system('mkdir -p ' + des_dir)

    # path to jsons with results
    # des_dir

    # tmp_json_data = {'end_frame': args.time_step-3,
    #                 'left_border': info_dict['left_border'],
    #                 'right_border': info_dict['right_border'],
    #                 'bottom_border': info_dict['bottom_border'],
    #                 'top_border': info_dict['top_border'],
    #                 'object_positions': [[]for _ in range(phases_dict["instance_idx"][2]-phases_dict["instance_idx"][1])], 
    #                 'particle_positions': [[]for _ in range(phases_dict["instance_idx"][1]-phases_dict["instance_idx"][0])],
    #                 'particle_velocities': [[]for _ in range(phases_dict["instance_idx"][1]-phases_dict["instance_idx"][0])]}


    # ground truth
    for step in range(args.time_step - 1):
        data_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step) + '.h5')
        data_nxt_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step + 1) + '.h5')

        data = load_data(data_names, data_path)
        data_nxt = load_data(data_names, data_nxt_path)
        # velocities_nxt = data_nxt[1]

        if step == 0:
            if args.env == 'BoxBath':
                positions, velocities, clusters = data
                n_shapes = 0
                scene_params = np.zeros(1)
            elif args.env == 'FluidFall' or args.env == 'DustBox':
                positions, velocities = data
                n_shapes = 0
                scene_params = np.zeros(1)
            elif args.env == 'RiceGrip':
                positions, velocities, shape_quats, clusters, scene_params = data
                n_shapes = shape_quats.shape[0]
            elif args.env == 'FluidShake':
                positions, velocities, shape_quats, scene_params = data
                n_shapes = shape_quats.shape[0]
            else:
                raise AssertionError("Unsupported env")

            count_nodes = positions.shape[0]
            n_particles = count_nodes - n_shapes
            print("n_particles", n_particles)
            print("n_shapes", n_shapes)

            p_gt = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))
            s_gt = np.zeros((args.time_step - 1, n_shapes, args.shape_state_dim))
            v_nxt_gt = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))

            p_pred = np.zeros((args.time_step - 1, n_particles + n_shapes, args.position_dim))

        # p_gt[step] = data[0][:, -args.position_dim:]
        # v_nxt_gt[step] = data_nxt[1][:, -args.position_dim:]

        p_gt[step] = data[0]
        v_nxt_gt[step] = data_nxt[1]

        # print(step, np.sum(np.abs(v_nxt_gt[step, :args.n_particles])))

        if args.env == 'RiceGrip' or args.env == 'FluidShake':
            s_gt[step, :, :3] = positions[n_particles:, :3]
            s_gt[step, :, 3:6] = p_gt[max(0, step-1), n_particles:, :3]
            s_gt[step, :, 6:10] = data[2]
            s_gt[step, :, 10:] = data[2]

        # positions = positions + data[1] * args.dt

        # for i, position in enumerate(data[0][:phases_dict["instance_idx"][1]]):
        #     tmp_json_data['particle_positions'][i].append(position.tolist())
        # for i, position in enumerate(data[0][phases_dict["instance_idx"][1]:phases_dict["instance_idx"][2]]):
        #     tmp_json_data['object_positions'][i].append(position.tolist())
        # for i, velocity in enumerate(data_nxt[1][:phases_dict["instance_idx"][1]]):
        #     tmp_json_data['particle_velocities'][i].append(velocity.tolist())

        # for i, position in enumerate(p_gt[step, :phases_dict["instance_idx"][1]]):
        #     tmp_json_data['particle_positions'][i].append(position.tolist())
        # for i, position in enumerate(p_gt[step, phases_dict["instance_idx"][1]:phases_dict["instance_idx"][2]]):
        #     tmp_json_data['object_positions'][i].append(position.tolist())
        # for i, velocity in enumerate(v_nxt_gt[step, :phases_dict["instance_idx"][1]]):
        #     tmp_json_data['particle_velocities'][i].append(velocity.tolist())
    
    # file = open(des_dir + "/tmp.json", "w")
    # file.write(json.dumps(tmp_json_data))
    # file.close()

    # model rollout phases_dict["instance_idx"] = [0, 199, 199+56]

    json_data = {'end_frame': args.time_step-3,
                 'left_border': info_dict['left_border'],
                 'right_border': info_dict['right_border'],
                 'bottom_border': info_dict['bottom_border'],
                 'top_border': info_dict['top_border'],
                 'object_positions': [[]for _ in range(phases_dict["instance_idx"][2]-phases_dict["instance_idx"][1])], 
                 'particle_positions': [[]for _ in range(phases_dict["instance_idx"][1]-phases_dict["instance_idx"][0])],
                 'particle_velocities': [[]for _ in range(phases_dict["instance_idx"][1]-phases_dict["instance_idx"][0])]}
    gt_json_data = {'end_frame': args.time_step-3,
                 'left_border': info_dict['left_border'],
                 'right_border': info_dict['right_border'],
                 'bottom_border': info_dict['bottom_border'],
                 'top_border': info_dict['top_border'],
                 'object_positions': [[]for _ in range(phases_dict["instance_idx"][2]-phases_dict["instance_idx"][1])], 
                 'particle_positions': [[]for _ in range(phases_dict["instance_idx"][1]-phases_dict["instance_idx"][0])],
                 'particle_velocities': [[]for _ in range(phases_dict["instance_idx"][1]-phases_dict["instance_idx"][0])]}

    data_path = os.path.join(args.dataf, 'valid', str(infos[idx]), '0.h5')
    data = load_data(data_names, data_path)

    for step in range(args.time_step - 1):
        if step % 10 == 0:
            print("Step %d / %d" % (step, args.time_step - 1))

        p_pred[step] = data[0]

        if (args.env == 'RiceGrip' or args.env == 'DustBox') and step == 0:
            data[0] = p_gt[step + 1].copy()
            data[1] = np.concatenate([v_nxt_gt[step]] * (args.n_his + 1), 1)
            continue

        # st_time = time.time()
        attr, state, rels, n_particles, n_shapes, instance_idx = \
                prepare_input(data, stat, args, phases_dict, args.verbose_data)

        Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]

        Rr, Rs = [], []
        for j in range(len(rels[0])):
            Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
            Rr.append(torch.sparse.FloatTensor(
                Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
            Rs.append(torch.sparse.FloatTensor(
                Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

        buf = [attr, state, Rr, Rs, Ra]

        with torch.set_grad_enabled(False):
            if use_gpu:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t].cuda())
                    else:
                        buf[d] = Variable(buf[d].cuda())
            else:
                for d in range(len(buf)):
                    if type(buf[d]) == list:
                        for t in range(len(buf[d])):
                            buf[d][t] = Variable(buf[d][t])
                    else:
                        buf[d] = Variable(buf[d])

            attr, state, Rr, Rs, Ra = buf
            # print('Time prepare input', time.time() - st_time)

            # st_time = time.time()
            # print("attr: ", attr.shape, "state: ", state.shape)
            vels = model(
                attr, state, Rr, Rs, Ra, n_particles,
                node_r_idx, node_s_idx, pstep,
                instance_idx, phases_dict, args.verbose_model)
            # print('Time forward', time.time() - st_time)

            # print(vels)

            if args.debug:
                data_nxt_path = os.path.join(args.dataf, 'valid', str(infos[idx]), str(step + 1) + '.h5')
                data_nxt = normalize(load_data(data_names, data_nxt_path), stat)
                label = None
                if args.env == "DustBox":
                    # что я делаю???
                    label = torch.FloatTensor(data_nxt[1][:phases_dict["instance_idx"][1]])
                else:
                    label = torch.FloatTensor(data_nxt[1][:n_particles])
                
                # print(label)
                loss = np.sqrt(criterionMSE(vels, label).item())
                print(loss)

        vels = denormalize([vels.data.cpu().numpy()], [stat[1]])[0]

        if args.env == 'RiceGrip' or args.env == 'FluidShake':
            vels = np.concatenate([vels, v_nxt_gt[step, n_particles:]], 0)
        
        # print(".........................................................................")
        # print(data[0].shape, data[1].shape)
        
        if args.env == "DustBox":
            prev_positions = data[0].copy()
            if step+1 != p_gt.shape[0]:
                data[0] = p_gt[step + 1].copy()
            data[0][:phases_dict["instance_idx"][1]] = prev_positions[:phases_dict["instance_idx"][1]] + vels * args.dt
        else:
            data[0] = data[0] + vels * args.dt

        # print(data[0].shape, data[1].shape)

        if args.env == 'RiceGrip' or args.env == "DustBox":
            # shifting the history
            # positions, restPositions
            if step+1 != p_gt.shape[0]:
                data[1][:, args.position_dim:] = data[1][:, :-args.position_dim]
        
        # print(data[0].shape, data[1].shape)

        if args.env == "DustBox":
            data[1][:, :args.position_dim] = v_nxt_gt[step].copy()
            # print(data[0].shape, data[1].shape)
            # print(data[1][:phases_dict["instance_idx"][1]].shape, vels.shape)
            # print(data[1][:phases_dict["instance_idx"][1]])
            data[1][:phases_dict["instance_idx"][1], :args.position_dim] = vels
        else:
            data[1][:, :args.position_dim] = vels

        # print(data[0].shape, data[1].shape)
        # print(".........................................................................")

        if args.debug:
            data[0] = p_gt[step + 1].copy()
            data[1][:, :args.position_dim] = v_nxt_gt[step]

        for i, position in enumerate(data[0][:phases_dict["instance_idx"][1]]):
            json_data['particle_positions'][i].append(position.tolist())
        for i, position in enumerate(data[0][phases_dict["instance_idx"][1]:phases_dict["instance_idx"][2]]):
            json_data['object_positions'][i].append(position.tolist())
        for i, velocity in enumerate(data[1][:phases_dict["instance_idx"][1]]):
            json_data['particle_velocities'][i].append(velocity.tolist())
        
        # gt_step = None
        # if step == args.time_step-2:
        #     gt_step = step-1
        # else:
        #     gt_step = step
        for i, position in enumerate(p_gt[step, :phases_dict["instance_idx"][1]]):
            gt_json_data['particle_positions'][i].append(position.tolist())
        for i, position in enumerate(p_gt[step, phases_dict["instance_idx"][1]:phases_dict["instance_idx"][2]]):
            gt_json_data['object_positions'][i].append(position.tolist())
        for i, velocity in enumerate(v_nxt_gt[step, :phases_dict["instance_idx"][1]]):
            gt_json_data['particle_velocities'][i].append(velocity.tolist())
            
    file = open(des_dir + "/result.json", "w")
    file.write(json.dumps(json_data))
    file.close()

    file = open(des_dir + "/gt.json", "w")
    file.write(json.dumps(gt_json_data))
    file.close()