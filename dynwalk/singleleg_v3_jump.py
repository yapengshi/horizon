"""Provides an example translation of `cart_pole_passive_simluation.cc`."""

from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import (FiniteHorizonLinearQuadraticRegulator,
                         FiniteHorizonLinearQuadraticRegulatorOptions,
                         LinearSystem)
from pydrake.all import (LinearQuadraticRegulator, JacobianWrtVariable,
                         MathematicalProgram, eq, le, ge, SnoptSolver)
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer, MeshcatContactVisualizer
from pydrake.all import (InverseKinematics, Solve, RotationMatrix, RollPitchYaw, Quaternion, RigidTransform, HalfSpace, CoulombFriction,
                         LeafSystem, BasicVector, PackageMap, DrakeVisualizer)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.lcm import DrakeLcm
from pydrake.geometry import (DrakeVisualizer, SceneGraph)
from pydrake.common import FindResourceOrThrow
from common.IK import *
from common.utils import *
import sys
import os
from copy import deepcopy
from curses import panel
import math
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
import pdb
sys.path.append(os.path.abspath(os.path.dirname(__file__) + r'../../../'))


np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000)


class JointLimit(NamedTuple):
    effort: float
    lower: float
    upper: float
    velocity: float


JOINT_LIMITS = {
    "joint_hip": JointLimit(100, -3.14, 3.14, 50),
    "joint_knee": JointLimit(100, -3.14, 3.14, 50),
    "joint_ankle": JointLimit(100, -3.14, 3.14, 50)
}

CONTACTS_PER_FRAME = {
    "foot": np.array([
        [0.07, 0.03, -0.028],  # foot_toe_l
        [0.07, -0.03, -0.028],  # foot_toe_r
        [-0.07, 0.03, -0.028],  # foot_heel_l
        [-0.07, -0.03, -0.028],  # foot_heel_r
    ]).T}


__model_pack_file = "robots/singleleg_v3"
__model_file = __model_pack_file + "/urdf/singleleg_v3_symmetrical.urdf"
__end_frames_name = ["base_link", "foot_sole"]
__initial_joint_name = ["joint_knee"]
__initial_joint_pos = [0.4]
__contact_frame_name = ["foot_toe", "foot_heel"]
__robot_instance = "singleleg_v3"

__com_squat = [0, 0, -0.06]

total_time = 0.5
com_z = 0.40
StepWith = 0.2
torsoP = np.deg2rad(15)
StepDuration = 0.3
zmp_state_size = 2
mbp_time_step = 2.0e-3

mu = 1.0  # Coefficient of friction
eta_min = -0.2
eta_max = 0.2

N_d = 4  # friction cone approximated as a i-pyramid
N_f = 3  # contact force dimension

g = 9.81

state_traj_des = None
com_traj_des = None

plot_cost = []
plot_x = []
plot_x_des = []
plot_u = []
plot_u_des = []
plot_tau = []
plot_q = []
plot_q_des = []
plot_v = []
plot_v_des = []
plot_lambda = []
plot_CM = []
plot_contact_SpaMom = []
plot_SpM_des = []
plot_theta = []
plot_torsoRot = []


def load_traj():
    global state_traj_des
    global com_traj_des
    file_name = 'data/biped_com.csv'
    com_traj_des = np.genfromtxt(file_name, delimiter=',', skip_header=True)
    file_name = 'data/biped_state.csv'
    state_traj_des = np.genfromtxt(file_name, delimiter=',', skip_header=True)
    # state_traj_des = np.tile(state_traj_des, (2, 1))
    # return state_traj_des.shape[0]*mbp_time_step
    return total_time


def initial_state(plant, plant_context, end_frames_name, initial_joint_name, initial_joint_pos):
    foot_in_base = plant.GetFrameByName(end_frames_name[1]).CalcPose(plant_context, plant.GetFrameByName(end_frames_name[0]))
    q0 = plant.GetPositions(plant_context)
    q0[6] = -foot_in_base.translation()[2]
    plant.SetPositions(plant_context, q0)
    print('init pos = ', q0)
    r0 = plant.CalcCenterOfMassPositionInWorld(plant_context)
    for i, name in enumerate(initial_joint_name):
        joint = plant.GetJointByName(name=name)
        joint.set_angle(plant_context, initial_joint_pos[i])
    q0 = plant.GetPositions(plant_context)

    IK = CoMIK(plant, end_frames_name)
    pose_list = [
        [[0.0, torsoP, 0.0], [0., 0., com_z]],
        [[0.0, 0.0, 0.0], [0.0, foot_in_base.translation()[1], 0.0]],
        [[0.0, 0.0, 0.0], [0.0, -foot_in_base.translation()[1], 0.0]]
    ]
    is_success, q = IK.solve(pose_list, q0=q0)
    if not is_success:
        print(f"pose: {pose_list[0][0].T}, {pose_list[0][1].T}")
        print(f"lfoot: {pose_list[1][0].T}, {pose_list[1][1].T}")
        print(f"rfoot: {pose_list[2][0].T}, {pose_list[2][1].T}")
        raise RuntimeError("Failed to IK!")
    plant.SetPositions(plant_context, q)


class PidController(LeafSystem):
    def __init__(self, plant, kp, ki, kd, nq_f=7, nv_f=6):
        LeafSystem.__init__(self)
        self.plant = plant
        self.nq, self.nv = self.plant.num_positions(), self.plant.num_velocities()
        self.na = self.plant.num_actuated_dofs()
        self.nq_f, self.nv_f = nq_f, nv_f
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.intergral = np.zeros(self.na)
        self.DeclareVectorInputPort("state", BasicVector(self.nq + self.nv))
        self.DeclareVectorInputPort("desire", BasicVector(self.nq + self.nv))
        self.DeclareVectorOutputPort("tau", BasicVector(self.plant.num_actuated_dofs()), self.CalcTau)

    def CalcTau(self, context, output):
        state = self.get_input_port(0).Eval(context)
        desire = self.get_input_port(1).Eval(context)
        q_err = (desire[self.nq_f:self.nq] - state[self.nq_f:self.nq])
        v_err = (desire[self.nq + self.nv_f:] - state[self.nq + self.nv_f:])
        self.intergral = self.intergral + q_err
        tau = self.kp * q_err + self.ki * self.intergral + self.kd * v_err
        output.SetFromVector(tau)


class TrajPlanner(LeafSystem):
    def __init__(self, plant, end_frames_name, robot_instance,
                 nq_f=7, nv_f=6):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = self.plant.CreateDefaultContext()
        self.nq = self.plant.num_positions()
        self.nv = self.plant.num_velocities()
        self.na = self.plant.num_actuated_dofs()
        self.dt = plant.time_step()
        self.nq_f, self.nv_f = nq_f, nv_f
        self.end_frames_name = end_frames_name
        self.DeclareVectorInputPort("state", BasicVector(self.nq + self.nv))
        self.DeclareVectorOutputPort("traj_des", BasicVector(3 * 3 + self.nq + self.nv + self.nv + 3 + 1), self.Output)

        self.total_mass = sum(self.plant.get_body(index).get_mass(self.plant_context)
                              for index in self.plant.GetBodyIndices(self.plant.GetModelInstanceByName(robot_instance)))
        self.traj_index = 0
        self.step_index = 0

        self.v_des_pre = np.array([0.0] * self.nv)
        self.vd_des_pre = np.array([0.0] * self.nv)
        self.comH_pre = com_z

        self.r_pre = np.array([0., 0., com_z])
        self.r_ik_pre = np.array([0., 0., com_z])
        self.rd_pre = np.array([0., 0., 0.])
        self.rdd_pre = np.array([0., 0., 0.])

        self.r_rot_pre = np.array([0., 0., 0])
        self.r_rot_ik_pre = np.array([0., 0., 0])
        self.rd_rot_pre = np.array([0., 0., 0.])
        self.rdd_rot_pre = np.array([0., 0., 0.])

        self.contact = np.array([0., 0., 0])
        self.rd_takeoff = np.array([0., 0., 0])

        self.lf_pre = np.array([0., StepWith / 2., 0])
        self.rf_pre = np.array([0., -StepWith / 2., 0])
        self.lf_ik_pre = np.array([0., StepWith / 2., 0])
        self.rf_ik_pre = np.array([0., -StepWith / 2., 0])
        self.lf_sole = np.array([0., StepWith / 2., 0])
        self.rf_sole = np.array([0., -StepWith / 2., 0])

        self.phase = 'takeoff'

        self.l0 = 0.48  # m
        # self.acc = np.array([1.5, 0, 10]) # m/s^2
        # self.theta_des = np.deg2rad(9.) #deg
        # self.rd_takeoff_des = 0.2 # m/s

        self.acc = np.array([0., 0, 6])  # m/s^2
        self.theta_des = np.deg2rad(0.)  # deg
        self.rd_takeoff_des = np.array([0., 0., 0.5])  # m/s

        self.acczflag = False
        self.torsoflag = False
        self.l_flag = False

        self.IK = velIK(plant, self.end_frames_name, 1e-6, 1e-6)

    def Initial(self, qv):
        self.plant.SetPositionsAndVelocities(self.plant_context, qv)

        self.q_des_pre = qv[0:self.nq]

    def Output(self, context, output):
        qv = self.get_input_port(0).Eval(context)
        if not np.array_equal(qv, self.plant.GetPositionsAndVelocities(self.plant_context)):
            self.plant.SetPositionsAndVelocities(self.plant_context, qv)

        plant = self.plant
        plant_context = self.plant_context
        dt = self.dt

        r_est = plant.CalcCenterOfMassPositionInWorld(plant_context)
        rd_est = plant.CalcJacobianCenterOfMassTranslationalVelocity(plant_context, JacobianWrtVariable.kV,
                                                                     plant.world_frame(), plant.world_frame()).dot(qv[self.nq:])
        lf_est = plant.GetFrameByName(self.end_frames_name[1]).CalcPose(plant_context, plant.world_frame()).translation()
        lf_rot_est = plant.GetFrameByName(self.end_frames_name[1]).CalcPose(plant_context,
                                                                            plant.world_frame()).rotation().ToQuaternion().xyz()
        l_est = np.linalg.norm(r_est - lf_est, ord=None, axis=None)
        CM_est = plant.CalcSpatialMomentumInWorldAboutPoint(plant_context, r_est)
        CM_est = np.concatenate([CM_est.rotational(), CM_est.translational()])
        SpM_est = plant.CalcSpatialMomentumInWorldAboutPoint(plant_context, lf_est)
        SpM_est = np.concatenate([SpM_est.rotational(), SpM_est.translational()])
        theta_est = math.atan((r_est[0] - lf_est[0]) / (r_est[2] - lf_est[2]))
        qrot = qv[:4] / np.linalg.norm(qv[:4], ord=None, axis=None)
        torsoRot_est = RollPitchYaw(Quaternion(qrot)).vector()

        # "SLIPState", ["x", "z", "l", "theta", "xdot", "zdot", "ldot", "thetadot"])
        if self.phase == 'takeoff':
            contact_des = np.array([2])
            rdd = self.acc
            rd = self.rd_pre + rdd * dt
            # TODO: add self.r_ik_pre feedback
            # rd[0] = rd[0] - 0.01*rd_est[0] - 0.5 * \
            #     (self.r_pre[0]-self.contact[0])
            r = self.r_pre + rd * dt

            CM_torso_fd = np.array([0., -1. * (torsoRot_est[1] - torsoP), 0])
            if CM_torso_fd[1] > 0.3:
                CM_torso_fd[1] = 0.3
            CM_rot = np.array([0., 0, 0]) + CM_torso_fd

            l = r - self.contact

            # TODO: add SpM, theta feedback
            footvel = np.array([0, 0, 0])
            footRotvel = np.array([0, 0, 0])

            self.traj_index = self.traj_index + 1
            # TODO: add assume no ik failure before rd_est[2] > self.rd_takeoff_des[2]
            if l_est > self.l0 and rd_est[2] > self.rd_takeoff_des[2]:
                self.traj_index = 0
                self.phase = 'flight'
                self.r_pre = r_est
                self.rd_pre = rd_est
                self.rd_takeoff = rd_est
                self.theta_des = self.theta_des - 0.2 * (self.rd_takeoff_des[0] - self.rd_takeoff[0])

        elif self.phase == 'flight':
            contact_des = np.array([3])
            rdd = np.array([0, 0, -g])
            rd = rd_est + rdd * dt
            r = r_est + rd * dt
            CM_rot = CM_est[:3]

            # TODO: add theta integration; vel feedback
            k_l = 5.
            k_theta = 10.
            # TODO: add l for nonzero theta; add theta plan
            l = np.array([0, 0, self.l0])
            vel_l = -k_l * (l - (r_est - lf_est))

            # assume theta_des_landing = -theta_des_takeoff
            w = np.array([0, k_theta * (-self.theta_des - theta_est), 0])
            vel_theta = np.cross(l, w)

            # TODO: add lf/rf, l, theta, sole pitch feedback
            footvel = rd + vel_l + vel_theta
            footRotvel = np.array([0, 0, 0]) - 0.5 * lf_rot_est

            self.traj_index = self.traj_index + 1
            if rd[2] < 0 and lf_est[2] < 1e-2:
                self.traj_index = 0
                self.phase = 'touchdown'
                self.contact = lf_est

        elif self.phase == 'touchdown':
            contact_des = np.array([4])
            accX, accZ = 0, 0
            if(rd_est[0] > 0):
                accX = -(self.acc[0] - 10 * (self.rd_takeoff_des[0] - self.rd_takeoff[0]))
            if(rd_est[2] < 0):
                accZ = self.acc[2]

            if(rd_est[0] <= 0):
                self.rd_pre[0] = 0
            if(rd_est[2] >= 0):
                self.rd_pre[2] = 0

            rdd = np.array([accX, 0, accZ])

            rd = self.rd_pre + rdd * dt
            # TODO: add self.r_ik_pre feedback
            rd[0] = rd[0] - 0.1 * (self.r_pre[0] - self.contact[0])
            r = self.r_pre + rd * dt

            CM_torso_fd = np.array([0., -2. * (torsoRot_est[1] - torsoP), 0])
            if CM_torso_fd[1] > 0.3:
                CM_torso_fd[1] = 0.3
            CM_rot = np.array([0., 0, 0]) + CM_torso_fd

            l = r - self.contact

            footvel = np.array([0, 0, 0])
            footRotvel = np.array([0, 0, 0])

            self.traj_index = self.traj_index + 1
            if abs(rd_est[2]) < 0.01:
                self.acczflag = True
            if abs(torsoRot_est[1] - torsoP) < 0.01:
                self.torsoflag = True
            if abs(self.r_pre[0] - self.contact[0]) < 0.01:
                self.l_flag = True

            # if self.acczflag and self.torsoflag and self.l_flag:
            if self.acczflag and self.l_flag:
                self.acczflag = False
                self.torsoflag = False
                self.l_flag = False
                self.traj_index = 0
                self.phase = 'takeoff'
        else:
            print('phase error !! ')

        # test SpM calc
        # calcSpM = np.cross(r_est - (lf_est+rf_est)/2, rd_est*self.total_mass) + CM_est[:3]
        # print(SpM_est[:3], '|', calcSpM)
        angularSpM_des = np.cross(l, rd * self.total_mass) + CM_rot

        self.r_pre = r
        self.rd_pre = rd
        self.rdd_pre = rdd
        lfvel = footvel  # TODO: add lf_ik_pre feedback
        rfvel = footvel
        lfrotvel = footRotvel
        rfrotvel = footRotvel

        # print(self.traj_index, '|', CM_rot, rd, lfvel, lfvel, self.lf_ik_pre, self.rf_ik_pre)
        # print(self.q_des_pre)
        # print(self.v_des_pre)
        # print()

        cm_v = rd * self.total_mass
        pose_list = [
            [[0, CM_rot[1], 0], [cm_v[0], 0, cm_v[2]]],
            [[0, lfrotvel[1], 0], [lfvel[0], 0, lfvel[2]]],
        ]
        start_formulate_time = time.time()
        is_success, q, v = self.IK.solve(pose_list, mbp_time_step, q0=self.q_des_pre, v0=self.v_des_pre)
        end_formulate_time = time.time()

        if not is_success:
            print('\nFailed to IK0!\n')
            exit()

        plant.SetPositions(plant_context, q)
        self.lf_ik_pre = plant.GetFrameByName(self.end_frames_name[1]).CalcPose(plant_context, plant.world_frame()).translation()
        self.r_ik_pre = plant.CalcCenterOfMassPositionInWorld(plant_context)
        self.r_rot_ik_pre = plant.GetFrameByName(self.end_frames_name[0]).CalcPose(plant_context,
                                                                                   plant.world_frame()).rotation().ToQuaternion().xyz()

        q_des = q
        v_des = v
        vd_des = (v_des - self.v_des_pre) / mbp_time_step
        for i in range(vd_des.shape[0]):
            if vd_des[i] > 300:
                vd_des[i] = 300
            if vd_des[i] < -300:
                vd_des[i] = -300

        self.q_des_pre = q_des
        self.v_des_pre = v_des
        self.vd_des_pre = vd_des

        # print(q_des.T)
        # print(v_des.T)
        # print(vd_des.T)
        # print()
        # print('Context: {:.3f}, slvoe: {:.4f}'.format(context.get_time(),
        #                                                     end_formulate_time - start_formulate_time,
        #                                                     ), end='\n')

        traj = np.concatenate([r, rd, rdd, q_des, v_des, vd_des, angularSpM_des, contact_des])
        output.SetFromVector(traj)

        plot_theta.append(theta_est)
        plot_torsoRot.append(torsoRot_est)


class WholeBodyController(LeafSystem):
    def __init__(self, plant, end_frames_name, nq_f=7, nv_f=6):
        LeafSystem.__init__(self)
        self.plant = plant
        self.context = self.plant.CreateDefaultContext()
        self.nq = self.plant.num_positions()
        self.nv = self.plant.num_velocities()
        self.na = self.plant.num_actuated_dofs()
        self.dt = plant.time_step()
        self.nq_f, self.nv_f = nq_f, nv_f
        self.end_frames_name = end_frames_name
        self.nq_com, self.nv_com = 3, 3
        self.DeclareVectorInputPort("state", BasicVector(self.nq + self.nv))
        self.DeclareVectorInputPort("desire", BasicVector(3 * 3 + self.nq + self.nv + self.nv + 3 + 1))
        self.DeclareVectorOutputPort("tau", BasicVector(self.plant.num_actuated_dofs()), self.CalcTau)

        self.__traj_index = 0
        self.__state_traj = state_traj_des
        self.__com_traj = com_traj_des
        self.contacts_per_frame = CONTACTS_PER_FRAME
        self.joint_limits = JOINT_LIMITS
        self.is_wbc = True
        if self.is_wbc:
            com_dim = 3
            self.x_size = com_dim * 2
            self.u_size = com_dim
            '''
            self.input_r_des_idx = self.DeclareVectorInputPort("r_des", BasicVector(com_dim)).get_index()
            self.input_rd_des_idx = self.DeclareVectorInputPort("rd_des", BasicVector(com_dim)).get_index()
            self.input_rdd_des_idx = self.DeclareVectorInputPort("rdd_des", BasicVector(com_dim)).get_index()
            self.input_q_des_idx = self.DeclareVectorInputPort("q_des", BasicVector(self.nq)).get_index()
            self.input_v_des_idx = self.DeclareVectorInputPort("v_des", BasicVector(self.nv)).get_index()
            self.input_vd_des_idx = self.DeclareVectorInputPort("vd_des", BasicVector(self.nv)).get_index()
            '''
            Q = 1.0e3 * np.identity(self.x_size)
            R = 0.1 * np.identity(self.u_size)

            def LQR(Q, R):
                A = np.vstack([np.hstack([0 * np.identity(com_dim), 1 * np.identity(com_dim)]),
                               np.hstack([0 * np.identity(com_dim), 0 * np.identity(com_dim)])])
                B_1 = np.vstack([0 * np.identity(com_dim),
                                1 * np.identity(com_dim)])
                K, S = LinearQuadraticRegulator(A, B_1, Q, R)

                def V_full(x, u, r, rd, rdd):
                    x_bar = x - np.concatenate([r, rd])
                    u_bar = u - rdd
                    '''
                    xd_bar = d(x - [r, rd].T)/dt
                            = xd - [rd, rdd].T
                            = Ax + Bu - [rd, rdd].T
                    '''
                    xd_bar = A.dot(x) + B_1.dot(u) - np.concatenate([rd, rdd])
                    return x_bar.T.dot(Q).dot(x_bar) + u_bar.T.dot(R).dot(u_bar) + 2 * x_bar.T.dot(S).dot(xd_bar)
                self.V_full = V_full

            def tvLQR(com_traj, Q, R):
                A = np.vstack([np.hstack([np.zeros((3, 3)), np.identity(3)]), np.zeros((3, 6))])
                B = np.vstack([np.identity(3), np.zeros((3, 3))])
                C = np.identity(6)
                D = np.zeros((6, 3))
                particle_sys = LinearSystem(A, B, C, D)
                particle_context = particle_sys.CreateDefaultContext()

                x_dim = self.nq_com + self.nv_com
                u_dim = self.nv_com
                breaks = np.linspace(0, com_traj.shape[0] * mbp_time_step, com_traj.shape[0])
                samples = np.zeros((x_dim, com_traj.shape[0]))
                for i in range(x_dim):
                    samples[i, :] = com_traj[:, i]
                x_com_pp = PiecewisePolynomial.CubicShapePreserving(breaks, samples, zero_end_point_derivatives=True)

                samples = np.zeros((u_dim, com_traj.shape[0]))
                for i in range(u_dim):
                    samples[i, :] = com_traj[:, x_dim + i]
                u_com_pp = PiecewisePolynomial.CubicShapePreserving(breaks, samples, zero_end_point_derivatives=True)

                '''
                y = [[] for i in range(x_dim)]
                label = []
                x = np.linspace(0, com_traj.shape[0]*mbp_time_step, com_traj.shape[0])
                for i in range(com_traj.shape[0]):
                    for j in range(x_dim):
                        y[j].append(x_com_pp.value(i*mbp_time_step)[j])
                for i in range(x_dim):
                    label.append('x{}'.format(i))
                plt.subplot(1,2,1)
                plot_scatter(x, y, label)

                y = [[] for i in range(u_dim)]
                label = []
                for i in range(com_traj.shape[0]):
                    for j in range(u_dim):
                        y[j].append(u_com_pp.value(i*mbp_time_step)[j])
                for i in range(u_dim):
                    label.append('u{}'.format(i))
                plt.subplot(1,2,2)
                plot_scatter(x, y, label)
                plt.show()
                '''

                options = FiniteHorizonLinearQuadraticRegulatorOptions()
                options.Qf = Q
                options.x0 = x_com_pp
                options.u0 = u_com_pp
                self.result = FiniteHorizonLinearQuadraticRegulator(system=particle_sys, context=particle_context,
                                                                    t0=options.x0.start_time(),
                                                                    tf=options.x0.end_time(),
                                                                    Q=Q,
                                                                    R=R,
                                                                    options=options)

                def V_full(x, u, r, rd, rdd):
                    x_bar = (x - np.concatenate([r, rd])).reshape(1, -1).T
                    xd_bar = (A.dot(x) + B.dot(u) -
                              np.concatenate([rd, rdd])).reshape(1, -1).T
                    S = self.result.S.value(self.__traj_index * mbp_time_step)
                    sx = self.result.sx.value(self.__traj_index * mbp_time_step)
                    '''
                    print('x_bar:{}\n'.format(x_bar))
                    print('x_bar_T:{}\n'.format(x_bar.T))
                    print('xd_bar:{}\n'.format(xd_bar))
                    print('S:{}\n'.format(S))
                    print('sx:{}\n'.format(sx))
                    print('1:{}\n'.format(x_bar.T.dot(Q).dot(x_bar)))
                    print('2:{}\n'.format((x_bar.T.dot(S.T+S)+sx.T).dot(xd_bar)))
                    print('2:{}\n'.format(x_bar.T.dot(Q).dot(x_bar) + (x_bar.T.dot(S.T+S)+sx.T).dot(xd_bar)))
                    '''
                    return (x_bar.T.dot(Q).dot(x_bar) + (x_bar.T.dot(S.T + S) + sx.T).dot(xd_bar))[0][0]
                self.V_full = V_full

            LQR(Q, R)
            # tvLQR(self.__com_traj, Q, R)
        else:
            # Only x, y coordinates of COM is considered
            com_dim = 2
            self.x_size = 2 * com_dim
            self.u_size = com_dim
            '''
            self.input_y_des_idx = self.DeclareVectorInputPort("y_des", BasicVector(zmp_state_size)).get_index()
            '''
            ''' Eq(1) '''
            A = np.vstack([np.hstack([0 * np.identity(com_dim), 1 * np.identity(com_dim)]),
                           np.hstack([0 * np.identity(com_dim), 0 * np.identity(com_dim)])])
            B_1 = np.vstack([0 * np.identity(com_dim),
                             1 * np.identity(com_dim)])

            ''' Eq(4) '''
            C_2 = np.hstack([np.identity(2), np.zeros((2, 2))])  # C in Eq(2)
            D = -com_z / g * np.identity(zmp_state_size)
            Q = 1.0 * np.identity(zmp_state_size)

            ''' Eq(6) '''
            '''
            y.T*Q*y
            = (C*x+D*u)*Q*(C*x+D*u).T
            = x.T*C.T*Q*C*x + u.T*D.T*Q*D*u + x.T*C.T*Q*D*u + u.T*D.T*Q*C*X
            = ..                            + 2*x.T*C.T*Q*D*u
            '''
            K, S = LinearQuadraticRegulator(A, B_1, C_2.T.dot(Q).dot(C_2), D.T.dot(Q).dot(D), C_2.T.dot(Q).dot(D))
            # Use original formulation

            def V_full(x, u, y_des):  # Assume constant com_z, we don't need tvLQR
                y = C_2.dot(x) + D.dot(u)

                def dJ_dx(x):
                    # https://math.stackexchange.com/questions/20694/vector-derivative-w-r-t-its-transpose-fracdaxdxt
                    return x.T.dot(S.T + S)
                y_bar = y - y_des
                # FIXME: This doesn't seem right...
                x_bar = x - np.concatenate([y_des, [0.0, 0.0]])
                # FIXME: xd_bar should depend on yd_des
                xd_bar = A.dot(x_bar) + B_1.dot(u)
                return y_bar.T.dot(Q).dot(y_bar) + dJ_dx(x_bar).dot(xd_bar)
            self.V_full = V_full

        self.w_V = 0.
        self.w_qdd = 1.0e0
        self.epsilon = 1.0e-8
        self.K_p = [500] * self.nv
        self.K_d = [45] * self.nv
        # Calculate values that don't depend on context
        self.B_7 = self.plant.MakeActuationMatrix()
        # From np.sort(np.nonzero(B_7)[0]) we know that indices 0-5 are the unactuated 6 DOF floating base and 6-35 are the actuated 30 DOF robot joints
        self.v_idx_act = 6  # Start index of actuated joints in generalized velocities
        self.B_a = self.B_7[self.v_idx_act:, :]
        self.B_a_inv = np.linalg.inv(self.B_a)

        # Sort joint effort limits to be the same order as tau in Eq(13)
        # self.sorted_max_efforts = np.array(
        #     [entry[1].effort for entry in self.getJointLimitsSortedByActuator(self.plant, self.joint_limits)])
        self.sorted_max_efforts = self.plant.GetEffortUpperLimits()

    def Initial(self, qv):
        self.plant.SetPositionsAndVelocities(self.context, qv)

    def getActuatorIndex(self, plant, joint_name):
        return int(plant.GetJointActuatorByName(joint_name + "_motor").index())

    def getJointLimitsSortedByActuator(self, plant, joint_limits):
        return sorted(joint_limits.items(), key=lambda entry: self.getActuatorIndex(plant, entry[0]))

    def getJointIndexInGeneralizedPositions(self, plant, joint_name):
        return int(plant.GetJointByName(joint_name).position_start())

    def getJointIndexInGeneralizedVelocities(self, plant, joint_name):
        return self.getJointIndexInGeneralizedPositions(plant, joint_name) - 1

    def create_qp1(self, plant_context, V, q_des, v_des, vd_des, contact_des):
        # Determine contact points
        active_contacts_per_frame = {}  # Note this should be in frame space
        # if(contact_des == 0):
        #     active_contacts_per_frame['left_foot'] = self.contacts_per_frame['left_foot']
        #     active_contacts_per_frame['right_foot'] = self.contacts_per_frame['right_foot']
        # elif(contact_des == 1):
        #     active_contacts_per_frame['left_foot'] = self.contacts_per_frame['left_foot']
        # elif(contact_des == -1):
        #     active_contacts_per_frame['right_foot'] = self.contacts_per_frame['right_foot']
        if(contact_des == 2 or contact_des == 3 or contact_des == 4):
            active_contacts_per_frame = self.contacts_per_frame

        else:
            print('contact_des error: ', contact_des)

        N_c = sum([active_contacts.shape[1]
                  for active_contacts in active_contacts_per_frame.values()])  # num contact points
        if N_c == 0:
            print("Not in contact!")
            return None

        ''' Eq(7) '''
        '''An Efficiently Solvable Quadratic Program for Stabilizing Dynamic Locomotion'''
        H = self.plant.CalcMassMatrixViaInverseDynamics(plant_context)
        # Note that CalcGravityGeneralizedForces assumes the form Mv̇ + C(q, v)v = tau_g(q) + tau_app
        # while Eq(7) assumes gravity is accounted in C (on the left hand side)
        C_7 = self.plant.CalcBiasTerm(plant_context) - self.plant.CalcGravityGeneralizedForces(plant_context)
        B_7 = self.B_7

        # TODO: Double check
        Phi_foots = []
        for frame, active_contacts in active_contacts_per_frame.items():
            if active_contacts.size:
                Phi_foots.append(self.plant.CalcJacobianTranslationalVelocity(plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName(frame),
                                                                              active_contacts, self.plant.world_frame(), self.plant.world_frame()))
        Phi = np.vstack(Phi_foots)

        ''' Eq(8) '''
        v_idx_act = self.v_idx_act
        H_f = H[0:v_idx_act, :]
        H_a = H[v_idx_act:, :]
        C_f = C_7[0:v_idx_act]
        C_a = C_7[v_idx_act:]
        B_a = self.B_a
        Phi_f_T = Phi.T[0:v_idx_act:, :]
        Phi_a_T = Phi.T[v_idx_act:, :]

        ''' Eq(9) '''
        # Assume flat ground for now
        n = np.array([[0],
                      [0],
                      [1.0]])
        d = np.array([[1.0, -1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, -1.0],
                      [0.0, 0.0, 0.0, 0.0]])
        v = np.zeros((N_d, N_c, N_f))
        for i in range(N_d):
            for j in range(N_c):
                v[i, j] = (n + mu * d)[:, i]

        def tau(qdd, lambd):
            return self.B_a_inv.dot(H_a.dot(qdd) + C_a - Phi_a_T.dot(lambd))
        self.tau = tau

        ''' Quadratic Program I '''
        prog = MathematicalProgram()
        # To ignore 6 DOF floating base
        qdd = prog.NewContinuousVariables(self.nv, name="qdd")
        self.qdd = qdd
        beta = prog.NewContinuousVariables(N_d, N_c, name="beta")
        self.beta = beta
        lambd = prog.NewContinuousVariables(N_f * N_c, name="lambda")
        self.lambd = lambd

        # Jacobians ignoring the 6DOF floating base
        J_foots = []
        for frame, active_contacts in active_contacts_per_frame.items():
            if active_contacts.size:
                num_active_contacts = active_contacts.shape[1]
                J_foot = np.zeros((N_f * num_active_contacts, self.nv))
                # TODO: Can this be simplified?
                for i in range(num_active_contacts):
                    J_foot[N_f * i:N_f * (i + 1), :] = self.plant.CalcJacobianSpatialVelocity(plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName(frame),
                                                                                              active_contacts[:, i], self.plant.world_frame(), self.plant.world_frame())[3:]
                J_foots.append(J_foot)
        J = np.vstack(J_foots)
        assert(J.shape == (N_c * N_f, self.nv))

        eta = prog.NewContinuousVariables(J.shape[0], name="eta")
        self.eta = eta

        q = self.plant.GetPositions(plant_context)
        qd = self.plant.GetVelocities(plant_context)

        com = self.plant.CalcCenterOfMassPositionInWorld(plant_context)
        comv = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(plant_context, JacobianWrtVariable.kV,
                                                                        self.plant.world_frame(), self.plant.world_frame()).dot(qd)
        CM = self.plant.CalcSpatialMomentumInWorldAboutPoint(plant_context, com)
        self.CMvec = np.concatenate([CM.rotational(), CM.translational()])
        lf_est = self.plant.GetFrameByName(self.end_frames_name[1]).CalcPose(plant_context, self.plant.world_frame()).translation()
        contact_SpaMom = self.plant.CalcSpatialMomentumInWorldAboutPoint(plant_context, lf_est)
        # self.contact_SpaMomvec = np.concatenate([contact_SpaMom.rotational(), contact_SpaMom.translational()])
        self.contact_SpaMomvec = contact_SpaMom.rotational()
        qrot = q[:4] / np.linalg.norm(q[:4], ord=None, axis=None)
        self.torsoRotvec = RollPitchYaw(Quaternion(qrot)).vector()

        # x = np.array([com[0], com[1], comv[0], comv[1]])
        x = np.concatenate((com, comv))
        self.x = x
        u = prog.NewContinuousVariables(self.u_size, name="u")  # x_com_dd, y_com_dd
        self.u = u

        ''' Eq(10) '''
        # Convert q, q_nom to generalized velocities form
        q_err = self.plant.MapQDotToVelocity(plant_context, q_des - q)
        # print(f"Pelvis error: {q_err[0:3]}")
        # FIXME: Not sure if it's a good idea to ignore the x, y, z position of pelvis
        frame_weights = np.ones((self.nv))
        # ignored_pose_indices = {3, 4, 5} # Ignore x position, y position
        ignored_pose_indices = {}  # Ignore x position, y position
        relevant_pose_indices = list(set(range(self.nv)) - set(ignored_pose_indices))
        qdd_ref = self.K_p * q_err + self.K_d * \
            (v_des - qd) + vd_des  # Eq(27) of [1]
        # qdd_ref[3] = 0 #cancel torso x feedback
        # qdd_ref[4] = 0 #cancel torso y feedback

        # qdd_err = qdd - qdd_ref
        # qdd_err = qdd_err*frame_weights
        # qdd_err = qdd_err[relevant_pose_indices]

        self.cost_v = prog.AddCost((V(self.x, u)) * self.w_V)

        Q = np.identity((self.nv)) * self.w_qdd
        x_desired = qdd_ref
        vras = np.array(qdd)
        self.cost_qdd = prog.AddQuadraticErrorCost(Q, x_desired, vras)

        Q = np.identity((N_d * N_c)) * self.epsilon * 2.0
        b = np.zeros((N_d * N_c, 1))
        vras = np.array(beta.flatten())
        self.cost_epsilon = prog.AddQuadraticCost(Q, b, vras)

        Q = np.identity(eta.shape[0]) * 2.0
        b = np.zeros((eta.shape[0]))
        vras = np.array(eta)
        self.cost_eta = prog.AddQuadraticCost(Q, b, vras)

        ''' Eq(11) '''
        Aeq = np.hstack([H_f, -Phi_f_T])
        beq = -C_f
        vras = np.concatenate([qdd, lambd])
        prog.AddLinearEqualityConstraint(Aeq, beq, vras)

        ''' Eq(12) '''
        alpha = 0.1
        # TODO: Double check
        Jd_qd_foots = []
        for frame, active_contacts in active_contacts_per_frame.items():
            if active_contacts.size:
                Jd_qd_foot = self.plant.CalcBiasTranslationalAcceleration(plant_context, JacobianWrtVariable.kV, self.plant.GetFrameByName(frame),
                                                                          active_contacts, self.plant.world_frame(), self.plant.world_frame())
                Jd_qd_foots.append(Jd_qd_foot.flatten())
        Jd_qd = np.concatenate(Jd_qd_foots)
        assert(Jd_qd.shape == (N_c * 3,))
        Aeq = np.hstack([J, -1 * np.eye(N=J.shape[0])])
        beq = -alpha * J.dot(qd) - Jd_qd
        vras = np.concatenate([qdd, eta])
        # prog.AddLinearEqualityConstraint(Aeq, beq, vras)

        ''' Eq(13) '''
        A = np.hstack([H_a, -Phi_a_T])
        lb = self.B_a.dot(-self.sorted_max_efforts) - C_a
        ub = self.B_a.dot(self.sorted_max_efforts) - C_a
        vars = np.concatenate([qdd, lambd])
        prog.AddLinearConstraint(A, lb, ub, vars)
        # prog.AddBoundingBoxConstraint(-200, 200, lambd).evaluator().set_description("Eq(13)")

        ''' Eq(14) '''
        for j in range(N_c):
            Aeq = np.hstack([v[:, j].T, np.diag([-1, -1, -1])])
            beq = np.zeros((3, 1))
            vras = np.concatenate([beta[:, j], lambd[N_f * j:N_f * j + 3]])
            prog.AddLinearEqualityConstraint(Aeq, beq, vras)

        ''' Eq(15) '''
        prog.AddBoundingBoxConstraint(0, np.inf, beta).evaluator().set_description("Eq(15)")

        ''' Eq(16) '''
        prog.AddBoundingBoxConstraint(eta_min, eta_max, eta).evaluator().set_description("Eq(16)")

        ''' Enforce u as com_dd '''
        Aeq = np.hstack([self.plant.CalcJacobianCenterOfMassTranslationalVelocity(plant_context, JacobianWrtVariable.kV,
                                                                                  self.plant.world_frame(), self.plant.world_frame()), np.diag([-1, -1, -1])])
        beq = self.plant.CalcBiasCenterOfMassTranslationalAcceleration(plant_context, JacobianWrtVariable.kV,
                                                                       self.plant.world_frame(), self.plant.world_frame())
        vras = np.concatenate([qdd, u])
        prog.AddLinearEqualityConstraint(Aeq, beq, vras)

        ''' Respect joint limits '''
        for name, limit in self.joint_limits.items():
            # Get the corresponding joint value
            joint_pos = self.plant.GetJointByName(name).get_angle(plant_context)
            # Get the corresponding actuator index
            act_idx = self.getActuatorIndex(self.plant, name)
            # Use the actuator index to find the corresponding generalized coordinate index
            # q_idx = np.where(B_7[:,act_idx] == 1)[0][0]
            q_idx = self.getJointIndexInGeneralizedVelocities(self.plant, name)

            # if joint_pos >= limit.upper:
            #     print(f"Joint {name} max reached")
            #     prog.AddLinearConstraint(qdd[q_idx] <= 0.0).evaluator().set_description(f"Joint[{q_idx}] upper limit")
            # elif joint_pos <= limit.lower:
            #     print(f"Joint {name} min reached")
            #     prog.AddLinearConstraint(qdd[q_idx] >= 0.0).evaluator().set_description(f"Joint[{q_idx}] lower limit")

        return prog

    def CalcTau(self, context, output):
        calc_t0 = time.time()
        q_v = self.get_input_port(0).Eval(context)
        traj = self.get_input_port(1).Eval(context)

        if not np.array_equal(q_v, self.plant.GetPositionsAndVelocities(self.context)):
            self.plant.SetPositionsAndVelocities(self.context, q_v)

        r, rd, rdd, q_des, v_des, vd_des, angularSpM_des, contact_des = np.split(
            traj, [3, 6, 9, 9 + self.nq, 9 + self.nq + self.nv, 9 + self.nq + self.nv + self.nv, 9 + self.nq + self.nv + self.nv + 3])

        start_formulate_time = time.time()
        def V(x, u): return self.V_full(x, u, r, rd, rdd)
        prog = self.create_qp1(self.context, V, q_des,
                               v_des, vd_des, contact_des)
        end_formulate_time = time.time()
        if not prog:
            print("Invalid program!")
            output.SetFromVector([0] * self.plant.num_actuated_dofs())
            return
        start_solve_time = time.time()
        result = Solve(prog)
        end_solve_time = time.time()
        if not result.is_success():
            print(f"wbc solver failed !!  time: ", context.get_time())
            output.SetFromVector([0] * self.plant.num_actuated_dofs())
        qdd_sol = result.GetSolution(self.qdd)
        lambd_sol = result.GetSolution(self.lambd)
        if contact_des == 3:
            lambd_sol = np.array([0] * lambd_sol.shape[0])
        x_sol = result.GetSolution(self.x)
        u_sol = result.GetSolution(self.u)
        beta_sol = result.GetSolution(self.beta)
        eta_sol = result.GetSolution(self.eta)

        # print(lambd_sol.T)
        # print(v_des.T)
        # print(vd_des.T)
        # print()

        tau = self.tau(qdd_sol, lambd_sol)

        if contact_des != 3:
            k_contactSpM = -20.
            SpMtorque = k_contactSpM * \
                (self.contact_SpaMomvec[1] - angularSpM_des[1])
            if SpMtorque > 10:
                SpMtorque = 10
            if SpMtorque < -10:
                SpMtorque = -10

            tau[2] = tau[2] - SpMtorque

            lf_limits = 0.07 * (lambd_sol[2] + lambd_sol[5])
            if tau[2] > lf_limits:
                tau[2] = lf_limits
            if tau[2] < -lf_limits:
                tau[2] = -lf_limits

        # print(tau)

        output.SetFromVector(tau)
        calc_t1 = time.time()

        plot_cost.append([result.EvalBinding(self.cost_v), result.EvalBinding(self.cost_qdd),
                          result.EvalBinding(self.cost_epsilon), result.EvalBinding(self.cost_eta)])
        plot_x.append(self.x)
        plot_x_des.append(np.concatenate((r, rd)))
        plot_u.append(u_sol)
        plot_u_des.append(rdd)
        plot_tau.append(tau)
        plot_q.append(q_v[:self.nq])
        plot_q_des.append(q_des)
        plot_v.append(q_v[self.nv:])
        plot_v_des.append(v_des)
        plot_lambda.append(lambd_sol)
        plot_CM.append(self.CMvec)
        plot_contact_SpaMom.append(self.contact_SpaMomvec)
        plot_SpM_des.append(angularSpM_des)

        # print("Solver: {}".format(result.get_solver_id().name()))
        # print(f"Cost: {result.get_optimal_cost()}")
        # print('Context: {:.3f}, Formulate: {:.4f}, Solve: {:.4f}, Calc: {:.4f}'.format(context.get_time(),
        #                                                                                end_formulate_time - start_formulate_time,
        #                                                                                end_solve_time - start_solve_time,
        #                                                                                calc_t1 - calc_t0), end='\r')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_realtime_rate", type=float, default=1.0,
                        help="Desired rate relative to real time.  See documentation for "
                        "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument("--simulation_time", type=float, default=4,
                        help="Desired duration of the simulation in seconds.")
    parser.add_argument("--time_step", type=float, default=mbp_time_step,
                        help="If greater than zero, the plant is modeled as a system with "
                        "discrete updates and period equal to this time_step. "
                        "If 0, the plant is modeled as a continuous system.")
    args = parser.parse_args()

    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = builder.AddSystem(MultibodyPlant(time_step=args.time_step))
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    package_map = PackageMap()
    package_map.PopulateFromFolder(__model_pack_file)
    parser = Parser(plant)
    parser.package_map().AddMap(package_map)
    parser.AddModelFromFile(__model_file)

    def add_ground(plant):
        color = np.array([0.9, 0.9, 0.9, 1.0])
        plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(), HalfSpace(), "GroundVisuaGeometry", color)
        ground_friction = CoulombFriction(1.0, 1.0)
        plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(), "GroundCollisionGeometry", ground_friction)
        plant.set_penetration_allowance(1.0e-3)
        plant.set_stiction_tolerance(1.0e-3)

    add_ground(plant)
    plant.Finalize()
    assert plant.geometry_source_is_registered()
    builder.Connect(scene_graph.get_query_output_port(),
                    plant.get_geometry_query_input_port())
    builder.Connect(plant.get_geometry_poses_output_port(),
                    scene_graph.get_source_pose_port(plant.get_source_id()))

    # args.simulation_time = load_traj()

    controller = builder.AddSystem(WholeBodyController(plant, __end_frames_name))
    planner = builder.AddSystem(TrajPlanner(plant, __end_frames_name, __robot_instance))

    builder.Connect(plant.get_state_output_port(),
                    controller.get_input_port(0))
    builder.Connect(plant.get_state_output_port(), planner.get_input_port(0))
    builder.Connect(planner.get_output_port(0), controller.get_input_port(1))
    builder.Connect(controller.get_output_port(0), plant.get_actuation_input_port())

    # DrakeVisualizer.AddToBuilder(builder=builder, scene_graph=scene_graph)
    proc, zmq_url, web_url = start_zmq_server_as_subprocess()
    visualizer = ConnectMeshcatVisualizer(builder=builder, scene_graph=scene_graph, zmq_url=zmq_url)
    contact_vis = builder.AddSystem(MeshcatContactVisualizer(meshcat_viz=visualizer,
                                                             plant=plant,
                                                             contact_force_scale=600))
    contact_input_port = contact_vis.GetInputPort("contact_results")
    builder.Connect(plant.get_contact_results_output_port(),
                    contact_input_port)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    initial_state(plant, plant_context, __end_frames_name, __initial_joint_name, __initial_joint_pos)
    qv = plant.GetPositionsAndVelocities(plant_context)

    planner.Initial(qv)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(args.target_realtime_rate)
    simulator.Initialize()

    visualizer.load()
    visualizer.start_recording()
    simulator.AdvanceTo(args.simulation_time)
    visualizer.stop_recording()
    visualizer.publish_recording()
    print('')


if __name__ == "__main__":
    try:
        np.set_printoptions(linewidth=1000)
        main()

        plot_cost = np.asarray(plot_cost)
        plot_x = np.asarray(plot_x)
        plot_x_des = np.asarray(plot_x_des)
        plot_u = np.asarray(plot_u)
        plot_u_des = np.asarray(plot_u_des)
        plot_tau = np.asarray(plot_tau)
        plot_q = np.asarray(plot_q)
        plot_q_des = np.asarray(plot_q_des)
        plot_v = np.asarray(plot_v)
        plot_v_des = np.asarray(plot_v_des)
        plot_lambda = np.asarray(plot_lambda)
        plot_CM = np.asarray(plot_CM)
        plot_contact_SpaMom = np.asarray(plot_contact_SpaMom)
        plot_SpM_des = np.asarray(plot_SpM_des)
        plot_theta = np.asarray(plot_theta)
        plot_torsoRot = np.asarray(plot_torsoRot)

        N = plot_cost.shape[0]
        T = N * mbp_time_step
        x = np.linspace(0, T, N)
        rows = 2
        clos = 2

        plt.figure()
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=0.15, hspace=0.15)
        y = []
        label = []
        for i in range(plot_cost.shape[1]):
            y.append(plot_cost[:, i])
            label.append('cost{}'.format(i))
        plt.subplot(rows, clos, 1)
        plot_line(x, y, label)

        y = []
        label = []
        for i in range(plot_x.shape[1]):
            y.append(plot_x[:, i])
            label.append('x{}'.format(i))
        for i in range(plot_x_des.shape[1]):
            y.append(plot_x_des[:, i])
            label.append('x_des{}'.format(i))
        plt.subplot(rows, clos, 2)
        plot_line(x, y, label)

        y = []
        label = []
        for i in range(plot_u.shape[1]):
            y.append(plot_u[:, i])
            label.append('u{}'.format(i))
        for i in range(plot_u_des.shape[1]):
            y.append(plot_u_des[:, i])
            label.append('u_des{}'.format(i))
        plt.subplot(rows, clos, 3)
        plot_line(x, y, label)

        y = []
        label = []
        for i in range(plot_tau.shape[1]):
            y.append(plot_tau[:, i])
            label.append('tau{}'.format(i))
        plt.subplot(rows, clos, 4)
        plot_line(x, y, label)

        plt.figure()
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=0.15, hspace=0.15)
        y = []
        label = []
        for i in range(plot_q.shape[1] - 6):
            y.append(plot_q[:, i])
            label.append('q{}'.format(i))
        plt.subplot(rows, clos, 1)
        plot_line(x, y, label)

        y = []
        label = []
        for i in range(plot_q_des.shape[1] - 6):
            y.append(plot_q_des[:, i])
            label.append('q_des{}'.format(i))
        plt.subplot(rows, clos, 2)
        plot_line(x, y, label)

        y = []
        label = []
        for i in range(plot_v.shape[1]):
            y.append(plot_v[:, i])
            label.append('v{}'.format(i))
        plt.subplot(rows, clos, 3)
        plot_line(x, y, label)

        y = []
        label = []
        for i in range(plot_v_des.shape[1] - 12):
            y.append(plot_v_des[:, 6 + i])
            label.append('v_des{}'.format(i))
        plt.subplot(rows, clos, 4)
        plot_line(x, y, label)

        plt.figure()
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=0.15, hspace=0.15)
        y = []
        label = []
        for i in range(plot_lambda.shape[1]):
            y.append(plot_lambda[:, i])
            label.append('lambda{}'.format(i))
        plt.subplot(rows, clos, 1)
        plot_line(x, y, label)

        y = []
        label = []
        for i in range(plot_CM.shape[1]):
            y.append(plot_CM[:, i])
            label.append('CM{}'.format(i))
        plt.subplot(rows, clos, 2)
        plot_line(x, y, label)

        y = []
        label = []
        for i in range(plot_contact_SpaMom.shape[1]):
            y.append(plot_contact_SpaMom[:, i])
            label.append('contact_SpaMom{}'.format(i))
        plt.subplot(rows, clos, 3)
        plot_line(x, y, label)
        y = []
        label = []
        for i in range(plot_SpM_des.shape[1]):
            y.append(plot_SpM_des[:, i])
            label.append('SpM_des{}'.format(i))
        plt.subplot(rows, clos, 4)
        plot_line(x, y, label)

        plt.figure()
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=0.15, hspace=0.15)
        y = []
        label = []
        for i in range(1):
            y.append(plot_theta[:])
            label.append('theta{}'.format(i))
        plt.subplot(rows, clos, 1)
        plot_line(x, y, label)
        y = []
        label = []
        for i in range(plot_torsoRot.shape[1]):
            y.append(plot_torsoRot[:, i])
            label.append('torsoRot{}'.format(i))
        plt.subplot(rows, clos, 2)
        plot_line(x, y, label)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                            top=0.95, wspace=0.1, hspace=0.15)
        plt.show()

        print('Program end, Press Ctrl + C to exit.')
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("")
