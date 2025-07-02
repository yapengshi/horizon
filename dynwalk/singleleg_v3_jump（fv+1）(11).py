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
from scipy.interpolate import interp1d # 导入插值库
import sys # 用于在文件加载失败时退出程序

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
    r0 = plant.CalcCenterOfMassPositionInWorld(plant_context)
    for i, name in enumerate(initial_joint_name):
        joint = plant.GetJointByName(name=name)
        joint.set_angle(plant_context, initial_joint_pos[i])
    q0 = plant.GetPositions(plant_context)
    print(q0)

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
    # 构造函数需要 dt_sim 参数，请确保在创建实例时传入
    # 例如: planner = builder.AddSystem(TrajPlanner(plant, __end_frames_name, __robot_instance, dt_sim=mbp_time_step))
    def __init__(self, plant, end_frames_name, robot_instance, dt_sim, nq_f=7, nv_f=6):
        LeafSystem.__init__(self)
        self.plant = plant
        self.nq = plant.num_positions()  # 应该是 10
        self.nv = plant.num_velocities()  # 应该是 9
        self.dt = dt_sim

        print(f"TrajPlanner initialized with nq={self.nq}, nv={self.nv}") # 调试信息

        self.DeclareVectorInputPort("state", BasicVector(self.nq + self.nv))
        self.DeclareVectorOutputPort("traj_des", BasicVector(3*3 + self.nq + self.nv + self.nv + 3 + 1), self.Output)

        # --- 从旧代码中引入的部分 ---
        plant_context_temp = self.plant.CreateDefaultContext()
        self.total_mass = sum(self.plant.get_body(index).get_mass(plant_context_temp)
                              for index in self.plant.GetBodyIndices(self.plant.GetModelInstanceByName(robot_instance)))
        self.v_des_pre = np.zeros(self.nv)
        
        # --- 开放式框架的参数 ---
        self.takeoff_contact_point = np.array([0., 0., 0.]) 
        self.landing_contact_point = np.array([0., 0., 0.]) # 假设是原地跳跃，落点和起点相同
        
        self.flight_start_time = 1.0
        self.touchdown_start_time = 1.5
        self.trajectory_end_time = 2.5

        # --- 加载轨迹 ---
        try:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            acc_traj_path = os.path.join(script_dir, 'trajectory_data/com_acceleration_traj.csv')
            acc_data = np.genfromtxt(acc_traj_path, delimiter=',', skip_header=1)
            state_traj_path = os.path.join(script_dir, 'trajectory_data/full_state_traj.csv')
            state_data = np.genfromtxt(state_traj_path, delimiter='\t', skip_header=1) # **** 注意：您的CSV文件是用制表符(\t)分隔的，不是逗号 ****
            print("Successfully loaded CoM and full state trajectories.")
        except (IOError, ValueError) as e:
            print(f"FATAL ERROR loading trajectory data: {e}")
            # 额外调试信息
            if isinstance(e, ValueError):
                print("This might be a delimiter issue. Check if your CSV is comma or tab separated.")
            sys.exit(1)

        # 2. 加载【完整的】状态轨迹
        try:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            # 加载CoM加速度 (仍然需要它作为rdd)
            acc_traj_path = os.path.join(script_dir, 'trajectory_data/com_acceleration_traj.csv')
            # 假设 acc 文件是逗号分隔
            acc_data = np.genfromtxt(acc_traj_path, delimiter=',', skip_header=1) 
            
            # 加载完整的q, v状态
            state_traj_path = os.path.join(script_dir, 'trajectory_data/full_state_traj.csv')
            
            # ********** 关键修改：尝试两种分隔符 **********
            try:
                # 尝试制表符
                print("Attempting to load state trajectory with TAB delimiter...")
                state_data = np.genfromtxt(state_traj_path, delimiter='\t', skip_header=1)
                if state_data.ndim != 2 or state_data.shape[1] < 2:
                    raise ValueError("Loaded data is not a 2D array with TABs.")
                print("Successfully loaded with TAB delimiter.")
            except ValueError:
                # 如果制表符失败，尝试逗号
                print("TAB delimiter failed. Attempting to load state trajectory with COMMA delimiter...")
                state_data = np.genfromtxt(state_traj_path, delimiter=',', skip_header=1)
                if state_data.ndim != 2 or state_data.shape[1] < 2:
                     raise ValueError("Loaded data is not a 2D array with COMMAs either.")
                print("Successfully loaded with COMMA delimiter.")
            
            # ********** 调试打印 **********
            print("\n--- DEBUG INFO ---")
            print(f"Path to state_traj_path: {state_traj_path}")
            print(f"Type of state_data: {type(state_data)}")
            print(f"Shape of state_data: {state_data.shape}")
            print(f"state_data is {state_data.ndim} dimensional.")
            print("First 3 rows of state_data:")
            print(state_data[:3, :])
            print("--- END DEBUG INFO ---\n")
            
            print("Successfully loaded CoM and full state trajectories.")

        except (IOError, ValueError) as e:
            print(f"FATAL ERROR loading trajectory data: {e}")
            # 检查文件是否存在
            if not os.path.exists(state_traj_path):
                print(f"FILE NOT FOUND at: {state_traj_path}")
            sys.exit(1)

        # --- 创建插值函数 ---
        self.time_acc = acc_data[:, 0]
        self.time_state = state_data[:, 0]
        
        # CoM位置轨迹 (从q重新计算，保持不变)
        com_fun = plant.CalcCenterOfMassPositionInWorld
        r_traj = []
        for i in range(state_data.shape[0]):
            # q_i 是从第1列到第1+nq列
            q_i = state_data[i, 1 : 1 + self.nq]
            plant.SetPositions(plant_context_temp, q_i)
            r_traj.append(com_fun(plant_context_temp))
        r_traj = np.array(r_traj)
        
        self.r_interp = [interp1d(self.time_state, r_traj[:, i], kind='linear', fill_value='extrapolate') for i in range(3)]

        # CoM速度轨迹 (这是关键修正点)
        # 根据CSV文件，浮动基座的线速度(vx, vy, vz)是v_0, v_1, v_2
        # 它们位于第 1+nq, 1+nq+1, 1+nq+2 列 (即索引 11, 12, 13)
        # Drake中，v的前6个元素是 [omega_x, omega_y, omega_z, vx, vy, vz]
        # CSV中，v的前6个元素是 [vx, vy, vz, omega_x, omega_y, omega_z] -- 这是一个常见的顺序差异！
        # 我们假设CSV中的 v_0, v_1, v_2 就是 CoM 的线速度 vx, vy, vz。
        # **但是从您打印的数据看，v_0, v_1, v_2 好像不是CoM线速度**
        # 从您打印的rd_interp数据看，v_3, v_4, v_5才是非零的。在CSV中，它们是第14, 15, 16列。
        # 让我们假设CSV中 v_3, v_4, v_5 代表的是 CoM线速度 vx, vy, vz
        # Drake的v向量是[omega; v_linear]，所以CoM速度索引是 v[3], v[4], v[5]
        # CSV中，v_0...v_8，列索引是11...19
        # 假设CoM速度是CSV中的 v_0, v_1, v_2 (列索引 11, 12, 13)
        # 1 + self.nq = 1 + 10 = 11.
        com_vel_start_col = 1 + self.nq
        self.rd_interp = [interp1d(self.time_state, state_data[:, com_vel_start_col + i], kind='linear', fill_value='extrapolate') for i in range(3)]
        
        # CoM加速度轨迹
        self.rdd_interp = [interp1d(self.time_acc, acc_data[:, i + 1], kind='linear', fill_value='extrapolate') for i in range(3)]
        
        # 关节轨迹
        self.q_des_interp = [interp1d(self.time_state, state_data[:, 1 + i], kind='linear', fill_value='extrapolate') for i in range(self.nq)]
        self.v_des_interp = [interp1d(self.time_state, state_data[:, 1 + self.nq + i], kind='linear', fill_value='extrapolate') for i in range(self.nv)]
        
        self.final_traj_packet = None

    def Output(self, context, output):
        if self.final_traj_packet is not None:
            output.SetFromVector(self.final_traj_packet)
            return

        t = context.get_time()
        
        # 1. 确定接触状态和接触点
        contact_point = None  # 默认飞行阶段
        if t < self.flight_start_time:
            contact_des = np.array([2])
            contact_point = self.takeoff_contact_point
        elif t < self.touchdown_start_time:
            contact_des = np.array([3])
            # contact_point 保持为 None
        else: # touchdown and after
            contact_des = np.array([4])
            contact_point = self.landing_contact_point
        
        t_clamped = np.clip(t, self.time_state[0], self.time_state[-1])

        # 2. 从插值器获取基础期望值
        r = np.array([interp(t_clamped) for interp in self.r_interp])
        rd = np.array([interp(t_clamped) for interp in self.rd_interp])
        rdd = np.array([interp(t_clamped) for interp in self.rdd_interp])
        q_des = np.array([interp(t_clamped) for interp in self.q_des_interp])
        v_des = np.array([interp(t_clamped) for interp in self.v_des_interp])

        # ==================== vd_des 计算 ====================
        vd_des = (v_des - self.v_des_pre) / self.dt
        np.clip(vd_des, -300, 300, out=vd_des)
        self.v_des_pre = v_des
        
        # ================== angularSpM_des 计算 ==================
        cm_rot = np.zeros(3)
        if contact_point is not None:
            l = r - contact_point
            angularSpM_des = np.cross(l, rd * self.total_mass) + cm_rot
        else:
            angularSpM_des = np.zeros(3)
            
        # 3. 打包发送
        traj = np.concatenate([r, rd, rdd, q_des, v_des, vd_des, angularSpM_des, contact_des])

        if t >= self.trajectory_end_time:
            if self.final_traj_packet is None:
                 self.final_traj_packet = traj

        output.SetFromVector(traj)
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
        self.K_p = [100] * self.nv
        self.K_d = [20] * self.nv
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
    planner = builder.AddSystem(TrajPlanner(plant, __end_frames_name, __robot_instance, dt_sim=mbp_time_step)) 

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

    # planner.Initial(qv)

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
        # --- Start of Replacement Block ---
        
        np.set_printoptions(linewidth=1000)
        main()

        # Convert lists to numpy arrays for easier manipulation
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
        
        # 检查是否有数据，如果没有则不绘图
        if plot_cost.shape[0] == 0:
            print("\nNo data to plot. Exiting.")
            sys.exit()

        N = plot_cost.shape[0]
        T = N * mbp_time_step
        x_axis = np.linspace(0, T, N)
        rows = 2
        clos = 2

        # --- Figure 1: CoM State and Control ---
        plt.figure(figsize=(15, 10))
        plt.suptitle("Center of Mass Trajectory Tracking", fontsize=16)

        # CoM Position
        plt.subplot(rows, clos, 1)
        for i in range(3):
            plt.plot(x_axis, plot_x[:, i], label=f'Actual Pos Dim {i}')
            plt.plot(x_axis, plot_x_des[:, i], '--', label=f'Desired Pos Dim {i}')
        plt.title("CoM Position (x, y, z)")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()
        plt.grid(True)

        # CoM Velocity
        plt.subplot(rows, clos, 2)
        for i in range(3):
            plt.plot(x_axis, plot_x[:, i+3], label=f'Actual Vel Dim {i}')
            plt.plot(x_axis, plot_x_des[:, i+3], '--', label=f'Desired Vel Dim {i}')
        plt.title("CoM Velocity (vx, vy, vz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()
        plt.grid(True)
        
        # CoM Acceleration (Control Input u)
        plt.subplot(rows, clos, 3)
        for i in range(plot_u.shape[1]):
            plt.plot(x_axis, plot_u[:, i], label=f'Actual Acc Dim {i}')
            plt.plot(x_axis, plot_u_des[:, i], '--', label=f'Desired Acc Dim {i}')
        plt.title("CoM Acceleration (ax, ay, az)")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s^2)")
        plt.legend()
        plt.grid(True)

        # Actuator Torques
        plt.subplot(rows, clos, 4)
        for i in range(plot_tau.shape[1]):
            plt.plot(x_axis, plot_tau[:, i], label=f'Joint {i+1} Torque')
        plt.title("Actuator Torques")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # --- Figure 2: Joint Space Tracking ---
        plt.figure(figsize=(15, 10))
        plt.suptitle("Joint Space Trajectory Tracking", fontsize=16)

        num_q_base = 7  # 7 DoF for floating base position (quaternion)
        num_v_base = 6  # 6 DoF for floating base velocity

        # Joint Positions
        plt.subplot(rows, clos, 1)
        for i in range(num_q_base, plot_q.shape[1]):
            plt.plot(x_axis, plot_q[:, i], label=f'Actual q joint {i-num_q_base+1}')
        plt.title("Actual Joint Positions")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")
        plt.legend()
        plt.grid(True)
        
        # Desired Joint Positions
        plt.subplot(rows, clos, 2)
        for i in range(num_q_base, plot_q_des.shape[1]):
            plt.plot(x_axis, plot_q_des[:, i], '--', label=f'Desired q joint {i-num_q_base+1}')
        plt.title("Desired Joint Positions")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")
        plt.legend()
        plt.grid(True)

        # Joint Velocities
        plt.subplot(rows, clos, 3)
        for i in range(num_v_base, plot_v.shape[1]):
            plt.plot(x_axis, plot_v[:, i], label=f'Actual v joint {i-num_v_base+1}')
        plt.title("Actual Joint Velocities")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.legend()
        plt.grid(True)

        # Desired Joint Velocities
        plt.subplot(rows, clos, 4)
        for i in range(num_v_base, plot_v_des.shape[1]):
            plt.plot(x_axis, plot_v_des[:, i], '--', label=f'Desired v joint {i-num_v_base+1}')
        plt.title("Desired Joint Velocities")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (rad/s)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # --- Figure 3: Contact Forces ---
        if plot_lambda.shape[1] > 0:
            plt.figure(figsize=(10, 5))
            for i in range(plot_lambda.shape[1]):
                plt.plot(x_axis, plot_lambda[:, i], label=f'Force Component {i}')
            plt.title("Contact Forces (lambda)")
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

        plt.show()

        print('Program end, Press Ctrl + C to exit.')
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nPlotting interrupted by user. Exiting.")

        # --- End of Replacement Block ---
