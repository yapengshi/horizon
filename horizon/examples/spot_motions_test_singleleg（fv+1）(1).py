#!/usr/bin/env python3

from horizon import problem
from horizon.utils import utils, kin_dyn, resampler_trajectory, plotter, mat_storer
from horizon.transcriptions.transcriptor import Transcriptor
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from horizon.solvers import solver
import os, argparse
from itertools import filterfalse
import numpy as np
import casadi as cs

def str2bool(v):
  #susendberg's function
  #将字符串v转换为布尔值
  return v.lower() in ("yes", "true", "t", "1")

def main(args):

    action = args.action
    rviz_replay = args.replay
    solver_type = args.solver
    codegen = args.codegen
    warmstart_flag = args.warmstart
    plot_sol = args.plot

    if codegen:
        if args.solver == 'ilqr':
            input("code for ilqr will be generated in: '/tmp/spot_motions'. Press a key to resume. \n")
        else:
            input("codegen available only for 'ilqr' solver. Will be ignored. Press a key to resume. \n")

    resampling = False
    load_initial_guess = False

    if rviz_replay:
        from horizon.ros.replay_trajectory import replay_trajectory
        import rospy
        plot_sol = False


    path_to_examples = os.path.dirname(os.path.realpath(__file__))

    # mat storer
    if warmstart_flag:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        save_dir = path_to_examples + '/mat_files'
        save_file = path_to_examples + f'/mat_files/{file_name}.mat'

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        ms = mat_storer.matStorer(save_file)

        if os.path.isfile(save_file):
            print(f'{file_name}.mat file found. Using previous solution as initial guess.')
            load_initial_guess = True
        else:
            print(f'{file_name}.mat file NOT found. The solution will be saved for future warmstarting.')

    # options
    transcription_method = 'multiple_shooting'
    transcription_opts = dict(integrator='RK4')

    tf = 2.5
    n_nodes = 50

    disp = [0., 0., 0., 0., 0., 0., 1.]

    if action == 'jump_forward' or action == 'leap':
        disp[0] = 2 # [m]
    if action == 'wheelie':
        node_action = (20, n_nodes)
    if action == 'jump_up':
        node_action = (20, 30)
    elif action == 'leap':
        node_action = [(15, 35), (30, 45)]
    else:
        node_action = (20, 30)


    # load urdf
    urdffile = os.path.join(path_to_examples, 'urdf', 'singleleg_v3_symmetrical.urdf')
    urdf = open(urdffile, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    # joint names
    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')

    contacts_name = ['foot']

    # parameters
    n_c = 1
    n_q = kindyn.nq()
    n_v = kindyn.nv()
    n_f = 3
    dt = tf / n_nodes

    # define dynamics
    prb = problem.Problem(n_nodes)
    q = prb.createStateVariable('q', n_q)
    q_dot = prb.createStateVariable('q_dot', n_v)
    q_ddot = prb.createInputVariable('q_ddot', n_v)
    f_list = [prb.createInputVariable(f'force_{i}', n_f) for i in contacts_name]
    x, x_dot = utils.double_integrator_with_floating_base(q, q_dot, q_ddot)
    prb.setDynamics(x_dot)
    prb.setDt(dt)
    # contact map
    contact_map = dict(zip(contacts_name, f_list))

    # import initial guess if present
    if load_initial_guess:
        if os.path.exists(path_to_examples + f'/mat_files/{file_name}.mat'):
            prev_solution = ms.load()
            q_ig = prev_solution['q']
            q_dot_ig = prev_solution['q_dot']
            q_ddot_ig = prev_solution['q_ddot']
            f_ig_list = [prev_solution[f.getName()] for f in f_list]


    # initial state and initial guess
    q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                       0.0, 0.9, -1.5238505,
                       ])
    
    # 临时创建一个 FK 函数
    FK_foot = cs.Function.deserialize(kindyn.fk('foot'))
    temp_q = np.array([0., 0., 0., 0., 0., 0., 1., 0.0, 0.9, -1.5238505])
    foot_pos_init = FK_foot(q=temp_q)['ee_pos'] # 这会得到脚在 base_link 坐标系下的位置
    print(f"Initial foot position relative to base: {foot_pos_init}")

    # 假设地面在 z=0, 脚底本身还有个偏移 -0.028 (来自URDF <frame name="foot_sole">)
    initial_foot_z = foot_pos_init[2] - 0.028

    # 设置正确的初始q
    q_init = np.array([0.0, 0.0, -initial_foot_z, 0.0, 0.0, 0.0, 1.0,
                   0.0, 0.9, -1.5238505,
                   ])
    
    q.setBounds(q_init, q_init, 0)
    q_dot.setBounds(np.zeros(n_v), np.zeros(n_v), 0)
    q.setInitialGuess(q_init)

    if load_initial_guess:
        q.setInitialGuess(q_ig)
        q_dot.setInitialGuess(q_dot_ig)
        q_ddot.setInitialGuess(q_ddot_ig)
        [f.setInitialGuess(f_ig) for f, f_ig in zip(f_list, f_ig_list)]
    else:
        [f.setInitialGuess([0, 0, 55]) for f in f_list]

    # transcription
    if solver_type != 'ilqr':
        th = Transcriptor.make_method(transcription_method, prb, opts=transcription_opts)

    # dynamic feasibility
    id_fn = kin_dyn.InverseDynamics(kindyn, contact_map.keys(), cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)
    tau = id_fn.call(q, q_dot, q_ddot, contact_map)
    prb.createIntermediateConstraint("dynamic_feasibility", tau[:6])

#    # =========================================================
#     # 创建质心加速度函数 (最终修正版)
#     # =========================================================
#     # 定义符号变量
#     q_sym = cs.MX.sym('q', n_q)
#     q_dot_sym = cs.MX.sym('q_dot', n_v)
#     q_ddot_sym = cs.MX.sym('q_ddot', n_v)

#     # 1. 获取计算质心速度的函数
#     # 这个函数直接将 q 和 q_dot (nv维) 映射到 v_com
#     v_com_fun = cs.Function.deserialize(kindyn.centerOfMassVelocity())
#     v_com_expr = v_com_fun(q=q_sym, qdot=q_dot_sym)['vcom']

#     # 2. 手动计算质心加速度 a_com
#     # a_com 是 v_com_expr 相对于时间 t 的全导数
#     # d(v_com)/dt = (∂v_com/∂q) * q_dot + (∂v_com/∂q_dot) * q_ddot
#     # 我们再次使用 jtimes 来实现这个全导数
#     a_com = cs.jtimes(v_com_expr, cs.vertcat(q_sym, q_dot_sym), cs.vertcat(q_dot_sym, q_ddot_sym))

#     # 3. 创建可调用的 CasADi 函数
#     com_acc_fun = cs.Function('com_acceleration', [q_sym, q_dot_sym, q_ddot_sym], [a_com], ['q', 'q_dot', 'q_ddot'], ['a_com'])
#     # =========================================================

    # final velocity is zero
    prb.createFinalConstraint('final_velocity', q_dot)


    # contact handling
    k_all = range(1, n_nodes + 1)
    if action == 'leap':
        list_swing = [list(range(*n_range)) for n_range in node_action]
        k_swing = [item for sublist in list_swing for item in sublist]
        k_swing_front = list(range(*[node for node in node_action[0]]))
        k_swing_hind = list(range(*[node for node in node_action[1]]))
        k_stance_front = list(filterfalse(lambda k: k in k_swing_front, k_all))
        k_stance_hind = list(filterfalse(lambda k: k in k_swing_hind, k_all))

    else:
        k_swing = list(range(*[node for node in node_action]))
    
    k_stance = list(filterfalse(lambda k: k in k_swing, k_all))
    
    # list of lifted legs
    lifted_legs = ['foot']

    if action != 'wheelie' and action != 'jump_on_wall' and action != 'jump_up':
        lifted_legs.extend(['foot_heel_l', 'foot_heel_r'])

    q_final = q_init

    def barrier(x):
        return cs.sum1(cs.if_else(x > 0, 0, x ** 2))

    # 【添加这行】创建一个空字典来保存运动学函数
    kinematic_functions = dict()

    for frame, f in contact_map.items():
        nodes_stance = k_stance if frame in lifted_legs else k_all

        if action == 'leap':
            nodes_stance = k_stance_front if frame in ['foot_toe_l', 'foot_toe_r'] else k_stance_hind
            nodes_swing = k_swing_front if frame in ['foot_toe_l', 'foot_toe_r'] else k_swing_hind

        FK = cs.Function.deserialize(kindyn.fk(frame))
        DFK = cs.Function.deserialize(kindyn.frameVelocity(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))
        DDFK = cs.Function.deserialize(kindyn.frameAcceleration(frame, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED))

        # 【添加这行】将函数存入字典，键是 frame 的名字 (例如 'foot')
        kinematic_functions[frame] = {'FK': FK, 'DFK': DFK, 'DDFK': DDFK}

        p = FK(q=q)['ee_pos']
        p_start = FK(q=q_init)['ee_pos']
        v = DFK(q=q, qdot=q_dot)['ee_vel_linear']
        a = DDFK(q=q, qdot=q_dot)['ee_acc_linear']

        prb.createConstraint(f"{frame}_vel", v, nodes=nodes_stance)
        prb.createIntermediateCost(f'{frame}_fn', barrier(f[2] - 25.0))

        # 【添加】腾空阶段约束
        if frame in lifted_legs: # 确保这是你想让它离地的脚
        # 获取脚底板的初始高度 (应该是0)
            p_start_z = FK(q=q_init)['ee_pos'][2] # 值为 0.028
    
            ground_clearance_cnstr = prb.createConstraint(f"{frame}_ground_clearance", p[2], nodes=k_swing)
    
            # 同时设置下界和上界
            # 下界是脚的初始高度 (0.028)
            # 上界是无穷大，表示对跳跃高度没有硬性限制
            ground_clearance_cnstr.setBounds(p_start_z, np.inf)

        if action == 'leap':
            if solver_type == 'ilqr':
                prb.createIntermediateCost(f'{frame}_ground', 1e3 * barrier(p[2] - p_start[2]), nodes=nodes_swing)
            else:
                gc = prb.createConstraint(f'{frame}_ground', p[2], nodes=nodes_swing)
                gc.setLowerBounds(p_start[2])

        if frame in lifted_legs:
            if action == 'jump_on_wall':
                mu = 1
                p_goal = p_start + [0.3, 0., 0.8]
                rot = -np.pi / 2.
                R_wall = np.array([[np.cos(rot), 0, np.sin(rot)],
                                  [0,            1,           0],
                                  [-np.sin(rot), 0, np.cos(rot)]])

                fc, fc_lb, fc_ub = kin_dyn.linearized_friction_cone(f, mu, R_wall)
                if solver_type == 'ipopt':
                    prb.createIntermediateConstraint(f"{frame}_fc_wall", fc, nodes=range(k_swing[-1], n_nodes), bounds=dict(lb=fc_lb, ub=fc_ub))
                else:
                    prb.createIntermediateCost(f"{frame}_fc_wall_lb", 1 * barrier(fc - fc_lb), nodes=range(k_swing[-1], n_nodes))
                    prb.createIntermediateCost(f"{frame}_fc_wall_ub", 1 * barrier(fc_ub - fc), nodes=range(k_swing[-1], n_nodes))

                prb.createFinalConstraint(f"lift_{frame}_leg", p - p_goal)


    # swing force is zero
    for leg in lifted_legs:
        if action == 'leap':
            nodes = k_swing if leg in ['foot_toe_l', 'foot_toe_r'] else k_swing
        else:
            nodes = k_swing

        fzero = np.zeros(n_f)
        contact_map[leg].setBounds(fzero, fzero, nodes=nodes)


    if action != 'wheelie' and action != 'jump_on_wall':
        if solver_type == 'ilqr':
            prb.createFinalConstraint(f"final_nominal_pos_base", q[:6] - q_final[:6])
            prb.createFinalCost(f"final_nominal_pos_joints", 1e3 * cs.sumsqr(q[7:] - q_final[7:]))
        else:
            prb.createFinalConstraint(f"final_nominal_pos", q - q_final)

    # 添加跳跃高度目标 (最大化 CoM 高度)
    # 计算质心 CoM
    com_fn = cs.Function.deserialize(kindyn.centerOfMass())
    com_pos = com_fn(q=q)['com']

# 添加一个代价函数，鼓励在腾空阶段(k_swing)把CoM抬高
# 这个代价项是整个跳跃动作的核心驱动力
    prb.createIntermediateCost("maximize_com_z", -1e2 * com_pos[2], nodes=k_swing)
    prb.createResidual("min_q_dot", q_dot)
        # prb.createIntermediateResidual("min_q_ddot", 1e-3* (q_ddot))
    for f in f_list:
        prb.createIntermediateResidual(f"min_{f.getName()}", cs.sqrt(3e-3) * f)

    # 【添加全局约束】防止机器人钻入地下
# 约束CoM的Z坐标在所有时间节点都必须大于一个安全值。
# 这个安全值可以是一个小的正数，比如 0.1 米，具体取决于你的机器人模型。
# k_all 应该是 range(0, n_nodes + 1) 或 range(1, n_nodes + 1) 取决于你的定义
# 我们用 range(0, n_nodes + 1) 包含初始节点
    all_nodes_with_init = range(prb.getNNodes() + 1) 
    prb.createConstraint("com_ground_clearance", 
                     com_pos[2], 
                     nodes=all_nodes_with_init, 
                     bounds=dict(lb=0.1, ub=np.inf))

    # =============
    # SOLVE PROBLEM
    # =============

    opts = dict()

    if solver_type == 'ipopt':
        opts['ipopt.tol'] = 0.001
        opts['ipopt.constr_viol_tol'] = n_nodes * 1e-3
        opts['ipopt.max_iter'] = 2000

    if solver_type == 'ilqr':
        opts['ilqr.max_iter'] =  200
        opts['ilqr.integrator'] ='RK4'
        opts['ilqr.closed_loop_forward_pass'] = True
        opts['ilqr.line_search_accept_ratio'] = 1e-9
        opts['ilqr.constraint_violation_threshold'] = 1e-3
        opts['ilqr.step_length_threshold'] = 1e-3
        opts['ilqr.alpha_min'] = 0.2
        opts['ilqr.kkt_decomp_type'] = 'qr'
        opts['ilqr.constr_decomp_type'] = 'qr'
        opts['ilqr.codegen_enabled'] = codegen
        opts['ilqr.codegen_workdir'] = '/tmp/spot_motions'

    if solver_type == 'gnsqp':
        qp_solver = 'osqp'
        if qp_solver == 'osqp':
            opts['gnsqp.qp_solver'] = 'osqp'
            opts['warm_start_primal'] = True
            opts['warm_start_dual'] = True
            opts['merit_derivative_tolerance'] = 1e-3
            opts['constraint_violation_tolerance'] = n_nodes * 1e-3
            opts['osqp.polish'] = True # without this
            opts['osqp.delta'] = 1e-6 # and this, it does not converge!
            opts['osqp.verbose'] = False
            opts['osqp.rho'] = 0.02
            opts['osqp.scaled_termination'] = False
        if qp_solver == 'qpoases': #does not work!
            opts['gnsqp.qp_solver'] = 'qpoases'
            opts['sparse'] = True
            opts["enableEqualities"] = True
            opts["initialStatusBounds"] = "inactive"
            opts["numRefinementSteps"] = 0
            opts["enableDriftCorrection"] = 0
            opts["terminationTolerance"] = 10e9 * 1e-7
            opts["enableFlippingBounds"] = False
            opts["enableNZCTests"] = False
            opts["enableRamping"] = False
            opts["enableRegularisation"] = True
            opts["numRegularisationSteps"] = 2
            opts["epsRegularisation"] = 5. * 10e3 * 1e-7
            opts['hessian_type'] = 'posdef'

    solv = solver.Solver.make_solver(solver_type, prb, opts)

    try:
        solv.set_iteration_callback()
    except:
        pass

    solv.solve()

    if solver_type == 'ilqr':
        solv.print_timings()

    solution = solv.getSolutionDict()
    dt_sol = solv.getDt()
    cumulative_dt = np.zeros(len(dt_sol) + 1)
    for i in range(len(dt_sol)):
        cumulative_dt[i + 1] = dt_sol[i] + cumulative_dt[i]

    solution_constraints_dict = dict()

    if warmstart_flag:
        if isinstance(dt, cs.SX):
            ms.store({**solution, **solution_constraints_dict})
        else:
            dt_dict = dict(dt=dt)
            ms.store({**solution, **solution_constraints_dict, **dt_dict})
        print(ms)  
    
    # ===========================================
    # 计算并打印足端的 p, p_start, v, a 的数值
    # ===========================================
    print("\n--- Numerical Foot Kinematics Analysis ---")

    # 1. 从之前保存的字典中获取 'foot' 的函数
    fk_fun = kinematic_functions['foot']['FK']
    dfk_fun = kinematic_functions['foot']['DFK']
    ddfk_fun = kinematic_functions['foot']['DDFK']

    # 2. 获取优化结果的数值
    sol_q = solution['q']
    sol_q_dot = solution['q_dot']
    sol_q_ddot = solution['q_ddot']

    # 3. 计算 p_start (这是一个单一的向量)
    p_start_val = fk_fun(q=q_init)['ee_pos']
    print(f"\nInitial Foot Position (p_start):\n{np.array(p_start_val).flatten()}")

    # 4. 计算整个轨迹的 p, v, a
    p_trajectory = fk_fun(q=sol_q)['ee_pos']
    v_trajectory = dfk_fun(q=sol_q, qdot=sol_q_dot)['ee_vel_linear']
    a_trajectory = ddfk_fun(q=sol_q[:, :-1], qdot=sol_q_dot[:, :-1], qddot=sol_q_ddot)['ee_acc_linear']

    # 5. 逐节点打印所有物理量
    print("\n--- Per-Node Foot Kinematics ---")
    print("-" * 105)
    print("Node | Time (s) |       p (Position)       |       v (Velocity)       |       a (Acceleration)")
    print("-" * 105)

    for i in range(n_nodes + 1):
        p_i = np.array(p_trajectory[:, i]).flatten()
        v_i = np.array(v_trajectory[:, i]).flatten()
        
        if i < n_nodes:
            a_i = np.array(a_trajectory[:, i]).flatten()
            a_str = f"[{a_i[0]:8.4f}, {a_i[1]:8.4f}, {a_i[2]:8.4f}]"
        else:
            a_str = "          N/A           "

        print(f"{i:4d} |   {cumulative_dt[i]:.2f}   | "
                f"[{p_i[0]:8.4f}, {p_i[1]:8.4f}, {p_i[2]:8.4f}] | "
                f"[{v_i[0]:8.4f}, {v_i[1]:8.4f}, {v_i[2]:8.4f}] | "
                f"{a_str}")

    print("-" * 105)
    print("\n")

    # # ===========================================
    # # 在不升级库的情况下，通过数值差分计算质心速度和加速度(可行方法)
    # # ===========================================
    # print("\n===================================")
    # print("Calculating CoM vel/acc via Numerical Differentiation")
    # print("===================================")

    # # 1. 获取优化结果
    # sol_q = solution['q']
    # sol_dt = solv.getDt() # 获取每个时间步的实际长度

    # # 2. 获取最基本的质心位置函数
    # # 这个函数在旧版本中一定存在
    # com_pos_fun = cs.Function.deserialize(kindyn.centerOfMass())

    # # 3. 计算整个轨迹的质心位置
    # # com_pos_trajectory 的维度是 (3, n_nodes + 1)
    # com_pos_trajectory = com_pos_fun(q=sol_q)['com']

    # # 4. 使用一阶向前差分计算质心速度
    # # 速度向量的长度会比位置向量少一个
    # # v_com_trajectory 的维度是 (3, n_nodes)
    # v_com_trajectory = np.zeros((3, n_nodes))
    # for i in range(n_nodes):
    #     # v_i = (p_{i+1} - p_i) / dt_i
    #     delta_p = com_pos_trajectory[:, i+1] - com_pos_trajectory[:, i]
    #     v_com_trajectory[:, i] = np.array(delta_p).flatten() / sol_dt[i]

    # # 5. 使用一阶向前差分计算质心加速度
    # # 加速度向量的长度会比速度向量少一个
    # # a_com_trajectory 的维度是 (3, n_nodes - 1)
    # a_com_trajectory = np.zeros((3, n_nodes - 1))
    # for i in range(n_nodes - 1):
    #     # a_i = (v_{i+1} - v_i) / dt_i
    #     delta_v = v_com_trajectory[:, i+1] - v_com_trajectory[:, i]
    #     # 注意：这里我们用哪个dt？可以用dt_i, dt_{i+1}或平均值。用dt_i比较简单。
    #     a_com_trajectory[:, i] = np.array(delta_v).flatten() / sol_dt[i]

    # # =========================================================
    # # 6. 打印整个过程的质心加速度，并对不同阶段进行标注
    # # =========================================================
    # print("\n--- Full Center of Mass Acceleration Trajectory ---")
    # gravity = 9.81

    # # a_com_trajectory 的长度是 n_nodes - 1，对应于时间间隔 0->1, 1->2, ..., 48->49
    # # 它的索引范围是 0 到 48 (n_nodes - 2)
    # for i in range(n_nodes - 1):
        
    #     # 获取 x, y, z 三个方向的加速度
    #     acc_x = a_com_trajectory[0, i]
    #     acc_y = a_com_trajectory[1, i]
    #     acc_z = a_com_trajectory[2, i]
        
    #     # 判断当前时间间隔所处的阶段
    #     phase = ""
    #     # 节点 i+1 的时间点
    #     current_node = i + 1 
    #     if current_node < node_action[0]:
    #         phase = "Stance (Before Takeoff)"
    #     elif current_node >= node_action[0] and current_node < node_action[1]:
    #         phase = "Swing (In Air)"
    #     else:
    #         phase = "Stance (After Landing)"

    #     # 格式化输出
    #     print(f"Interval {i:2d}->{i+1:2d} | "
    #         f"Time: {cumulative_dt[i]:.2f}s | "
    #         f"Phase: {phase:<24} | "
    #         f"a_com = [ {acc_x:8.4f}, {acc_y:8.4f}, {acc_z:8.4f} ] m/s^2")

    # print("---------------------------------------------------\n")

    # # ===========================================
    # # 计算、打印并绘制质心加速度
    # # ===========================================
    # # 获取优化结果
    # sol_q = solution['q']
    # sol_q_dot = solution['q_dot']
    # sol_q_ddot = solution['q_ddot']

    # # 调用函数计算质心加速度
    # # 注意：q_ddot 在最后一个节点没有定义，所以我们只计算前 n_nodes 个点
    # com_acc_trajectory = com_acc_fun(q=sol_q[:, :-1], q_dot=sol_q_dot[:, :-1], q_ddot=sol_q_ddot)['a_com']

    # # 打印结果
    # print("\n===================================")
    # print("Center of Mass Acceleration (Z-axis)")
    # print("===================================")
    # # 我们来验证一下腾空阶段的加速度
    # print(f"Takeoff node: {node_action[0]}, Landing node: {node_action[1]}")
    # total_mass = kindyn.mass()
    # gravity = 9.81
    # expected_acc_z = -gravity

    # for i in range(node_action[0], node_action[1]):
    #     # 获取第i个节点的Z轴加速度
    #     acc_z = com_acc_trajectory[2, i]
    #     print(f"Node {i} (Time {cumulative_dt[i]:.2f}s): Calculated a_com_z = {acc_z:.4f} m/s^2. Expected: {expected_acc_z:.4f}")
    # print("===================================\n")

# ===========================================
# 计算并绘制关节轨迹和质心轨迹
# ===========================================
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # 添加 output_folder 定义
    output_folder = 'trajectory_data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# 使用完整的累积时间轴
    time_axis = cumulative_dt  # 形状: (n_nodes + 1,)

# 1. 计算关节轨迹
    joint_trajectory = solution['q']  # 形状: (n_q, n_nodes + 1)

# 2. 计算质心轨迹
    com_fn = cs.Function.deserialize(kindyn.centerOfMass())
    com_pos = com_fn(q=solution['q'])['com']  # 形状: (3, n_nodes + 1)

# 3. 创建绘图
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# 3.1 关节轨迹图
    ax_joints = plt.subplot(gs[0])
    for i in range(7, joint_trajectory.shape[0]):  # 跳过前7个基座状态
    # 确保数据点数量与时间轴匹配
        if joint_trajectory.shape[1] == len(time_axis):
            ax_joints.plot(time_axis, joint_trajectory[i, :], label=f'Joint {i-6}')
        else:
            ax_joints.plot(time_axis[:joint_trajectory.shape[1]], joint_trajectory[i, :], label=f'Joint {i-6}')
        
    ax_joints.set_title('Joint Angles Trajectory')
    ax_joints.set_xlabel('Time (s)')
    ax_joints.set_ylabel('Joint Angle (rad)')
    ax_joints.legend()
    ax_joints.grid(True)

# 3.2 质心轨迹图
    ax_com = plt.subplot(gs[1])
    dimensions = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    for i in range(3):
    # 确保数据点数量与时间轴匹配
        com_data = np.array(com_pos[i, :]).flatten()  # 关键修复：先转换为 NumPy 数组
    
    # 确保数据点数量与时间轴匹配
        if len(com_data) == len(time_axis):
            ax_com.plot(time_axis, com_data, 
                   color=colors[i], 
                   label=f'COM {dimensions[i]}')
        else:
            ax_com.plot(time_axis[:len(com_data)], com_data, 
                   color=colors[i], 
                   label=f'COM {dimensions[i]}')
        
    ax_com.set_title('Center of Mass Trajectory')
    ax_com.set_xlabel('Time (s)')
    ax_com.set_ylabel('Position (m)')
    ax_com.legend()
    ax_com.grid(True)

# 4. 添加动作关键点标记
    if action in ['jump_up', 'jump_forward', 'leap']:
    # 计算起跳和落地时间
        if isinstance(node_action, tuple):
            takeoff_time = cumulative_dt[node_action[0]]
            landing_time = cumulative_dt[node_action[1]]
        elif isinstance(node_action, list):
            takeoff_time = cumulative_dt[node_action[0][0]]
            landing_time = cumulative_dt[node_action[-1][1]]
    
    # 在两张图上都添加标记
        for ax in [ax_joints, ax_com]:
            ax.axvline(x=takeoff_time, color='m', linestyle='--', alpha=0.7, label='Takeoff')
            ax.axvline(x=landing_time, color='c', linestyle='--', alpha=0.7, label='Landing')
        ax_joints.legend()
        ax_com.legend()

# 5. 保存并显示图形
    plt.tight_layout()
    plot_path = os.path.join(output_folder, 'trajectory_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Trajectory plot saved to {plot_path}")
    plt.show()

# 6. 额外保存数据到CSV
# 6.1 保存关节轨迹
    joint_file_path = os.path.join(output_folder, 'joint_trajectory.csv')
    np.savetxt(joint_file_path, joint_trajectory[7:, :].T, delimiter=',')  # 跳过基座状态

# 6.2 保存质心轨迹
    com_file_path = os.path.join(output_folder, 'com_trajectory.csv')
    np.savetxt(com_file_path, com_pos.T, delimiter=',')
    # ========================================================
#     if plot_sol:
#         import matplotlib.pyplot as plt
#         from matplotlib import gridspec

#         hplt = plotter.PlotterHorizon(prb, solution)
#         # hplt.plotVariables(show_bounds=True, same_fig=True, legend=False)
#         hplt.plotVariables([elem.getName() for elem in f_list], show_bounds=True, gather=2, legend=False)
#         # hplt.plotFunctions(show_bounds=True, same_fig=True)
#         # hplt.plotFunction('inverse_dynamics', show_bounds=True, legend=True, dim=range(6))

#         pos_contact_list = list()
#         fig = plt.figure()
#         fig.suptitle('Contacts')
#         gs = gridspec.GridSpec(2, 2)
#         i = 0
#         for contact in contacts_name:
#             ax = fig.add_subplot(gs[i])
#             ax.set_title('{}'.format(contact))
#             i += 1
#             FK = cs.Function.deserialize(kindyn.fk(contact))
#             pos = FK(q=solution['q'])['ee_pos']
#             for dim in range(n_f):
#                 ax.plot(np.atleast_2d(cumulative_dt), np.array(pos[dim, :]), marker="x", markersize=3,
#                         linestyle='dotted')  # marker="x", markersize=3, linestyle='dotted'

#         plt.figure()
#         for contact in contacts_name:
#             FK = cs.Function.deserialize(kindyn.fk(contact))
#             pos = FK(q=solution['q'])['ee_pos']

#             plt.title(f'feet position - plane_xy')
#             plt.scatter(np.array(pos[0, :]), np.array(pos[1, :]), linewidth=0.1)

#         plt.figure()
#         for contact in contacts_name:
#             FK = cs.Function.deserialize(kindyn.fk(contact))
#             pos = FK(q=solution['q'])['ee_pos']

#             plt.title(f'feet position - plane_xz')
#             plt.scatter(np.array(pos[0, :]), np.array(pos[2, :]), linewidth=0.1)

#         plt.show()
#     # ======================================================
#     contact_map = {contacts_name[i]: solution[f_list[i].getName()] for i in range(n_c)}

#     # resampling
#     if resampling:

#         if isinstance(dt, cs.SX):
#             dt_before_res = solution['dt'].flatten()
#         else:
#             dt_before_res = dt

#         dt_res = 0.001
#         dae = {'x': x, 'p': q_ddot, 'ode': x_dot, 'quad': 1}
#         q_res, qdot_res, qddot_res, contact_map_res, tau_res = resampler_trajectory.resample_torques(
#             solution["q"], solution["q_dot"], solution["q_ddot"], dt_before_res, dt_res, dae, contact_map,
#             kindyn,
#             cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED)#添加代码，重现"q_ddot"

#     if rviz_replay:

#         try:
#             # set ROS stuff and launchfile  # remember to run a robot_state_publisher
#             import subprocess#导入 subprocess 模块，该模块允许你生成新的进程，连接到它们的输入/输出/错误管道，并获取它们的返回码。
#             os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples#将当前示例代码所在的路径添加到 ROS_PACKAGE_PATH 环境变量中，这样 ROS 就能找到相关的包
#             subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])#使用 subprocess.Popen 异步启动 roslaunch 命令，执行指定的启动文件 launcher.launch，并传递参数 robot:=spot。
#             rospy.loginfo("'spot' visualization started.")#使用 rospy 记录一条信息日志，表示 spot 机器人的可视化已经启动。
            
#         except:
#             print('Failed to automatically run RVIZ. Launch it manually.')

#         if resampling:
#             repl = replay_trajectory(dt_res, joint_names, q_res, contact_map_res,
#                                      cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
#         else:
#             # remember to run a robot_state_publisher
#             #“记得运行一个 robot_state_publisher”。robot_state_publisher 是 ROS 里的一个常用节点，它的功能是将机器人的关节状态信息发布出去，并且根据机器人的 URDF（Unified Robot Description Format）文件发布机器人的 TF（Transform）变换信息。在 RViz 里可视化机器人时，通常需要 robot_state_publisher 节点来提供机器人的位姿信息。
#             repl = replay_trajectory(dt, joint_names, solution['q'], contact_map,
#                                      cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)

#         repl.sleep(1.)
#         repl.replay(is_floating_base=True)

#     else:
#         print("To visualize the robot trajectory, start the script with the '--replay")


# if __name__ == '__main__':

#     spot_actions = ('wheelie', 'jump_up', 'jump_forward', 'jump_on_wall', 'leap', 'jump_twist')
#     spot_solvers = ('ipopt', 'ilqr', 'gnsqp')

#     parser = argparse.ArgumentParser(
#         description='spot motions: a set of motions performed by the BostonDynamics quadruped robot')
#     parser.add_argument('--action', '-a', help='choose which action spot will perform', choices=spot_actions,
#                         default=spot_actions[1])
#     parser.add_argument('--solver', '-s', help='choose which solver will be used', choices=spot_solvers,
#                         default=spot_solvers[0])
#     parser.add_argument('--replay', '-r', help='visualize the robot trajectory in rviz', action='store_true',
#                         default=False)
#     parser.add_argument("--codegen", '-c', type=str2bool, nargs='?', const=True, default=False,
#                         help="generate c++ code for faster solving")
#     parser.add_argument("--warmstart", '-w', type=str2bool, nargs='?', const=True, default=False,
#                         help="save solutions to mat file")
#     parser.add_argument("--plot", '-p', type=str2bool, nargs='?', const=True, default=True, help="plot solutions")

#     args = parser.parse_args()
#     main(args)
    