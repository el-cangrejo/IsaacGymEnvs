import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

import matplotlib.pyplot as plt

class TaskGrasp(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.robot_position_noise = self.cfg["env"]["robotPositionNoise"]
        self.robot_rotation_noise = self.cfg["env"]["robotRotationNoise"]
        self.robot_dof_noise = self.cfg["env"]["robotDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_rot_scale": self.cfg["env"]["rotationRewardScale"],
            "r_fintip_scale": self.cfg["env"]["fingertipRewardScale"],
            "r_fintip_dist_scale": self.cfg["env"]["fingertipDistanceScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_lift_height_scale": self.cfg["env"]["liftHeightRewardScale"],
            "r_actions_reg_scale": self.cfg["env"]["actionsRegularizationRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs : 
        self.cfg["env"]["numObservations"] = 19 if self.control_type == "osc" else 15
        # actions : 
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}       # will be dict mapping names to relevant sim handles
        self.num_dofs = None    # Total number of DOFs per env
        self.actions = None     # Current actions to be deployed

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._robot_effort_limits = None        # Actuator effort limits for robot
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array
        self._task_conditional = None # One-hot encoded binary variable describing task intention : handoff (=1) or use (=2)

        self.up_axis = "z"

        super().__init__(config=self.cfg, rl_device=rl_device,
                sim_device=sim_device, graphics_device_id=graphics_device_id,
                headless=headless,
                virtual_screen_capture=virtual_screen_capture,
                force_render=force_render)

        # Kinova + Seed defaults
        robot_default_dof_pos = [0.19, 0.42, -1.39, 1.04, -1.00, 0.55, 0.00] + [0.0] * 17
        # robot_default_dof_pos = [-0.5, 1.2, 0.09, 1.14, -1.7, -1.5, -0.6] + [0.0] * 17
        robot_default_dof_pos[20] = 1.57
        self.robot_default_dof_pos = to_torch(robot_default_dof_pos, device=self.device)

        # OSC Gains
        self.kp = to_torch([200.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._robot_effort_limits[:7].unsqueeze(0)

        self.vis_reward = False 
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

        if self.vis_reward:
            self.fig, self.ax = plt.subplots()

            self.dist_rew_list = []
            self.fintip_rew_list = []
            self.lift_rew_list = []
            self.height_rew_list = []

        sphere_pose = gymapi.Transform()
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robot_asset_file = "urdf/seed_kinova_robot/seed_kinova.urdf"

        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self._robot_effort_limits = []

        for i in range(self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])
            self._robot_effort_limits.append(robot_dof_props['effort'][i])

        # arm properties
        robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
        # robot_dof_props["armature"][:7].fill(0.001)
        robot_dof_props["stiffness"][:7].fill(50.0)
        robot_dof_props["damping"][:7].fill(10.0)
        # hand properties
        robot_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        robot_dof_props["armature"][7:].fill(0.001)
        robot_dof_props["stiffness"][7:].fill(100.0)
        robot_dof_props["damping"][7:].fill(50.0)

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        self.robot_dof_lower_limits[:7] = -3.14
        self.robot_dof_upper_limits[:7] = 3.14
        self._robot_effort_limits = to_torch(self._robot_effort_limits, device=self.device)

        # Define start pose for robot
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Create table asset
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_opts.disable_gravity = True
        table_asset = self.gym.create_box(self.sim, *[0.8, 0.8, table_thickness], table_opts)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(.7, 0., 0.25)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        ## Load object asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False
        asset_options.thickness = 0.0001
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.flip_visual_attachments = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.convex_hull_downsampling = 30

        object_asset = self.gym.load_asset(self.sim, asset_root, "urdf/ycb/YcbHammer/model.urdf", asset_options)

        self.object_default_state = torch.tensor([0.7, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.4, 0.0, 0.01)
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        max_agg_bodies = num_robot_bodies + num_table_bodies + num_object_bodies
        max_agg_shapes = num_robot_shapes + num_table_shapes + num_object_shapes

        self.robots = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create robot actor and set properties
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)

            # Create object actor
            self._object_id = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 2, 0)

            self.gym.end_aggregate(env_ptr)
            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)

        self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        robot_handle = 0

        self.handles = {
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "grasp_link"),
            "fftip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "fftip"),
            "thtip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "thtip"),
            "object_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self._object_id, "object"),
            "handoff_link" : self.gym.find_actor_rigid_body_handle(env_ptr, self._object_id, "handoff_p"),
            "use_link" : self.gym.find_actor_rigid_body_handle(env_ptr, self._object_id, "use_p"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._fftip_state = self._rigid_body_state[:, self.handles["fftip"], :]
        self._thtip_state = self._rigid_body_state[:, self.handles["thtip"], :]
        self._object_state = self._root_state[:, self._object_id, :]
        self._handoff_link_state = self._rigid_body_state[:, self.handles["handoff_link"], :]
        self._use_link_state = self._rigid_body_state[:, self.handles["use_link"], :]

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, robot_handle)['grasp_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._hand_control = self._pos_control[:, 7:]

        # Initialize task intentions
        self._task_conditional = torch.nn.functional.one_hot(torch.randint(1, 3, (self.num_envs,)) % 2).to(self.device)
        # self._task_conditional = torch.nn.functional.one_hot((torch.ones((self.num_envs,)).to(dtype=int)) % 2, num_classes=2).to(self.device)

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.down_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        self.grasp_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

    def _update_states(self):
        self.states.update({
            # Robot
            "q": self._q[:, :7],
            "q_hand": self._q[:, 7:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "fftip_pos" : self._fftip_state[:, :3],
            "thtip_pos" : self._thtip_state[:, :3],
            # Object
            "object_quat": self._object_state[:, 3:7],
            "object_pos": self._object_state[:, :3],
            "object_pos_relative": self._object_state[:, :3] - self._eef_state[:, :3],
            "object_fftip_pos_relative": self._object_state[:, :3] - self._fftip_state[:, :3],
            "object_thtip_pos_relative": self._object_state[:, :3] - self._thtip_state[:, :3],
            "handoff_pos": self._handoff_link_state[:, :3],
            "use_pos": self._use_link_state[:, :3],
            "task": self._task_conditional,
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], reward_dict = compute_robot_reward(
                self.reset_buf, self.progress_buf, self.actions, self.states,
                self.grasp_up_axis, self.down_axis, self.num_envs,
                self.reward_settings, self.max_episode_length)

        if self.vis_reward:
            self.ax.cla()

            self.dist_rew_list.append(reward_dict["Distance Reward"][0].cpu().detach())
            self.fintip_rew_list.append(reward_dict["Fingertips Reward"][0].cpu().detach())
            self.lift_rew_list.append(reward_dict["Lift Reward"][0].cpu().detach())
            self.height_rew_list.append(reward_dict["Lift Height Reward"][0].cpu().detach())

            self.ax.plot(self.dist_rew_list, c='r', label='Distance rew')
            self.ax.plot(self.fintip_rew_list, c='b', label='Fingertip rew')
            self.ax.plot(self.lift_rew_list, c='g', label='Lift rew')
            self.ax.plot(self.height_rew_list, c='y', label='Height rew')

            plt.legend()
            plt.draw()
            plt.pause(0.0001)

    def compute_observations(self):
        self._refresh()
        obs = ["object_pos", "object_quat", "object_pos_relative", "eef_pos", "eef_quat", "task"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        pos = tensor_clamp(self.robot_default_dof_pos.unsqueeze(0),
            self.robot_dof_lower_limits.unsqueeze(0), self.robot_dof_upper_limits)
        pos[:, 20] = 1.5
    
        # Reset object states by sampling random poses
        self._reset_init_object_state(env_ids=env_ids)
        self._object_state[env_ids] = self._init_object_state[env_ids]

        # # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Randomly choose task intention
        self._task_conditional[env_ids, :] = torch.nn.functional.one_hot(torch.randint(1, 3, (len(env_ids),)) % 2, num_classes=2).to(self.device)
        # print (self._task_conditional)
        # self._task_conditional = torch.nn.functional.one_hot((torch.zeros((len(env_ids),)).to(dtype=int)) % 2, num_classes=2).to(self.device)
        # self._task_conditional = torch.nn.functional.one_hot((torch.ones((len(env_ids),)).to(dtype=int)) % 2, num_classes=2).to(self.device)
        # print (self._task_conditional)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update object states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        if self.vis_reward:
            self.dist_rew_list = []
            self.fintip_rew_list = []
            self.lift_rew_list = []
            self.height_rew_list = []

    def _reset_init_object_state(self, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = self.object_default_state.repeat(num_resets, 1).to(device=self.device)
        sampled_cube_state[:, :2] = sampled_cube_state[:, :2] + \
                                    1.0 * self.start_position_noise * \
                                    (torch.rand(num_resets, 2, device=self.device) - 0.5)

        # # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        self._init_object_state[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        # j_eef_inv = m_eef @ self._j_eef @ mm_inv
        # u_null = self.kd_null * -qd + self.kp_null * (
        #         (self.robot_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        # u_null[:, 7:] *= 0
        # u_null = self._mm @ u_null.unsqueeze(-1)
        # u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._robot_effort_limits[:7].unsqueeze(0), self._robot_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_hand = self.actions[:, :-1], self.actions[:, -1:]
        
        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)

        self._arm_control[:, :] = u_arm 
        self._hand_control[:, :] = u_hand
        self._hand_control[:, 13] = 1.57

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # if self.viewer:
        #     self.gym.clear_lines(self.viewer)
        #     for i in range(self.num_envs):
        #         use_pos = self._use_link_state[i, :3]
        #         sphere_pos = gymapi.Vec3(use_pos[0], use_pos[1], use_pos[2])
        #         sphere_pose = gymapi.Transform(p=sphere_pos)
        #         gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

        #         handoff_pos = self._handoff_link_state[i, :3]
        #         sphere_pos = gymapi.Vec3(handoff_pos[0], handoff_pos[1], handoff_pos[2])
        #         sphere_pose = gymapi.Transform(p=sphere_pos)
        #         gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_robot_reward(
    reset_buf, progress_buf, actions, states, grasp_up_axis, down_axis,
    num_envs, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor, int, Dict[str, float], float) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]

    # distance from grasp link to the object
    d_handoff = torch.norm(states["eef_pos"] - states["handoff_pos"], dim=-1)
    d_use = torch.norm(states["eef_pos"] - states["use_pos"], dim=-1)
    d_eef = torch.where(states["task"][:, 1] == 1, d_handoff, d_use)

    dist_reward = 1 - torch.tanh(2 * d_eef)

    # reward for index fingertip
    d_handoff_fftip = torch.norm(states["fftip_pos"] - states["handoff_pos"], dim=-1)
    d_use_fftip = torch.norm(states["fftip_pos"] - states["use_pos"], dim=-1)
    d_fftip = torch.where(states["task"][:, 1] == 1, d_handoff_fftip, d_use_fftip)

    # reward for thumb fingertip
    d_handoff_thtip = torch.norm(states["thtip_pos"] - states["handoff_pos"], dim=-1)
    d_use_thtip = torch.norm(states["thtip_pos"] - states["use_pos"], dim=-1)
    d_thtip = torch.where(states["task"][:, 1] == 1, d_handoff_thtip, d_use_thtip)

    fintip_reward = 1 - torch.tanh(reward_settings["r_fintip_dist_scale"] * (d_fftip + d_thtip) / 2.)

    # Use fingertip reward only when hand is close
    # ft_sched_val = (d_eef < 0.05) * 1.0
    # fintip_reward = torch.where(ft_sched_val == 1.0, fintip_reward, ft_sched_val)

    # grasp axis should look down
    grasp_axis = tf_vector(states["eef_quat"], grasp_up_axis)
    dot = torch.bmm(grasp_axis.view(num_envs, 1, 3), down_axis.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    rot_reward = torch.sign(dot) * dot ** 2

    # reward for lifting object
    object_height = states["object_pos"][:, 2]
    object_lifted = object_height > 0.45
    lift_reward = object_lifted
    lift_height = object_height - 0.3

    # Regularization on the actions
    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = reward_settings["r_dist_scale"] * dist_reward \
            + reward_settings["r_rot_scale"] * rot_reward \
            + reward_settings["r_fintip_scale"] * fintip_reward \
            + reward_settings["r_lift_scale"] * lift_reward \
            + reward_settings["r_lift_height_scale"] * lift_height \
            + reward_settings["r_actions_reg_scale"] * action_penalty

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    # Return rewards in dict for debugging
    reward_dict = {"Distance Reward": reward_settings["r_dist_scale"] * dist_reward,
                   "Rotation Reward": reward_settings["r_rot_scale"] * rot_reward,
                   "Fingertips Reward": reward_settings["r_fintip_scale"] * fintip_reward,
                   "Lift Reward": reward_settings["r_lift_scale"] * lift_reward,
                   "Lift Height Reward": reward_settings["r_lift_height_scale"] * lift_height,
                   "Action Regularization Reward": reward_settings["r_actions_reg_scale"] * action_penalty,}

    return rewards, reset_buf, reward_dict

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

@torch.jit.script
def random_pos(num: int, device: str) -> torch.Tensor:
    radius = 0.8
    height = 0.03
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    z = torch.tensor([height], device=device).repeat((num, 1))

    return torch.cat((x[:, None], y[:, None], z), dim=-1)

@torch.jit.script
def remap(x: torch.Tensor, l1: float, h1: float, l2: float, h2: float) -> torch.Tensor:
    return l2 + (x - l1) * (h2 - l2) / (h1 - l1)
    
@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat
