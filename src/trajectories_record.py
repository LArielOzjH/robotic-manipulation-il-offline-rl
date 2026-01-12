import argparse
import time
import os
from glob import glob
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper
from robosuite.wrappers import VisualizationWrapper

from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import numpy as np

class PandaPickPlace(PickPlace):
    
    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # can sample anywhere in bin
        bin_x_half = self.model.mujoco_arena.table_full_size[0] / 2 - 0.10
        bin_y_half = self.model.mujoco_arena.table_full_size[1] / 2 - 0.10

        # each object should just be sampled in the bounds of the bin (with some tolerance)
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="CollisionObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-bin_x_half, bin_x_half],
                y_range=[-bin_y_half, bin_y_half],
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=self.z_offset,
            )
        )

        # each visual object should just be at the center of each target bin
        index = 0
        for vis_obj in self.visual_objects:

            # get center of target bin
            bin_x_low = self.bin2_pos[0]
            bin_y_low = self.bin2_pos[1]
            if index == 0 or index == 2:
                bin_x_low -= self.bin_size[0] / 2
            if index < 2:
                bin_y_low -= self.bin_size[1] / 2
            bin_x_high = bin_x_low + self.bin_size[0] / 2
            bin_y_high = bin_y_low + self.bin_size[1] / 2
            bin_center = np.array(
                [
                    (bin_x_low + bin_x_high) / 2.0,
                    (bin_y_low + bin_y_high) / 2.0,
                ]
            )

            # placement is relative to object bin, so compute difference and send to placement initializer
            rel_center = bin_center - self.bin1_pos[:2]

            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{vis_obj.name}ObjectSampler",
                    mujoco_objects=vis_obj,
                    x_range=[rel_center[0], rel_center[0]],
                    y_range=[rel_center[1], rel_center[1]],
                    rotation=0.0,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.bin1_pos,
                    z_offset=self.bin2_pos[2] - self.bin1_pos[2],
                    # rng=self.rng,
                )
            )
            index += 1

class PandaPickPlaceBread(PandaPickPlace):
    """
    Easier version of task - place one bread into its bin.
    """

    def __init__(self, **kwargs):
        assert "single_object_mode" not in kwargs and "object_type" not in kwargs, "invalid set of arguments"
        super().__init__(single_object_mode=2, object_type="bread", **kwargs)



class MaxDataCollectionWrapper(DataCollectionWrapper):
    '''
    Declaration:
    Inherited from the DataCollectionWrapper class, I changed the "step" and 
    "_start_new_episode" (reset related) function to document more details while 
    collecting trajectories, Others methods keep.
    Details:
        1. gripper0_right_eef
        2. Bread_main
        3. gripper0_right_leftfinger 
        4. gripper0_right_finger_joint1_tip 
        5. gripper0_right_rightfinger 
        6. gripper0_right_finger_joint2_tip
    '''
    def __init__(self, env, directory, collect_freq=1, flush_freq=100):
        super().__init__(env, directory, collect_freq, flush_freq)
        
        self.obs = []
        self.reward = []
        
    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            obs=np.array(self.obs),
            reward=np.array(self.reward),
            action_infos=self.action_infos,
            successful=self.successful,
            env=env_name,
        )
        self.states = []
        self.action_infos = []
        self.obs = []
        self.reward = []
        self.successful = False
        
    def _check_success(self):
        """
        Check if all objects have been successfully placed in their corresponding bins.

        Returns:
            bool: True if all objects are placed correctly
        """
        # remember objects that are in the correct bins
        for i, obj in enumerate(self.objects):
            obj_str = obj.name
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj_str]]
            dist = min(
                [
                    np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - obj_pos)
                    for arm in self.robots[0].arms
                ]
            )
            r_reach = 1 - np.tanh(10.0 * dist)
            self.objects_in_bins[i] = int((not self.not_in_bin(obj_pos, i)) and r_reach < 0.6)

        # returns True if a single object is in the correct bin
        if self.single_object_mode in {1, 2}:
            return np.sum(self.objects_in_bins) > 0

        # returns True if all objects are in correct bins
        return np.sum(self.objects_in_bins) == len(self.objects)
    
    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        body_names = [
            "gripper0_right_eef",
            "Bread_main",
            "gripper0_right_leftfinger",
            # "gripper0_right_finger_joint1_tip",
            "gripper0_right_rightfinger",
            # "gripper0_right_finger_joint2_tip",
        ]
        ret = self.env.step(action)
        self.t += 1
        
        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            keep_keys = [
                        "robot0_joint_pos",     
                        "robot0_joint_vel",     
                        "robot0_eef_pos",       
                        "robot0_eef_quat",      
                        "robot0_gripper_qpos",   
                        "Bread_pos",             
                        "Bread_quat",            
                        "Bread_to_robot0_eef_pos" 
                    ]
            obs = ret[0]
            # print(obs["Bread_pos"])
            current_obs = np.concatenate([obs[k].flatten() for k in keep_keys])
            self.obs.append(current_obs)
            
            
            state = self.env.sim.get_state().flatten()
            sim = self.env.sim
            m = sim.model
            d = sim.data
            feat_list = []

            # 1) 每个 body: xpos (3) + xquat (4)
            for name in body_names:
                try:
                    bid = m.body_name2id(name)
                except Exception as e:
                    raise ValueError(f"Body name not found in MuJoCo model: {name}") from e

                xpos = np.asarray(d.body_xpos[bid], dtype=np.float32).reshape(-1)   # (3,)
                xquat = np.asarray(d.body_xquat[bid], dtype=np.float32).reshape(-1) # (4,)

                if xpos.shape[0] != 3:
                    raise ValueError(f"Unexpected xpos shape for {name}: {xpos.shape}")
                if xquat.shape[0] != 4:
                    raise ValueError(f"Unexpected xquat shape for {name}: {xquat.shape}")

                feat_list.append(xpos)
                feat_list.append(xquat)

            # 2) eef 相对 Bread 的向量: (Bread_pos - eef_pos) (3,)
            eef_id = m.body_name2id("gripper0_right_eef")
            bread_id = m.body_name2id("Bread_main")

            eef_pos = np.asarray(d.body_xpos[eef_id], dtype=np.float32).reshape(3)
            bread_pos = np.asarray(d.body_xpos[bread_id], dtype=np.float32).reshape(3)

            rel_eef_to_bread = bread_pos - eef_pos  # (3,)
            feat_list.append(rel_eef_to_bread)
            
            goal_center = np.asarray([0.1, 0.28, 0.8], dtype=np.float32).reshape(3)
            feat_list.append(goal_center)                 # (3,)
            feat_list.append(goal_center - bread_pos)     # (3,)

            # 3) 拼起来并 append
            extra_feat = np.concatenate(feat_list, axis=0)     # (52,)
            state_aug = np.concatenate([state, extra_feat], axis=0)
            self.states.append(state_aug)

            reward = ret[1]
            self.reward.append(reward)
            
            info = {}
            info["actions"] = np.array(action)

            # (if applicable) store absolute actions
            step_info = ret[3]
            if "action_abs" in step_info.keys():
                info["actions_abs"] = np.array(step_info["action_abs"])

            self.action_infos.append(info)

        # check if the demonstration is successful
        if self.env._check_success():
            self.successful = True

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret
        
    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """
        body_names = [
            "gripper0_right_eef",
            "Bread_main",
            # "robot0_right_hand",
            "gripper0_right_leftfinger",
            # "gripper0_right_finger_joint1_tip",
            "gripper0_right_rightfinger",
            # "gripper0_right_finger_joint2_tip",
        ]

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)

        # NOTE: was previously self.env.model.get_xml(). Was causing the following issue in rare cases:
        # ValueError: Error: eigenvalues of mesh inertia violate A + B >= C
        # switching to self.env.sim.model.get_xml() does not create this issue
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        
        # Change the _current_task_instance_state
        state = self.env.sim.get_state().flatten()

        sim = self.env.sim
        m = sim.model
        d = sim.data
        feat_list = []

        # 1) 每个 body: xpos (3) + xquat (4)
        for name in body_names:
            try:
                bid = m.body_name2id(name)
            except Exception as e:
                raise ValueError(f"Body name not found in MuJoCo model: {name}") from e

            xpos = np.asarray(d.body_xpos[bid], dtype=np.float32).reshape(-1)   # (3,)
            xquat = np.asarray(d.body_xquat[bid], dtype=np.float32).reshape(-1) # (4,)

            if xpos.shape[0] != 3:
                raise ValueError(f"Unexpected xpos shape for {name}: {xpos.shape}")
            if xquat.shape[0] != 4:
                raise ValueError(f"Unexpected xquat shape for {name}: {xquat.shape}")

            feat_list.append(xpos)
            feat_list.append(xquat)

        # 2) eef 相对 Bread 的向量: (Bread_pos - eef_pos) (3,)
        eef_id = m.body_name2id("gripper0_right_eef")
        bread_id = m.body_name2id("Bread_main")

        eef_pos = np.asarray(d.body_xpos[eef_id], dtype=np.float32).reshape(3)
        bread_pos = np.asarray(d.body_xpos[bread_id], dtype=np.float32).reshape(3)

        rel_eef_to_bread = bread_pos - eef_pos  # (3,)
        feat_list.append(rel_eef_to_bread)

        goal_center = np.asarray([0.1, 0.28, 0.8], dtype=np.float32).reshape(3)
        feat_list.append(goal_center)                 # (3,)
        feat_list.append(goal_center - bread_pos)     # (3,)

        # 3) 拼起来并 append
        extra_feat = np.concatenate(feat_list, axis=0)     # (52,)
        state_aug = np.concatenate([state, extra_feat], axis=0)
        
        self._current_task_instance_state = np.array(state_aug)

        # trick for ensuring that we can play MuJoCo demonstrations back
        # deterministically by using the recorded actions open loop
        self.env.reset_from_xml_string(self._current_task_instance_xml)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(self._current_task_instance_state)
        self.env.sim.forward()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift") 
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Used Robot") 
    parser.add_argument(
        "--config", type=str, default="default", help="Environment Configuration (if needed)"
    ) 
    parser.add_argument("--arm", type=str, default="right", help="Controlled Arm (e.g. 'right' or 'left')")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch Gripper Control on Grasp")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch Camera Angle on Grasp")
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller Selection. Can be a generic name (e.g. 'BASIC' or 'WHOLE_BODY_MINK_IK') or a json file (see robosuite/controllers/config example) or None (use robot's default controller if exists)",
    ) 
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=0.1, help="Position Input Scaling")
    parser.add_argument("--rot-sensitivity", type=float, default=0.1, help="Rotation Input Scaling")
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Pause when simulation speed exceeds specified frame rate; 20 fps is real-time.",
    )

    args = parser.parse_args()

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
        z_rotation=0
    )

    env = VisualizationWrapper(env, indicator_configs=None)
    env = MaxDataCollectionWrapper(env, "../PoS0_05RL", flush_freq=1200)

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "mjgui":
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device selection: please select 'keyboard' or 'spacemouse'.")



    obs = env.reset()
    # print(env.action_dim)
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    env.render()

    last_grasp = 0

    device.start_control()
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Main loop
    for t in range(1000000):
        start = time.time()

        active_robot = env.robots[device.active_robot]


        input_ac_dict = device.input2action()


        if input_ac_dict is None:
            break

        from copy import deepcopy

        action_dict = deepcopy(input_ac_dict)  # {}

        for arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):  
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[arm].input_type
                # For Franka Emika this is always "delta"
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)
        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        env.step(env_action)
        env.render()

        if args.max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / args.max_fr - elapsed
            if diff > 0:
                time.sleep(diff)
                
        # if env._check_success():
        #     print("Task completed successfully")
        #     print(f"Total time: {t} steps")
        #     env._flush()
        #     break
