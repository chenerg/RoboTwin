import mplib.planner
import mplib
import numpy as np
import pdb
import traceback
import numpy as np
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import transforms3d as t3d
import envs._GLOBAL_CONFIGS as CONFIGS


try:
    # ********************** CuroboPlanner (optional) **********************
    from curobo.types.math import Pose as CuroboPose
    import time
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.util import logger
    import torch
    import yaml
    from curobo.util import logger
    logger.setup_logger(level="error", logger_name="curobo")

    class CuroboPlanner:

        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
        ):
            super().__init__()
            ta.setup_logging("CRITICAL")  # hide logging
            logger.setup_logger(level="error", logger_name="'curobo")

            if yml_path != None:
                self.yml_path = yml_path
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints

            # translate from baselink to arm's base
            with open(self.yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data["planner"]["frame_bias"]

            # motion generation
            if True:
                world_config = {
                    "cuboid": {
                        "table": {
                            "dims": [0.7, 2, 0.04],  # x, y, z
                            "pose": [
                                self.robot_origion_pose.p[1],
                                0.0,
                                0.74 - self.robot_origion_pose.p[2],
                                1,
                                0,
                                0,
                                0.0,
                            ],  # x, y, z, qw, qx, qy, qz
                        },
                    }
                }
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
            )

            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen.warmup()
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
                num_graph_seeds=1,
            )
            self.motion_gen_batch = MotionGen(motion_gen_config)
            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)

        @staticmethod
        def _to_python_value(value):
            """Convert CuRobo enums and CUDA tensors into printable values."""
            if value is None:
                return None
            if hasattr(value, "value"):
                return value.value
            if torch.is_tensor(value):
                value = value.detach().cpu()
                return value.item() if value.numel() == 1 else value.tolist()
            return value

        @staticmethod
        def _status_name(value):
            """Keep the enum member name; ``Enum.value`` is often too generic."""
            if value is None:
                return None
            name = getattr(value, "name", None)
            return name if name is not None else str(value)

        @staticmethod
        def _as_numpy(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            try:
                return np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return None

        @classmethod
        def _minimum_finite_value(cls, value):
            array = cls._as_numpy(value)
            if array is None:
                return None
            finite = array[np.isfinite(array)]
            return None if finite.size == 0 else float(np.min(finite))

        def _joint_limit_diagnostics(self, joint_positions, near_limit=0.05):
            """Return joint-limit margins using the bounds used by CuRobo."""
            positions = self._as_numpy(joint_positions)
            if positions is None:
                return {"available": False, "reason": "joint positions unavailable"}
            positions = positions.reshape(-1)

            bounds_source = None
            for candidate in (
                getattr(getattr(self.motion_gen.ik_solver, "solver", None), "safety_rollout", None),
                getattr(self.motion_gen, "rollout_fn", None),
            ):
                if candidate is None:
                    continue
                lower = self._as_numpy(getattr(candidate, "action_bound_lows", None))
                upper = self._as_numpy(getattr(candidate, "action_bound_highs", None))
                if lower is None or upper is None:
                    continue
                lower = lower.reshape(-1)
                upper = upper.reshape(-1)
                if lower.size == positions.size and upper.size == positions.size:
                    bounds_source = candidate
                    break

            if bounds_source is None:
                return {"available": False, "reason": "CuRobo joint bounds unavailable"}

            lower_margin = positions - lower
            upper_margin = upper - positions
            margins = np.minimum(lower_margin, upper_margin)
            names = list(self.active_joints_name)
            if len(names) != positions.size:
                names = [f"joint_{index}" for index in range(positions.size)]

            violations = []
            near_limits = []
            for index, margin in enumerate(margins):
                item = {
                    "joint": names[index],
                    "position": float(positions[index]),
                    "lower": float(lower[index]),
                    "upper": float(upper[index]),
                    "margin": float(margin),
                }
                if margin < 0:
                    violations.append(item)
                elif margin <= near_limit:
                    near_limits.append(item)

            return {
                "available": True,
                "within_limits": len(violations) == 0,
                "minimum_margin": float(np.min(margins)),
                "near_limit_threshold": float(near_limit),
                "violations": violations,
                "near_limits": near_limits,
            }

        def _constraint_diagnostics(self, joint_state):
            """Ask CuRobo to separate joint, world and self-collision constraints."""
            try:
                valid, status = self.motion_gen.check_start_state(joint_state)
                return {
                    "available": True,
                    "valid": bool(valid),
                    "status_name": self._status_name(status),
                    "status": self._to_python_value(status),
                }
            except Exception as exc:
                # Diagnostics must never turn a normal planning failure into a crash.
                return {
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }

        @staticmethod
        def _category_from_constraint_status(status_name):
            mapping = {
                "INVALID_START_STATE_JOINT_LIMITS": "joint_limit",
                "INVALID_START_STATE_SELF_COLLISION": "self_collision",
                # The only world obstacle configured by this planner is named "table".
                "INVALID_START_STATE_WORLD_COLLISION": "table_collision",
            }
            return mapping.get(status_name)

        def _ik_failure_diagnostics(self, goal_pose, start_joint_state):
            """Inspect the best failed IK candidate without changing planning semantics."""
            diagnostics = {
                "available": False,
                "note": "target-candidate checks are diagnostic evidence, not a proof that every IK seed failed for the same reason",
            }
            try:
                ik_result = self.motion_gen.ik_solver.solve_single(
                    goal_pose,
                    retract_config=start_joint_state.position,
                    return_seeds=1,
                )
                position_error = self._minimum_finite_value(ik_result.position_error)
                rotation_error = self._minimum_finite_value(ik_result.rotation_error)
                position_threshold = float(self.motion_gen.ik_solver.position_threshold)
                rotation_threshold = float(self.motion_gen.ik_solver.rotation_threshold)
                diagnostics.update(
                    {
                        "available": True,
                        "success": bool(torch.any(ik_result.success).item()),
                        "position_error": position_error,
                        "position_threshold": position_threshold,
                        "position_error_above_threshold": (
                            position_error is not None and position_error > position_threshold
                        ),
                        "rotation_error": rotation_error,
                        "rotation_threshold": rotation_threshold,
                        "rotation_error_above_threshold": (
                            rotation_error is not None and rotation_error > rotation_threshold
                        ),
                    }
                )

                solutions = ik_result.solution
                if solutions is not None and solutions.numel() > 0:
                    candidates = solutions.reshape(-1, solutions.shape[-1])
                    errors = self._as_numpy(getattr(ik_result, "error", None))
                    candidate_index = 0
                    if errors is not None and errors.size == candidates.shape[0]:
                        candidate_index = int(np.nanargmin(errors.reshape(-1)))
                    candidate = candidates[candidate_index].reshape(1, -1)
                    candidate_state = JointState.from_position(
                        candidate,
                        joint_names=self.active_joints_name,
                    )
                    diagnostics["candidate_joint_positions"] = self._to_python_value(candidate.reshape(-1))
                    diagnostics["candidate_joint_limits"] = self._joint_limit_diagnostics(candidate)
                    diagnostics["candidate_constraints"] = self._constraint_diagnostics(candidate_state)
            except Exception as exc:
                diagnostics["reason"] = f"{type(exc).__name__}: {exc}"
            return diagnostics

        def _classify_planning_failure(self, result, goal_pose, start_joint_state):
            status_name = self._status_name(getattr(result, "status", None))
            start_constraints = self._constraint_diagnostics(start_joint_state)
            start_joint_limits = self._joint_limit_diagnostics(start_joint_state.position)
            evidence = {
                "start_constraints": start_constraints,
                "start_joint_limits": start_joint_limits,
            }

            category = self._category_from_constraint_status(start_constraints.get("status_name"))
            confidence = "exact" if category is not None else None
            if category is None:
                category = self._category_from_constraint_status(status_name)
                confidence = "exact" if category is not None else None

            ik_diagnostics = None
            if status_name in ("IK_FAIL", "IK Fail", "MotionGenStatus.IK_FAIL"):
                ik_diagnostics = self._ik_failure_diagnostics(goal_pose, start_joint_state)
                evidence["target_ik"] = ik_diagnostics
                candidate_status = (
                    ik_diagnostics.get("candidate_constraints", {}).get("status_name")
                    if ik_diagnostics is not None
                    else None
                )
                candidate_category = self._category_from_constraint_status(candidate_status)
                if category is None and candidate_category is not None:
                    category = candidate_category
                    confidence = "best_ik_candidate"
                if category is None and ik_diagnostics.get("available"):
                    if ik_diagnostics.get("success"):
                        category = "ik_sampling_instability"
                        confidence = "diagnostic_retry_succeeded"
                    else:
                        position_bad = ik_diagnostics.get("position_error_above_threshold")
                        rotation_bad = ik_diagnostics.get("rotation_error_above_threshold")
                        if position_bad and rotation_bad:
                            category = "position_and_rotation_error"
                        elif position_bad:
                            category = "position_error"
                        elif rotation_bad:
                            category = "rotation_error"
                        else:
                            category = "ik_no_feasible_seed"
                        confidence = "threshold_evidence"

            if category is None:
                category = {
                    "GRAPH_FAIL": "graph_search",
                    "Graph Fail": "graph_search",
                    "TRAJOPT_FAIL": "trajectory_optimization",
                    "TrajOpt Fail": "trajectory_optimization",
                    "FINETUNE_TRAJOPT_FAIL": "finetune_trajectory_optimization",
                    "Finetune TrajOpt Fail": "finetune_trajectory_optimization",
                }.get(status_name, "unknown")
                confidence = "exact" if category != "unknown" else "unavailable"

            return category, confidence, evidence

        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            arms_tag=None,
        ):  
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
            target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
            if not ("aloha-agilex" in self.yml_path):
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else: # patch for aloha-agilex
                T_target = t3d.affines.compose(target_pose_p, t3d.quaternions.quat2mat(target_pose_q), [1, 1, 1])
                T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target
                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            goal_pose_of_ee = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)

            # output
            res_result = dict()
            if result.success.item() == False:
                res_result["status"] = "Fail"
                failure_category, failure_confidence, failure_evidence = self._classify_planning_failure(
                    result,
                    goal_pose_of_ee,
                    start_joint_states,
                )
                diagnostic_fields = (
                    "status",
                    "valid_query",
                    "attempts",
                    "trajopt_attempts",
                    "used_graph",
                    "solve_time",
                    "ik_time",
                    "graph_time",
                    "trajopt_time",
                    "finetune_time",
                    "total_time",
                    "position_error",
                    "rotation_error",
                    "cspace_error",
                )
                diagnostics = {
                    field: self._to_python_value(getattr(result, field, None))
                    for field in diagnostic_fields
                }
                diagnostics.update(
                    {
                        "arm": arms_tag,
                        "status_name": self._status_name(getattr(result, "status", None)),
                        "failure_category": failure_category,
                        "failure_confidence": failure_confidence,
                        "failure_evidence": failure_evidence,
                        "start_joint_positions": joint_angles,
                        "target_pose_in_base_frame": (
                            list(target_pose_p) + list(target_pose_q)
                        ),
                    }
                )
                res_result["diagnostics"] = diagnostics
                print(f"[CuroboPlanner] Planning failed: {diagnostics}")
                return res_result
            else:
                res_result["status"] = "Success"
                res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
                res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
                return res_result

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position
            """

            num_poses = len(target_gripper_pose_list)
            # transformation from world to arm's base
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            poses_list = []
            for target_gripper_pose in target_gripper_pose_list:
                world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
                base_target_pose_p, base_target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)

                if not ("aloha-agilex" in self.yml_path):
                    base_target_pose_p[0] += self.frame_bias[0]
                    base_target_pose_p[1] += self.frame_bias[1]
                    base_target_pose_p[2] += self.frame_bias[2]
                else: # patch for aloha-agilex
                    T_target = t3d.affines.compose(base_target_pose_p, t3d.quaternions.quat2mat(base_target_pose_q), [1, 1, 1])
                    T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                    if arms_tag == "left":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                    elif arms_tag == "right":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                    else:
                        raise ValueError(f"Invalid arms_tag: {arms_tag}")

                    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                    T_new = T_rot @ T_bias @ T_target
                    base_target_pose_p = T_new[:3, 3]
                    base_target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

                base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
                poses_list.append(base_target_pose_list)

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            goal_pose_of_ee = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            joint_angles_cuda = (torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1))
            joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
            start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            try:
                result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)
            except Exception as e:
                return {"status": ["Failure" for i in range(10)]}

            # output
            res_result = dict()
            # Convert boolean success values to "Success"/"Failure" strings
            success_array = result.success.cpu().numpy()
            status_array = np.array(["Success" if s else "Failure" for s in success_array], dtype=object)
            res_result["status"] = status_array

            if np.all(res_result["status"] == "Failure"):
                return res_result

            res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
            res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
            return res_result

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def _trans_from_world_to_base(self, base_pose, target_pose):
            '''
                transform target pose from world frame to base frame
                base_pose: np.array([x, y, z, qw, qx, qy, qz])
                target_pose: np.array([x, y, z, qw, qx, qy, qz])
            '''
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q
    
except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')
    print('Exception traceback:')
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)

        return result

    def plan_grippers(self, now_val, target_val):
        num_step = 200  # TODO
        dis_val = target_val - now_val
        per_step = dis_val / num_step
        res = {}
        vals = np.linspace(now_val, target_val, num_step)
        res["num_step"] = num_step
        res["per_step"] = per_step  # dis per step
        res["result"] = vals
        return res
