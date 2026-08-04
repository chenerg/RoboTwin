import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def print_episode_failure(episode_idx, seed, reason, error=None):
    print(" -------------")
    print(f"[{reason}] simulate data episode {episode_idx} fail! (seed = {seed})")
    if error is not None:
        print(f"Error ({type(error).__name__}): {error}")
    print(" -------------")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []
    state_path = os.path.join(args["save_path"], "collection_state.json")
    failure_manifest_path = os.path.join(args["save_path"], "fail", "episodes.json")

    def load_json(path, default):
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def save_json(path, value):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=4)
        os.replace(temporary_path, path)

    def save_seed_list():
        with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
            for seed in seed_list:
                file.write(f"{seed} ")

    failure_records = load_json(failure_manifest_path, [])
    collection_state = load_json(state_path, {})

    def save_failed_trajectory(seed, reason):
        failure_idx = len(failure_records)
        failure_record = {
            "episode_idx": failure_idx,
            "seed": seed,
            "reason": reason,
        }
        TASK_ENV.save_traj_data(
            failure_idx,
            category="fail",
            metadata={"success": False, **failure_record},
        )
        failure_records.append(failure_record)
        save_json(failure_manifest_path, failure_records)

    def close_after_failure():
        try:
            TASK_ENV.close_env()
        except Exception as close_error:
            print(f"Could not close failed simulation cleanly: {close_error}")
        if args["render_freq"]:
            try:
                TASK_ENV.viewer.close()
            except Exception as viewer_error:
                print(f"Could not close viewer cleanly: {viewer_error}")

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
        recorded_seeds = seed_list + [record["seed"] for record in failure_records]
        if recorded_seeds:
            epid = max(epid, max(recorded_seeds) + 1)
        epid = max(epid, collection_state.get("next_seed", 0))
        fail_num = max(collection_state.get("failure_count", 0), len(failure_records))
        print(f"Resume seed search from seed {epid}: success={suc_num}, fail={fail_num}")

        while suc_num < args["episode_num"] and fail_num <= args["episode_num"]:
            stage = "setup"
            failure_counted = False
            trajectory_saved = False
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)

                stage = "play_once"
                TASK_ENV.play_once()

                if not TASK_ENV.plan_success:
                    print_episode_failure(suc_num, epid, "PLAN_FAIL")
                    save_failed_trajectory(epid, "PLAN_FAIL")
                    trajectory_saved = True
                    fail_num += 1
                    failure_counted = True
                else:
                    stage = "check_success"
                    task_success = TASK_ENV.check_success()
                    if not task_success:
                        print_episode_failure(suc_num, epid, "TASK_SUCCESS_CHECK_FAIL")
                        save_failed_trajectory(epid, "TASK_SUCCESS_CHECK_FAIL")
                        trajectory_saved = True
                        fail_num += 1
                        failure_counted = True
                    else:
                        stage = "save_trajectory"
                        TASK_ENV.save_traj_data(
                            suc_num,
                            metadata={"success": True, "episode_idx": suc_num, "seed": epid},
                        )
                        print(f"simulate data episode {suc_num} success! (seed = {epid})")
                        seed_list.append(epid)
                        suc_num += 1

                stage = "close_env"
                TASK_ENV.close_env()

                if args["render_freq"]:
                    stage = "close_viewer"
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print_episode_failure(suc_num, epid, "SETUP_STABILITY_FAIL", e)
                if not failure_counted:
                    fail_num += 1
                close_after_failure()
                time.sleep(0.3)
            except Exception as e:
                reason = f"{stage.upper()}_EXCEPTION"
                print_episode_failure(suc_num, epid, reason, e)
                traceback.print_exc()
                if stage not in ("setup", "close_env", "close_viewer") and not trajectory_saved:
                    try:
                        save_failed_trajectory(epid, reason)
                    except Exception as save_error:
                        print(f"Could not preserve failed trajectory: {save_error}")
                if not failure_counted:
                    fail_num += 1
                close_after_failure()
                time.sleep(1)

            epid += 1
            save_seed_list()
            save_json(
                state_path,
                {
                    "next_seed": epid,
                    "success_count": suc_num,
                    "failure_count": fail_num,
                },
            )

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
        if suc_num < args["episode_num"]:
            print(
                f"Failure count {fail_num} exceeded target {args['episode_num']}; "
                "skip the remaining seed search and render collected trajectories."
            )
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        def first_incomplete_episode(category):
            output_dir = args["save_path"] if category == "success" else os.path.join(args["save_path"], category)
            idx = 0
            hdf5_path = os.path.join(output_dir, "data", f"episode{idx}.hdf5")
            video_path = os.path.join(output_dir, "video", f"episode{idx}.mp4")
            while os.path.exists(hdf5_path) and os.path.exists(video_path):
                idx += 1
                hdf5_path = os.path.join(output_dir, "data", f"episode{idx}.hdf5")
                video_path = os.path.join(output_dir, "video", f"episode{idx}.mp4")
            return idx

        def save_scene_info(category, episode_idx, info, metadata):
            output_dir = args["save_path"] if category == "success" else os.path.join(args["save_path"], category)
            info_file_path = os.path.join(output_dir, "scene_info.json")
            info_db = load_json(info_file_path, {})
            if category == "success":
                info_db[f"episode_{episode_idx}"] = info
            else:
                info_db[f"episode_{episode_idx}"] = {"scene_info": info, **metadata}
            save_json(info_file_path, info_db)

        def replay_episode(episode_idx, episode_seed, category, failure_reason=None):
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=episode_seed, **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx, category=category)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            replay_error = None
            try:
                info = TASK_ENV.play_once()
            except Exception as error:
                if category == "success":
                    raise
                info = {}
                replay_error = f"{type(error).__name__}: {error}"
                print_episode_failure(episode_idx, episode_seed, "FAIL_PLAY_ONCE_EXCEPTION", error)
                traceback.print_exc()

            replay_plan_success = bool(TASK_ENV.plan_success)
            replay_task_success = False
            if replay_plan_success and replay_error is None:
                try:
                    replay_task_success = bool(TASK_ENV.check_success())
                except Exception as error:
                    if category == "success":
                        raise
                    replay_error = f"{type(error).__name__}: {error}"
                    print_episode_failure(episode_idx, episode_seed, "FAIL_SUCCESS_CHECK_EXCEPTION", error)
                    traceback.print_exc()

            if category == "success" and not replay_plan_success:
                raise RuntimeError("Stored successful trajectory replay reported plan_success=False")
            if category == "success" and not replay_task_success:
                raise AssertionError("Stored successful trajectory failed its success check")

            # An immediate planning failure may not execute an action and therefore
            # may not have written any frames. Preserve at least its initial/final state.
            if category == "fail" and TASK_ENV.FRAME_IDX == 0:
                TASK_ENV._take_picture()

            metadata = {
                "success": category == "success",
                "seed": episode_seed,
                "replay_success": replay_task_success,
            }
            if failure_reason is not None:
                metadata["failure_reason"] = failure_reason
            replay_failure_reason = getattr(TASK_ENV, "replay_failure_reason", None)
            if replay_failure_reason is not None:
                metadata["replay_failure_reason"] = replay_failure_reason
            if replay_error is not None:
                metadata["replay_error"] = replay_error

            save_scene_info(category, episode_idx, info, metadata)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))

            TASK_ENV.merge_pkl_to_hdf5_video(category=category, metadata=metadata)

            TASK_ENV.remove_data_cache()
            return metadata

        success_start_idx = first_incomplete_episode("success")
        for episode_idx in range(success_start_idx, len(seed_list)):
            print(f"\033[34mTask name: {args['task_name']} [success]\033[0m")
            episode_seed = seed_list[episode_idx]
            try:
                replay_episode(episode_idx, episode_seed, "success")
            except UnStableError as e:
                print_episode_failure(episode_idx, episode_seed, "SETUP_STABILITY_FAIL", e)
                raise
            except Exception as e:
                reason = "SUCCESS_REPLAY_EXCEPTION"
                print_episode_failure(episode_idx, episode_seed, reason, e)
                traceback.print_exc()
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                try:
                    TASK_ENV.remove_data_cache()
                except Exception:
                    pass
                raise

        failure_start_idx = first_incomplete_episode("fail")
        for failure_record in failure_records[failure_start_idx:]:
            episode_idx = failure_record["episode_idx"]
            episode_seed = failure_record["seed"]
            failure_reason = failure_record["reason"]
            print(f"\033[34mTask name: {args['task_name']} [fail: {failure_reason}]\033[0m")
            try:
                replay_metadata = replay_episode(
                    episode_idx,
                    episode_seed,
                    "fail",
                    failure_reason=failure_reason,
                )
                failure_record["replay_success"] = replay_metadata["replay_success"]
                if "replay_error" in replay_metadata:
                    failure_record["replay_error"] = replay_metadata["replay_error"]
                else:
                    failure_record.pop("replay_error", None)
            except Exception as e:
                failure_record["replay_error"] = f"{type(e).__name__}: {e}"
                print_episode_failure(episode_idx, episode_seed, "FAIL_REPLAY_EXCEPTION", e)
                traceback.print_exc()
                try:
                    TASK_ENV.close_env()
                except Exception:
                    pass
                try:
                    TASK_ENV.remove_data_cache()
                except Exception:
                    pass
            save_json(failure_manifest_path, failure_records)

        if seed_list:
            command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
            os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
