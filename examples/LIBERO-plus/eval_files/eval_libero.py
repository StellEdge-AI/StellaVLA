import dataclasses
import json
import logging
import math
import os
import pathlib

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


# ── Render the current end-effector state as prompt text, exactly as training
#    did. state[3:6] is axis-angle. ──
_GRIP_SEP_CLOSED = 0.005
_GRIP_SEP_OPEN = 0.08
def _gripper_open_pct(state_vec):
    if len(state_vec) < 8:
        return 0
    sep = float(state_vec[6]) - float(state_vec[7])
    frac = (sep - _GRIP_SEP_CLOSED) / (_GRIP_SEP_OPEN - _GRIP_SEP_CLOSED)
    return int(round(min(1.0, max(0.0, frac)) * 100))
def _describe_robot_state(state_vec):
    if state_vec is None or len(state_vec) < 6:
        return ""
    x, y, z = (int(round(float(state_vec[i]) * 1000)) for i in range(3))
    r, p, yw = (int(round(float(state_vec[i]) * 180.0 / math.pi)) for i in range(3, 6))
    g = _gripper_open_pct(state_vec)
    return (f"end-effector position x={x} y={y} z={z} mm; "
            f"orientation roll={r} pitch={p} yaw={yw} deg; gripper open {g}%")


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [256, 256]  # trained at 256; a 224 downscale would be lossy

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    log_path: str = "experiments/libero/logs"

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    # task slice [start_idx, end_idx) for parallel sharding across GPUs; end_idx=-1 => full suite
    start_idx: int = 0
    end_idx: int = -1
    # Explicit task ids (comma-separated, 0-indexed), for targeting a category or
    # a hand-picked subset instead of a contiguous shard. Overrides start/end.
    task_ids: str = ""


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    # task slice [start_idx, end_idx) for sharded parallel runs across GPUs
    _start = max(0, args.start_idx)
    _end = num_tasks_in_suite if args.end_idx < 0 else min(args.end_idx, num_tasks_in_suite)
    _explicit = [int(x) for x in str(args.task_ids).split(",") if x.strip() != ""]
    TASK_LIST = ([t for t in _explicit if 0 <= t < num_tasks_in_suite]
                 if _explicit else list(range(_start, _end)))

    disturb_res = {}
    LIBERO_HOME = os.environ.get('LIBERO_HOME', 'path_to_LIBERO-plus_home')
    with open(os.path.join(LIBERO_HOME,'libero/libero/benchmark/task_classification.json')) as f:
        TASK_MAPPING = json.load(f)[args.task_suite_name]
    ID2CATEGORY = {}
    for item in TASK_MAPPING:
        category = item["category"]
        item_name = item["name"]
        ID2CATEGORY[item['id']] = (category, item_name)
        if category not in disturb_res:
            disturb_res[category] = {"total_count": 0, "success_count": 0}
        # count total only for tasks in THIS shard (item id is 1-indexed = task_id+1)
        if (item['id'] - 1) in set(TASK_LIST):
            disturb_res[category]["total_count"] += 1

    # Start evaluation

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(TASK_LIST):
        
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0
            
            # full_actions = np.load("./debug/action.npy")
            
            while t < max_steps + args.num_steps_wait:
                
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = { # 
                    "observation.primary": np.expand_dims(
                        img, axis=0
                    ),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # The server injects the clean base-LIBERO demo for this task,
                # resolved from the instruction text plus (suite, task_id).
                # state_text and the 8-dim axis-angle state mirror the LIBERO
                # client.
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                    "state_text": _describe_robot_state(state),
                    "state": np.asarray(state, dtype=np.float32),
                    "suite": args.task_suite_name,
                    "task_id": int(task_id),
                    "episode_id": f"{args.task_suite_name}_t{task_id}_e{episode_idx}",
                }

                response = client_model.step(example=example_dict, step=step)
                raw_action = response["raw_action"]
                
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(f"Unexpected action sizes: "
                                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                    f"Falling back to LIBERO_DUMMY_ACTION.")
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)
                
                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    disturb_res[ID2CATEGORY[task_id+1][0]]['success_count'] += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"

            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{ID2CATEGORY[task_id+1][1]}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=25,
            )
            
            full_actions = np.stack(full_actions)
            
            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )
    _outname = f'{args.task_suite_name}_{_start}_to_{_end}.json'  # sharded; aggregate sums all shards
    with open(os.path.join(args.log_path, _outname), 'w', encoding='utf-8') as f:
        json.dump(disturb_res, f)
    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    tyro.cli(eval_libero)
