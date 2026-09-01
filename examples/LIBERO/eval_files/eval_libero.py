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

# Inlined so the simulator env does not need the policy package installed.
_GRIP_SEP_CLOSED = 0.005
_GRIP_SEP_OPEN = 0.08
def _gripper_open_pct(state_vec):
    if len(state_vec) < 8:
        return 0
    sep = float(state_vec[6]) - float(state_vec[7])
    frac = (sep - _GRIP_SEP_CLOSED) / (_GRIP_SEP_OPEN - _GRIP_SEP_CLOSED)
    return int(round(min(1.0, max(0.0, frac)) * 100))
def _describe_robot_state(state_vec):
    import math
    if state_vec is None or len(state_vec) < 6:
        return ""
    x, y, z = (int(round(float(state_vec[i]) * 1000)) for i in range(3))
    r, p, yw = (int(round(float(state_vec[i]) * 180.0 / math.pi)) for i in range(3, 6))
    g = _gripper_open_pct(state_vec)
    return (f"end-effector position x={x} y={y} z={z} mm; "
            f"orientation roll={r} pitch={p} yaw={yw} deg; gripper open {g}%")


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


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

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    # Parallel worker split — handle only episodes whose flat index
    #   (task_id * num_trials_per_task + episode_idx) % num_workers == worker_id
    # Default num_workers=1 → all episodes (back-compat).
    worker_id: int = 0
    num_workers: int = 1

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


    if args.num_workers > 1:
        logging.info(f"Worker split: worker_id={args.worker_id} / num_workers={args.num_workers} "
                     f"(handle (tid*N+eid)%num_workers==worker_id)")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Skip tasks where this worker has zero assigned episodes (avoid env init cost)
        my_eps = [eid for eid in range(args.num_trials_per_task)
                  if (task_id * args.num_trials_per_task + eid) % args.num_workers == args.worker_id]
        if not my_eps:
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in my_eps:
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

                # suite/task_id/episode_id let the server look up and lock this
                # episode's demo. NOTE: no "state" key — the LIBERO training
                # pipeline never emitted one, so the model trained with
                # state=None and its state encoder is untrained. Sending a state
                # would route through that dead weight.
                # State-aug parity: current-obs eef state as prompt TEXT. The lerobot
                # observation.state[3:6] is AXIS-ANGLE (meta names: axis_angle1/2/3) —
                # training rendered THOSE values via _describe_robot_state, so eval MUST
                # pass axis-angle too (NOT euler; they diverge by >100deg mid-trajectory).
                # NOT the DiT state path.
                _eef_aa = _quat2axisangle(obs["robot0_eef_quat"])
                _state8 = np.concatenate((obs["robot0_eef_pos"], _eef_aa, obs["robot0_gripper_qpos"]))
                _state_text = _describe_robot_state(_state8)
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                    "state_text": _state_text,
                    # Numeric 8-dim eef state (raw axis-angle, matching dataset
                    # emit_state) for the DiT state_encoder. Required by the
                    # state-input variant (action_model.state_dim=8). For older
                    # state_dim=9 models, the encoder is untrained and sending
                    # this would dim-mismatch — those evals are complete and
                    # not re-run.
                    "state": _state8.astype(np.float32),
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
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            full_actions = np.stack(full_actions)

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

        # Per-task results file (worker suffix for parallel runs).
        # Format mirrors auto_eval_stellavla_*: "<task_desc>\n<succ>/<eps>\n"
        # so the launcher's tally loop picks them up.
        try:
            log_dir = pathlib.Path(args.video_out_path).parent / "logs" / args.task_suite_name
            log_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_w{args.worker_id:02d}" if args.num_workers > 1 else ""
            with (log_dir / f"task{task_id:02d}_results{suffix}.txt").open("w") as f:
                f.write(f"{task_description}\n{task_successes}/{task_episodes}\n")
        except Exception as e:
            logging.warning(f"Failed to write per-task results file: {e}")

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
        "bddl_file_name": task_bddl_file,
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
