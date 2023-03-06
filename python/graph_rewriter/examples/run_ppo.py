import os
import json
from collections import deque
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np
from jax.random import PRNGKey
from jax import clear_backends

import taso as ts

from graph_rewriter.agents.gat_ppo_agent import GATPPO
from graph_rewriter.environment.taso_hierarchical import FlatEnvironment
from graph_rewriter.utils.inference import ppo_inference, print_available_block, get_complexity

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "Unknown", "path to the onnx model")
flags.DEFINE_string("an", "gat-ppo-v4-", "agent name")
flags.DEFINE_string(
    "timestamp", None,
    "Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS")

flags.DEFINE_integer("num_episodes", 1000, "")
flags.DEFINE_integer("clear_backend_round", 300,
                     "how many rounds to clear jax cache")
flags.DEFINE_integer("horizon", 100, "hard horizon, in case policy got stuck")
flags.DEFINE_integer("seed", 43, "")
flags.DEFINE_integer("max_num_candidates", int(500), "")
flags.DEFINE_integer("episodes_per_batch", 10, "How often will we update?")
flags.DEFINE_integer("mini_batch_size", 16, "")
flags.DEFINE_integer("update_round", 4, "how many epoch to update?")
flags.DEFINE_integer("cont_train", 1, "whether load from regular checkpoint")
flags.DEFINE_integer("verbose", 0, "print each iteration")


def load_graph_from_file(filename: str):
    print(f"Loading graph from file: {filename}")
    clean_filename = filename.split("/")[-1].split(".")[0]
    return clean_filename, ts.load_onnx(filename)


def custom_reward(time_step: int, last_runtime: float, init_runtime: float,
                  new_runtime: float, terminal: bool):
    percentage_diff = (last_runtime - new_runtime) / init_runtime
    percentage_diff *= 100
    return percentage_diff


def main(_):

    # import graph
    graph_name, graph = load_graph_from_file(FLAGS.path)

    # seed
    key = PRNGKey(FLAGS.seed)

    # logs
    path_prefix = "output-data/logs/graph_rewriter/"
    path_prefix += f"{graph_name}/{FLAGS.an}"
    if FLAGS.timestamp is None:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        path_prefix += current_time
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        print(f"Created Tensorboard log directory: {path_prefix}")
    else:
        path_prefix += FLAGS.timestamp
        print("Continuing provided log")

    output_filename = f"{path_prefix}/results.csv"
    info_filename = f"{path_prefix}/agent_spec.txt"
    train_log_dir = f"{path_prefix}/train"
    checkpoint_dir = f"{path_prefix}/checkpoint"
    regular_checkpoint_dir = f"{path_prefix}/regular_checkpoint"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # env
    env = FlatEnvironment(num_locations=FLAGS.max_num_candidates,
                          real_measurements=False,
                          reward_function=custom_reward,
                          node_cost_model=True)
    env.set_graph(graph)
    init_state = env.reset()
    num_actions = env.get_num_actions()  # xfers.size()

    # agent
    spec = {
        "num_actions": num_actions,
        "num_candidates": FLAGS.max_num_candidates,

        # for GAT
        "num_head": 4,
        "hidden_dim": 32,
        "message_passing_steps": 5,
        "gat_global_update_mlp": 64,

        # agent related
        "key": key,
        "state_input": init_state,
        "gamma_discount": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 5e-4,
        "clip_ratio": 0.2,
        "global_norm": 0.5,
        "target_kl": 0.05,
        "mini_batch_size": FLAGS.mini_batch_size,
        "update_round": FLAGS.update_round,

        # for v4
        "gat_attn_mlp": 64,
        "gat_node_update_mlp": -1,
        "policy_feature": [256, 64],
        "vf_feature": [256, 64],

        # utils
        "num_episodes": FLAGS.num_episodes,
        "episodes_per_batch": FLAGS.episodes_per_batch,
        "name": graph_name,
        "checkpoint_path": checkpoint_dir,
        "regular_checkpoint_dir": regular_checkpoint_dir,
    }
    agent = GATPPO(**spec)

    # load agent
    cont_train = bool(FLAGS.cont_train)
    if FLAGS.timestamp is None:
        # fresh start
        start_episode = 1
        best_inference_runtime = float("inf")
    elif cont_train:
        # load from regular ckpt
        prev_meta = agent.load_regular()
        start_episode = prev_meta["episode"] + 1
        best_inference_runtime = prev_meta["best_inference_runtime"]
    else:
        # load from best ckpt
        prev_meta = agent.load()
        start_episode = prev_meta["episode"] + 1
        best_inference_runtime = prev_meta["best_inference_runtime"]

    # file handler
    if FLAGS.timestamp is None:
        with open(info_filename, "wt") as fp:
            hp = deepcopy(spec)
            hp["seed"] = FLAGS.seed
            hp["max_num_candidates"] = FLAGS.max_num_candidates
            hp.pop("key")
            hp.pop("state_input")
            json.dump({"hparams": hp, "graphs": [graph_name]}, fp)

    print(f"Output filename: {output_filename}")
    output_file = open(output_filename, "at")

    print()
    print("=====================================")
    print("=====================================")

    # Storing samples.
    episode_rewards = deque(maxlen=FLAGS.episodes_per_batch)
    states = []
    main_actions = []
    main_log_probs = []
    main_vf_values = []
    rewards = []
    dones = []

    # =================== env loop ===================
    print(f"Training on graph: {graph_name}")
    initial_runtime = env.get_cost()
    print(f"initial cost model runtime: {initial_runtime:.4f}")
    initial_runtime_graph = env.real_measurements_runtime
    print(f"initial end-to-end inference time: {initial_runtime_graph:.4f}")
    verbose = bool(FLAGS.verbose)
    for current_episode in tqdm(range(start_episode, FLAGS.num_episodes + 1)):

        # init the graph
        env.set_graph(graph)
        state = env.reset()
        initial_runtime = env.get_cost()

        episode_reward = 0
        timestep = 0
        action_history = []
        complexity = 0
        while True:

            complexity += get_complexity(env)
            if verbose:
                print_available_block(env)

            # if explore, sampling from action probs
            # else, deterministically argmax
            main_action, main_log_prob, main_vf_value = agent.act(state,
                                                                  explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, done, info = env.step(main_action)

            if verbose:
                print(f"action: {main_action[0]}", end=" - ")
                print(f"rewards: {reward:.2f}", end=" - ")
                print(f"complexity: {complexity}")

            # Append to buffer.
            states.append(state)

            # Main action
            main_actions.append(main_action)
            main_log_probs.append(main_log_prob)
            main_vf_values.append(main_vf_value)

            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

            # Store the xfer applied
            action_history.append((info["xfer_id"], info["location_id"]))

            timestep += 1

            # If done, reset.
            if done or timestep > FLAGS.horizon:
                episode_rewards.append(episode_reward)
                ave_complexity = complexity / timestep

                # cost model
                final_runtime = env.get_cost()
                # end-to-end inf
                graph_runtime = env.last_measured_runtime
                output_file.write(
                    f"Episode: {current_episode} - episode_reward: {episode_reward:.4f} - cost model runtime: {final_runtime:.4f} - end-to-end inference time {graph_runtime:.4f} - average complexity: {ave_complexity}\n"
                )
                output_file.flush()

                print(f"Episode: {current_episode}", end=" - ")
                print(f"Episode timestep: {timestep}", end=" - ")
                print(f"average complexity {ave_complexity:.1f}", end=" - ")
                print(f"Cost model runtime: {final_runtime:.4f}", end=" - ")
                print(f"end-to-end inference time: {graph_runtime:.4f}")

                print(
                    f"Cost model diff (the lower the better):\t"
                    f"{final_runtime - initial_runtime:+.4f} "
                    f"({(final_runtime - initial_runtime) / initial_runtime:+.2%})"
                )
                print(
                    f"Graph inference time diff (the lower the better):\t"
                    f"{graph_runtime - initial_runtime_graph:+.4f} "
                    f"({(graph_runtime - initial_runtime_graph) / initial_runtime_graph:+.2%})"
                )
                print(action_history)
                print("-" * 40)

                # The Update Step
                if current_episode % FLAGS.episodes_per_batch == 0:
                    print("============================================")
                    ave_reward = np.mean(episode_rewards)
                    std_reward = np.std(episode_rewards)
                    print(f"Last {FLAGS.episodes_per_batch} episodes", end=" ")
                    print(f"Mean reward: {ave_reward:.4f}", end=" - ")
                    print(f"Std: {std_reward:.4f}")
                    info = agent.update(states=states,
                                        main_actions=main_actions,
                                        main_log_probs=main_log_probs,
                                        main_vf_values=main_vf_values,
                                        rewards=rewards,
                                        dones=dones)
                    main_actor_loss = info["main_actor_loss"]
                    main_vf_loss = info["main_vf_loss"]
                    main_entropy = info["main_entropy"]
                    main_kl = info["main_kl"]
                    update_time = info["update_time"]
                    # Loss should be decreasing.
                    print(f"[update] took {update_time:.2f} seconds",
                          end=" - ")
                    print(f"policy loss = {main_actor_loss:.4f}", end=" - ")
                    print(f"vf loss = {main_vf_loss:.4f}", end=" - ")
                    print(f"entropy {main_entropy:.4f}", end=" - ")
                    print(f"KL {main_kl:.4f}")

                    # reset
                    states = []
                    main_actions = []
                    main_log_probs = []
                    main_vf_values = []
                    rewards = []
                    dones = []

                    # Log to tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar("chart/update_time",
                                          update_time,
                                          step=current_episode)
                        tf.summary.scalar("chart/episode_reward",
                                          episode_reward,
                                          step=current_episode)
                        for k, v in info.items():
                            tf.summary.scalar("agent/" + k,
                                              v,
                                              step=current_episode)

                    agent.regular_save(current_episode, best_inference_runtime)

                if current_episode % FLAGS.clear_backend_round == 0:
                    # require jaxlib >= 0.3.15
                    clear_backends()

                # exit this episode because done
                break

    output_file.close()

    print("============================================")


if __name__ == "__main__":
    app.run(main)
