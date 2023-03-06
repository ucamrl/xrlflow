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

import taso as ts

from graph_rewriter.agents.gat_hierarchical_agent import GATHierarchicalPPOAgent
from graph_rewriter.environment.taso_hierarchical import HierarchicalEnvironment
from graph_rewriter.utils.inference import inference

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "Unknown", "path to the onnx model")
flags.DEFINE_string("agent_name", "GAT_hierarchical_agent", "")
flags.DEFINE_string(
    "timestamp", None,
    "Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS")

flags.DEFINE_integer("num_episodes", 2000, "")
flags.DEFINE_integer("horizon", 50, "hard horizon, in case policy got stuck")
flags.DEFINE_integer("seed", 42, "")
# flags.DEFINE_integer("max_num_candidates", int(5 * 1e4), "")
flags.DEFINE_integer("max_num_candidates", int(200), "")
flags.DEFINE_integer("episodes_per_batch", 10, "How often will we update?")
flags.DEFINE_boolean("cont_train", True,
                     "whether load from regular checkpoint")
flags.DEFINE_boolean("verbose", False, "print each iteration")


def load_graph_from_file(filename: str):
    print(f"Loading graph from file: {filename}")
    clean_filename = filename.split('/')[-1].split('.')[0]
    return clean_filename, ts.load_onnx(filename)


def print_available_block(env):
    """show how many xfer/locations are truly available for applying"""
    l = env.locations
    n = len(l)
    loc = []
    for i in range(n):
        if l[i] != 0:
            loc.append((i, l[i]))  # xfer id, num of locations
    print(loc)


def custom_reward(time_step: int, last_runtime: float, init_runtime: float,
                  new_runtime: float, terminal: bool):
    # use percetage of decrease as reward
    # print("reward: ")
    # print(last_runtime, new_runtime, init_runtime)
    percentage_diff = (last_runtime - new_runtime) / init_runtime
    percentage_diff *= 100
    bonus = 0.05
    if time_step > 10:
        bonus = 0
    elif time_step > 40:
        bonus = -1
    elif time_step < 5 and terminal:
        # penalize too early
        bonus = time_step - 5
    return bonus + percentage_diff


# NOTE: can update globals per GAT update
# NOTE: can change vf init() and use rule activation later
def main(_):

    # import graph
    graph_name, graph = load_graph_from_file(FLAGS.path)

    # seed
    key = PRNGKey(FLAGS.seed)

    # logs
    path_prefix = f"output-data/logs/graph_rewriter/{graph_name}/{FLAGS.agent_name}"
    if FLAGS.timestamp is None:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        path_prefix += current_time
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        print(f'Created Tensorboard log directory: {path_prefix}')
    else:
        path_prefix += FLAGS.timestamp
        print('Continuing provided log')

    output_filename = f'{path_prefix}/results.csv'
    info_filename = f'{path_prefix}/agent_spec.txt'
    train_log_dir = f'{path_prefix}/train'
    checkpoint_dir = f'{path_prefix}/checkpoint'
    regular_checkpoint_dir = f'{path_prefix}/regular_checkpoint'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # env
    env = HierarchicalEnvironment(num_locations=FLAGS.max_num_candidates,
                                  real_measurements=False,
                                  reward_function=custom_reward)
    env.set_graph(graph)
    init_state = env.reset()
    # how many xfers
    num_actions = env.get_num_actions()  # xfers.size()

    # agent
    spec = {
        "num_actions": num_actions,
        "num_candidates": FLAGS.max_num_candidates,

        # gnn related
        "gat_attn_mlp": 32,
        "gat_node_update_mlp": -1,  # not use
        "message_passing_steps": 5,
        "gat_global_update_mlp": 64,

        # agent related
        "key": key,
        "state_input": init_state,
        "gamma_discount": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 0.0025,
        "clip_ratio": 0.3,
        "global_norm": 0.5,
        "target_kl": None,  # 0.05,
        "policy_feature": [64, 16],
        "vf_feature": [64, 16],
        "mini_batch_size": 8,
        "update_round": 3,

        # utils
        "name": graph_name,
        "checkpoint_path": checkpoint_dir,
        "regular_checkpoint_dir": regular_checkpoint_dir,
    }
    agent = GATHierarchicalPPOAgent(**spec)

    # load agent
    if FLAGS.timestamp is None:
        # fresh start
        start_episode = 1
        best_inference_runtime = float("inf")
    elif FLAGS.cont_train:
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
        with open(info_filename, 'wt') as fp:
            hp = deepcopy(spec)
            hp["seed"] = FLAGS.seed
            hp["max_num_candidates"] = FLAGS.max_num_candidates
            hp.pop("key")
            hp.pop("state_input")
            json.dump({'hparams': hp, 'graphs': [graph_name]}, fp)

    print(f'Output filename: {output_filename}')
    output_file = open(output_filename, 'at')

    print()
    print("=====================================")
    print("=====================================")

    # Storing samples.
    episode_rewards = deque(maxlen=FLAGS.episodes_per_batch)
    states = []
    main_actions = []
    main_log_probs = []
    main_vf_values = []
    sub_actions = []
    sub_log_probs = []
    sub_vf_values = []
    rewards = []
    dones = []

    # =================== env loop ===================
    print(f'Training on graph: {graph_name}')
    initial_runtime = env.get_cost()
    best_training_runtime = float("inf")
    print(f'initial runtime: {initial_runtime:.4f}')
    initial_runtime_graph = graph.cost()
    print(f'initial_runtime_graph: {initial_runtime_graph:.4f}')
    for current_episode in tqdm(range(start_episode, FLAGS.num_episodes + 1)):

        # init the graph
        env.set_graph(graph)
        state = env.reset()
        initial_runtime = env.get_cost()

        episode_reward = 0
        timestep = 0
        xfers_applied = {}
        while True:

            if FLAGS.verbose:
                print_available_block(env)

            # if explore, sampling from action prob - else, deterministically argmax
            main_action, main_log_prob, main_vf_value, sub_action, sub_log_prob, sub_vf_value = agent.act(
                states=state, explore=True)

            # Action delivered in shape (1,), need ()
            next_state, reward, done, _ = env.step((main_action, sub_action))

            if FLAGS.verbose:
                print(
                    f"action: {main_action[0]} @ {sub_action} - rewards: {reward:.2f}"
                )

            # Append to buffer.
            states.append(state)

            # Main action
            main_actions.append(main_action)
            main_log_probs.append(main_log_prob)
            main_vf_values.append(main_vf_value)

            # Sub action
            sub_actions.append(sub_action)
            sub_log_probs.append(sub_log_prob)
            sub_vf_values.append(sub_vf_value)

            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

            # Store the xfer applied
            if str(main_action[0]) not in xfers_applied:
                xfers_applied[str(main_action[0])] = 0
            xfers_applied[str(main_action[0])] += 1

            timestep += 1

            # If done, reset.
            if done or timestep > FLAGS.horizon:
                episode_rewards.append(episode_reward)

                # Env reset is handled in outer loop
                final_runtime = env.get_cost()
                # log
                output_file.write(
                    f'Episode: {current_episode} - episode_reward: {episode_reward:.4f}, runtime: {final_runtime:.4f}\n'
                )
                output_file.flush()

                print(
                    f"Episode: {current_episode} - Episode timestep: {timestep}"
                )
                print(f'Final runtime:\t{final_runtime:.4f}')
                print(
                    f'Difference (the lower the better):\t'
                    f'{final_runtime - initial_runtime:+.4f} ({(final_runtime - initial_runtime) / initial_runtime:+.2%})'
                )
                print(xfers_applied
                      )  # k: block id - v: how many times its been mutated
                print('-' * 40)

                # if performance boost: save
                if final_runtime < best_training_runtime:
                    best_training_runtime = final_runtime

                # The Update Step
                # Do an update after collecting specified number of batches.
                # This is a hyper-parameter that will require a lot of experimentation.
                # One episode could be one rewrite of the graph, and it may be desirable to perform a small
                # update after every rewrite.
                if current_episode % FLAGS.episodes_per_batch == 0:
                    print("============================================")
                    ave_reward = np.mean(episode_rewards)
                    std_reward = np.std(episode_rewards)
                    print(
                        f"Finished episode = {current_episode}, Mean reward for last {FLAGS.episodes_per_batch} episodes = {ave_reward:.4f} - Std: {std_reward:.4f}"
                    )
                    # Simply pass collected trajectories to the agent for a single update.
                    info = agent.update(states=states,
                                        main_actions=main_actions,
                                        main_log_probs=main_log_probs,
                                        main_vf_values=main_vf_values,
                                        sub_actions=sub_actions,
                                        sub_log_probs=sub_log_probs,
                                        sub_vf_values=sub_vf_values,
                                        rewards=rewards,
                                        dones=dones)
                    main_actor_loss = info["main_actor_loss"]
                    main_vf_loss = info["main_vf_loss"]
                    main_entropy = info["main_entropy"]
                    sub_actor_loss = info["sub_actor_loss"]
                    sub_vf_loss = info["sub_vf_loss"]
                    sub_entropy = info["sub_entropy"]
                    main_kl = info["main_kl"]
                    sub_kl = info["sub_kl"]
                    # Loss should be decreasing.
                    print(
                        f"policy loss = {main_actor_loss:.4f} - vf loss = {main_vf_loss:.4f} - entropy {main_entropy:.4f} - main_kl: {main_kl:.4f}"
                    )
                    print(
                        f"sub policy loss = {sub_actor_loss:.4f}, sub vf loss = {sub_vf_loss:.4f} - sub entropy {sub_entropy:.4f} - sub_kl: {sub_kl:.4f}"
                    )

                    # reset
                    states = []
                    main_actions = []
                    main_log_probs = []
                    main_vf_values = []
                    sub_actions = []
                    sub_log_probs = []
                    sub_vf_values = []
                    rewards = []
                    dones = []

                    # Log to tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('episode_reward',
                                          episode_reward,
                                          step=current_episode)
                        tf.summary.scalar('main_actor_loss',
                                          main_actor_loss,
                                          step=current_episode)
                        tf.summary.scalar('main_entropy',
                                          main_entropy,
                                          step=current_episode)
                        tf.summary.scalar('main_vf_loss',
                                          main_vf_loss,
                                          step=current_episode)
                        tf.summary.scalar('sub_actor_loss',
                                          sub_actor_loss,
                                          step=current_episode)
                        tf.summary.scalar('sub_entropy',
                                          sub_entropy,
                                          step=current_episode)
                        tf.summary.scalar('sub_vf_loss',
                                          sub_vf_loss,
                                          step=current_episode)
                        for k, v in info.items():
                            tf.summary.scalar(k, v, step=current_episode)

                    # disable explore inference RL
                    inference_runtime, _ = inference(agent, env, graph,
                                                     graph_name, FLAGS.horizon,
                                                     False, False, False)
                    if inference_runtime < best_inference_runtime:
                        best_inference_runtime = inference_runtime
                        agent.save(current_episode, best_inference_runtime)
                        print(f'Checkpoint Episode = {current_episode}')
                    # regularly save regardless
                    agent.regular_save(current_episode, best_inference_runtime)

                # exit this episode because done
                break

    output_file.close()

    print('============================================')
    print(f"Best runtime during training {best_training_runtime:.4f}")


if __name__ == '__main__':
    app.run(main)
