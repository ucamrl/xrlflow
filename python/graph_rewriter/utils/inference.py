import json
import time


def load_agent_property(fn: str) -> dict:
    with open(fn, "r") as f:
        data = json.load(f)
    return data


def print_available_block(env):
    """show how many xfer/locations are truly available for applying"""
    locations = env.locations
    n = len(locations)
    loc = []
    for i in range(n):
        if locations[i] != 0:
            loc.append((i, locations[i]))  # xfer id, num of locations
    print("available xfers: ", loc)


def get_complexity(env):
    """sum of all available rules to all available places"""
    locations = env.locations
    n = len(locations)
    cnt = 0
    for i in range(n):
        if locations[i] != 0:
            cnt += locations[i]
    return cnt


def inference(agent, env, graph, graph_name, inference_horizon: int,
              rand: bool, verbose: bool, measure: bool):
    env.set_graph(graph)
    state = env.reset()
    initial_runtime = env.initial_runtime
    if measure:
        graph_initial_measure_runtime = env.graph.run_time()
    print("=====================================")
    print(f"Run inference on graph: {graph_name}")
    print("Start runtime: {:.4f}".format(initial_runtime))
    rewards = []
    terminal = False
    episode_reward = 0
    time_step = 0
    actions_history = []

    start_time = time.perf_counter()
    while True:
        if verbose:
            print_available_block(env)

        c = get_complexity(env)
        main_action, main_log_prob, main_vf_value, sub_action, sub_log_prob, sub_vf_value = agent.act(
            env, states=state, explore=rand)

        # Action delivered in shape (1,), need ()
        next_state, reward, terminal, _ = env.step((main_action, sub_action))

        rewards.append(reward)

        state = next_state
        episode_reward += reward
        time_step += 1

        actions_history.append((int(main_action), int(sub_action)))

        if verbose:
            print(
                "Iteration {}. Graph: {}. Complexity: {}. block_id: {}. candidate_id: {}. Reward: {:.6f}. Terminal: {}"
                .format(len(rewards), graph_name, c, main_action, sub_action,
                        reward, terminal))

        # If terminal, reset. deterministic policy may need a hard horizon
        if terminal or time_step > inference_horizon:
            final_runtime = env.last_runtime
            break

    time_taken_rl = time.perf_counter() - start_time
    print("Time taken for RL inference: {:.2f} seconds".format(time_taken_rl))
    print(
        f"Episode timestep: {time_step} - Final runtime: {final_runtime:.4f}")
    print(
        "Observed speedup for RL inference: {:.4f} seconds (final runtime: {:.4f})."
        .format(final_runtime - initial_runtime, final_runtime))
    print(
        f'Difference (the lower the better):\t'
        f'{final_runtime - initial_runtime:+.4f} ({(final_runtime - initial_runtime) / initial_runtime:+.2%})'
    )
    print("action history:")
    print(actions_history)
    print("-" * 40)
    if measure:
        graph_measure_runtime = env.get_pre_process_graph().run_time()
        print(
            f"TASO graph initial measured runtime: {graph_initial_measure_runtime:.4f} ms"
        )
        print(f"TASO graph runtime: {graph_measure_runtime:.4f} ms")
        print("-" * 40)

    return final_runtime, time_taken_rl


def ppo_inference(agent,
                  env,
                  graph,
                  graph_name,
                  inference_horizon: int,
                  rand: bool,
                  verbose: bool,
                  measure: bool,
                  graph_initial_measure_runtime=None):
    if measure and graph_initial_measure_runtime is None:
        graph_initial_measure_runtime = graph.run_time()
    else:
        graph_initial_measure_runtime = graph_initial_measure_runtime
    env.set_graph(graph)
    state = env.reset()
    initial_runtime = env.initial_runtime
    print("=====================================")
    cost_model = env.get_cost()
    e2e = env.last_measured_runtime
    print("Initial measurement::")
    print(
        f"cost model {cost_model:.4f} - e2e {graph_initial_measure_runtime:.4f}"
    )
    print("=====================================")
    rewards = []
    terminal = False
    episode_reward = 0
    time_step = 0
    actions_history = []
    cc = 0

    start_time = time.perf_counter()
    while True:
        if verbose:
            print_available_block(env)

        c = get_complexity(env)
        main_action, _, _ = agent.act(states=state, explore=rand)

        # Action delivered in shape (1,), need ()
        next_state, reward, terminal, info = env.step(main_action)

        rewards.append(reward)

        state = next_state
        episode_reward += reward
        time_step += 1
        cc += c

        xfer_id = info["xfer_id"]
        location_id = info["location_id"]
        actions_history.append((info["xfer_id"], info["location_id"]))

        cost_model = env.get_cost()
        e2e = env.last_measured_runtime

        if verbose:
            print(
                "Iteration {}. Graph: {}. Complexity: {}. action: {}@{}. Reward: {:.6f}. Terminal: {}"
                .format(len(rewards), graph_name, c, xfer_id, location_id,
                        reward, terminal))
            print(f"cost model {cost_model:.4f} - e2e {e2e:.4f}")
            print("=====================================")

        # If terminal, reset. deterministic policy may need a hard horizon
        if terminal or time_step > inference_horizon:
            # final_runtime = env.last_runtime
            final_runtime = env.get_cost()
            break

    time_taken_rl = time.perf_counter() - start_time
    print("-" * 40)
    print("Time taken for RL inference: {:.2f} seconds".format(time_taken_rl))
    ave_c = cc / time_step
    print(f"average complexity: {ave_c}")
    print("action history:")
    print(actions_history)
    print("-" * 40)
    print("Cost model:: ")
    print(
        f"Episode timestep: {time_step} - Final runtime: {final_runtime:.4f}")
    print(
        "Observed speedup for RL inference: {:.4f} seconds (final runtime: {:.4f})."
        .format(final_runtime - initial_runtime, final_runtime))
    print(
        f'Difference (the lower the better):\t'
        f'{final_runtime - initial_runtime:+.4f} ({(final_runtime - initial_runtime) / initial_runtime:+.2%})'
    )
    print("-" * 40)
    if measure:
        graph_measure_runtime = env.get_pre_process_graph().run_time()
        print("end-to-end::")
        print(
            f"TASO graph initial measured runtime: {graph_initial_measure_runtime:.4f} ms"
        )
        print(f"TASO graph runtime: {graph_measure_runtime:.4f} ms")
        per_diff = (graph_initial_measure_runtime -
                    graph_measure_runtime) / graph_initial_measure_runtime
        print(f"TASO graph % speedup {per_diff:+.2%}")
        print("-" * 40)

    return final_runtime, time_taken_rl
