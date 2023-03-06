from absl import app
from absl import flags
import time

import taso as ts

from graph_rewriter.environment.taso_hierarchical import HierarchicalEnvironment
from graph_rewriter.utils.inference import print_available_block, get_complexity

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "Unknown", "path to the onnx model")
flags.DEFINE_string("agent_name", "GAT-hierarchical-ppo-v2-", "")
flags.DEFINE_string(
    "timestamp", None,
    "Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS")

flags.DEFINE_integer("num_episodes", 2000, "")
flags.DEFINE_integer("clear_backend_round", 100,
                     "how many rounds to clear jax cache")
flags.DEFINE_integer("horizon", 50, "hard horizon, in case policy got stuck")
flags.DEFINE_integer("seed", 43, "")
flags.DEFINE_integer("max_num_candidates", int(100), "")
flags.DEFINE_integer("episodes_per_batch", 10, "How often will we update?")
flags.DEFINE_integer("cont_train", 1, "whether load from regular checkpoint")
flags.DEFINE_boolean("verbose", False, "print each iteration")


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

    # env
    env = HierarchicalEnvironment(num_locations=FLAGS.max_num_candidates,
                                  real_measurements=False,
                                  reward_function=custom_reward)
    # =================== env loop ===================
    # init the graph
    env.set_graph(graph)
    _ = env.reset()
    # init measurement
    initial_runtime = env.graph.cost()

    timestep = 0
    complexity = 0
    start_time = time.perf_counter()
    while True:

        complexity += get_complexity(env)
        print_available_block(env)
        timestep += 1
        break

    time_taken = time.perf_counter() - start_time
    # after
    ave_complexity = complexity / timestep
    final_runtime = env.graph.cost()
    print("-" * 40)
    print(f"average complexity {ave_complexity:.1f}")
    print(f"time taken {time_taken:.2f} seconds")
    print(f"Cost model Initial runtime:\t{initial_runtime:.4f}")
    print(f"Cost model Final runtime:\t{final_runtime:.4f}")
    print(
        f"Difference (the lower the better):\t"
        f"{final_runtime - initial_runtime:+.4f} ({(final_runtime - initial_runtime) / initial_runtime:+.2%})"
    )
    print("-" * 40)


if __name__ == "__main__":
    app.run(main)
