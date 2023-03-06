from absl import app
from absl import flags

from jax.random import PRNGKey

from graph_rewriter.agents.gat_hierarchical_agent_v2 import GATHierarchicalPPOAgent_v2
from graph_rewriter.environment.taso_hierarchical import HierarchicalEnvironment
from graph_rewriter.utils.inference import inference, load_agent_property

import taso as ts

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "Unknown", "path to the onnx model")
flags.DEFINE_string("agent_name", "GAT-hierarchical-ppo-v2-", "")
flags.DEFINE_string(
    "timestamp", None,
    "Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS")

flags.DEFINE_integer("horizon", 50, "hard horizon, in case policy got stuck")
flags.DEFINE_integer("seed", 42, "")
flags.DEFINE_integer("step", None, "ckpt step to load; None is the latest")
flags.DEFINE_integer("rand", 1, "whether sample from ppo deterministically")
flags.DEFINE_integer("reg_ckpt", 1, "whether to load from regular ckpt")


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

    # seed; will be overwritten by loading
    key = PRNGKey(FLAGS.seed)

    # log
    path_prefix = "output-data/logs/graph_rewriter/"
    path_prefix += f"{graph_name}/{FLAGS.agent_name}"
    if FLAGS.timestamp is None:
        raise ValueError(
            "must provide a checkpoint to load state for inference")
    else:
        path_prefix += FLAGS.timestamp
    info_filename = f"{path_prefix}/agent_spec.txt"
    info = load_agent_property(info_filename)

    # env
    env = HierarchicalEnvironment(
        num_locations=info["hparams"]["max_num_candidates"],
        real_measurements=False,
        reward_function=custom_reward)
    env.set_graph(graph)
    init_state = env.reset()
    # how many xfers
    # num_actions = env.get_num_actions()

    # agent
    spec = info["hparams"]
    spec.pop("seed")
    spec.pop("max_num_candidates")
    spec["key"] = key
    spec["state_input"] = init_state
    # the following is not important
    if "num_episodes" not in spec:
        spec["num_episodes"] = 10
    if "episodes_per_batch" not in spec:
        spec["episodes_per_batch"] = 1
    agent = GATHierarchicalPPOAgent_v2(**spec)

    # load
    rand = bool(FLAGS.rand)
    reg_ckpt = bool(FLAGS.reg_ckpt)
    print("load from regular ckpt:: ", reg_ckpt)
    print("Exploration:: ", rand)
    if reg_ckpt:
        prev_meta = agent.load_regular(FLAGS.step)
    else:
        prev_meta = agent.load(FLAGS.step)
    best_inference_runtime = prev_meta["best_inference_runtime"]
    print(
        f"best-inference-runtime from training: {best_inference_runtime:.4f}")

    inference(agent, env, graph, graph_name, FLAGS.horizon, rand, True, True)


if __name__ == "__main__":
    app.run(main)
