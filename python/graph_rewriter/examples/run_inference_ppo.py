from absl import app
from absl import flags

from jax.random import PRNGKey

from graph_rewriter.agents.gat_ppo_agent import GATPPO
from graph_rewriter.environment.taso_hierarchical import FlatEnvironment
from graph_rewriter.utils.inference import ppo_inference, print_available_block, get_complexity, load_agent_property

import taso as ts

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "Unknown", "path to the onnx model")
flags.DEFINE_string("an", "gat-ppo-v4-", "")
flags.DEFINE_string(
    "timestamp", None,
    "Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS")

flags.DEFINE_integer("max_num_candidates", int(500), "")
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
    path_prefix += f"{graph_name}/{FLAGS.an}"
    if FLAGS.timestamp is None:
        raise ValueError(
            "must provide a checkpoint to load state for inference")
    else:
        path_prefix += FLAGS.timestamp
    info_filename = f"{path_prefix}/agent_spec.txt"
    info = load_agent_property(info_filename)

    # env
    # NOTE env setting must match its training setting, e.g. node_cost_model
    env = FlatEnvironment(num_locations=FLAGS.max_num_candidates,
                          real_measurements=False,
                          reward_function=custom_reward,
                          node_cost_model=True)
    env.set_graph(graph)
    init_state = env.reset()

    # agent
    spec = info["hparams"]
    spec.pop("seed")
    spec.pop("max_num_candidates")
    print("=" * 40)
    print("spec::")
    print(spec)
    print("=" * 40)
    spec["key"] = key
    spec["state_input"] = init_state
    agent = GATPPO(**spec)

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

    ppo_inference(agent, env, graph, graph_name, FLAGS.horizon, rand, True,
                  True)


if __name__ == "__main__":
    app.run(main)
