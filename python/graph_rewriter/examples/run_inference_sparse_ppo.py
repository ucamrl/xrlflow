from absl import app
from absl import flags

from jax.random import PRNGKey

from graph_rewriter.agents.gn_ppo_agent import GNPPO
from graph_rewriter.environment.taso_hierarchical import FlatEnvironment_sparse
from graph_rewriter.utils.inference import ppo_inference, load_agent_property

import taso as ts

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "Unknown", "path to the onnx model")
flags.DEFINE_string("an", "gn-ppo-v4-", "")
flags.DEFINE_string("gn", None, "graph name")
flags.DEFINE_string(
    "timestamp", None,
    "Timestamp of the checkpoint to evaluate in the format YYYYMMDD-HHMMSS")

flags.DEFINE_integer("max_num_candidates", int(500), "")
flags.DEFINE_integer("horizon", 100, "hard horizon, in case policy got stuck")
flags.DEFINE_integer("seed", 42, "")
flags.DEFINE_integer("step", None, "ckpt step to load; None is the latest")
flags.DEFINE_integer("rand", 1, "whether sample from ppo deterministically")
flags.DEFINE_integer("reg_ckpt", 1, "whether to load from regular ckpt")
flags.DEFINE_integer(
    "mi", 10000,
    "measure interval; a large number denotes no intermediate measurement")


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
    # import graph; graph_name can be overwritten for transfer inference
    graph_name, graph = load_graph_from_file(FLAGS.path)
    if FLAGS.gn is not None:
        graph_name = FLAGS.gn

    # change to measure_runtime upfront
    graph_initial_measure_runtime = graph.run_time()

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
    env = FlatEnvironment_sparse(
        num_locations=FLAGS.max_num_candidates,
        real_measurements=False,
        reward_function=custom_reward,
        # ensure no intermediate measurement when time opt speed
        measure_interval=FLAGS.mi,
        node_cost_model=False)
    env.set_graph(graph)
    init_state = env.reset()

    # agent
    spec = info["hparams"]
    pops = ["seed", "max_num_candidates"]
    for p in pops:
        if p in spec:
            spec.pop(p)
    print("=" * 40)
    print("spec::")
    print(spec)
    print("=" * 40)
    spec["key"] = key
    spec["state_input"] = init_state
    agent = GNPPO(**spec)

    # load
    print("=" * 40)
    rand = bool(FLAGS.rand)
    reg_ckpt = bool(FLAGS.reg_ckpt)
    print("load from regular ckpt:: ", reg_ckpt)
    print("Exploration:: ", rand)
    print("measure_interval:: ", FLAGS.mi)
    print("=" * 40)
    if reg_ckpt:
        agent.load_regular(FLAGS.step)
    else:
        agent.load(FLAGS.step)

    # NOTE: overwrite agent's key for stochastic policy
    # otherwises it restores from most recent key
    agent.key = key

    ppo_inference(agent, env, graph, graph_name, FLAGS.horizon, rand, True,
                  True, graph_initial_measure_runtime)


if __name__ == "__main__":
    app.run(main)
