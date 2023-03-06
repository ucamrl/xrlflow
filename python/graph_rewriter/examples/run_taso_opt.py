import time

from absl import app
from absl import flags

import taso as ts

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "Unknown", "path to the onnx model")
flags.DEFINE_float("alpha", 1.0, "")
flags.DEFINE_integer("budget", 100, "")


def load_graph_from_file(filename: str):
    print(f"Loading graph from file: {filename}")
    clean_filename = filename.split('/')[-1].split('.')[0]
    return clean_filename, ts.load_onnx(filename)


def run_taso_optimize(graph,
                      name: str = 'Untitled',
                      alpha: float = 1.0,
                      budget: float = 1000):
    """
    run taso optimisation on a given graph
    """
    print("Training on graph: {}".format(name))

    start_runtime_taso = graph.cost()
    graph_initial_measure_runtime = graph.run_time()

    start_time = time.perf_counter()
    optimized_graph = ts.optimize(graph, alpha=alpha, budget=budget)
    time_taken_taso = time.perf_counter() - start_time

    final_runtime_taso = optimized_graph.cost()

    print("-" * 40)
    print("Optimized graph {} in ".format(name))
    print("Time taken for TASO search: {:.2f} seconds".format(time_taken_taso))
    print(
        f"Difference (the lower the better):\t"
        f"{final_runtime_taso - start_runtime_taso:+.4f} ({(final_runtime_taso - start_runtime_taso) / start_runtime_taso:+.2%})"
    )
    print("-" * 40)
    print(f"cost model start_runtime: {start_runtime_taso:.4f}")
    print(f"cost model final_runtime_taso: {final_runtime_taso:.4f}")
    print("TASO graph initial measured runtime: {:.4f} ms".format(
        graph_initial_measure_runtime))
    graph_final_measured_runtime = optimized_graph.run_time()
    print("TASO graph measured runtime: {:.4f} ms".format(
        graph_final_measured_runtime))
    per_diff = (graph_initial_measure_runtime -
                graph_final_measured_runtime) / graph_initial_measure_runtime
    print(f"TASO graph % speedup {per_diff:+.2%}")
    print("-" * 40)

    return optimized_graph, start_runtime_taso, final_runtime_taso, time_taken_taso


def main(args):
    graph_name, graph = load_graph_from_file(FLAGS.path)
    graphs = [(graph_name, graph)]

    for current_graph_file, current_graph in graphs:
        optimized_graph, start_runtime, final_runtime_taso, time_taken_taso = run_taso_optimize(
            current_graph,
            current_graph_file,
            alpha=FLAGS.alpha,
            budget=FLAGS.budget)
        print(f"graph: {current_graph_file}")
        print(f"start_runtime: {start_runtime:.4f} seconds")
        print(f"final_runtime_taso: {final_runtime_taso:.4f} seconds")


if __name__ == '__main__':
    app.run(main)
