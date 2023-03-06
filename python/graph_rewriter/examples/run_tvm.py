import time
from absl import app
from absl import flags

import onnx

import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "Unknown", "path to the onnx model")
flags.DEFINE_float("alpha", 1.0, "")
flags.DEFINE_integer("budget", 100, "")


def load_graph_from_file(filename: str):
    print(f"Loading graph from file: {filename}")
    clean_filename = filename.split('/')[-1].split('.')[0]
    return clean_filename, onnx.load(filename)


def run_tvm(graph: "onnx graph", name: str = 'Untitled'):
    """run tvm on an onnx model"""

    print("-" * 40)
    t1 = time.perf_counter()
    print("Optimized graph {} in ".format(name))

    # load
    mod, params = relay.frontend.from_onnx(graph)
    t2 = time.perf_counter()

    # build
    # target = "cuda"
    target = "cuda -libs=cudnn"
    # target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    t3 = time.perf_counter()

    timing_number = 5
    timing_repeat = 1

    res = module.benchmark(dev, number=timing_number, repeat=timing_repeat)
    print(res)
    print(res.mean)
    t4 = time.perf_counter()
    print("Time taken TVM load: {:.2f} seconds".format(t2 - t1))
    print("Time taken TVM build: {:.2f} seconds".format(t3 - t2))
    print("Time taken TVM benchmark: {:.2f} seconds".format(t4 - t3))


def main(args):
    graph_name, graph = load_graph_from_file(FLAGS.path)
    graphs = [(graph_name, graph)]

    for current_graph_file, current_graph in graphs:
        run_tvm(current_graph, current_graph_file)


if __name__ == '__main__':
    app.run(main)
