import onnx
import taso as ts
from xflowrl.graphs.bert import build_graph_bert
from xflowrl.graphs.inceptionv3 import build_graph_inception_v3
from xflowrl.graphs.nasnet import build_graph_nasnet


def export_onnx(graph, file_name):
    onnx_model = ts.export_onnx(graph)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, file_name)


def load_graph_from_file(filename):
    print(f"Loading graph from file: {filename}")
    clean_filename = filename.split('/')[-1].split('.')[0]
    return clean_filename, ts.load_onnx(filename)


def load_graph_by_name(graph_name):
    graphs = {
        'BERT': build_graph_bert,
        'NASnet': build_graph_nasnet,
        'InceptionV3': build_graph_inception_v3
    }
    if graph_name not in graphs:
        raise ValueError(f"Invalid graph name: {graph_name}")
    else:
        print(f"Building graph from name: {graph_name}")
        return graph_name, graphs[graph_name]()


def load_graph(graph):
    try:
        return load_graph_by_name(graph)
    except ValueError:
        return load_graph_from_file(graph)
