import onnx
import taso as ts

seq_length = 64
hidden_dims = 1024


def build_graph_bert():
    graph = ts.new_graph()
    graph_in = graph.new_input(dims=(seq_length, hidden_dims))
    graph_in = graph.relu(graph_in)
    t = graph_in
    for m in range(8):
        t = _attention(graph, t, 16)
    return graph


def _attention(graph, input, heads):
    d_model = input.dim(1)
    d_k = d_model // heads
    assert input.dim(1) % heads == 0
    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(d_model, d_model)))
    # compute query, key, value tensors
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(64, 16, 64))
    k = graph.reshape(k, shape=(64, 16, 64))
    v = graph.reshape(v, shape=(64, 16, 64))
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(1, 0, 2), shuffle=True)
    k = graph.transpose(k, perm=(1, 0, 2), shuffle=True)
    v = graph.transpose(v, perm=(1, 0, 2), shuffle=True)
    # perform matrix multiplications
    logits = graph.matmul(q, k)
    output = graph.matmul(logits, v)
    # transpose the output back
    output = graph.transpose(output, perm=(1, 0, 2), shuffle=True)
    output = graph.reshape(output, shape=(64, 1024))

    # a final linear layer
    linear = graph.new_weight(dims=(d_model, d_model))
    output = graph.matmul(input, linear)
    return output


if __name__ == '__main__':
    built_graph = build_graph_bert()
    onnx_model = ts.export_onnx(built_graph)
    onnx.save(onnx_model, 'output-data/graphs/bert.onnx')
