import onnx
import taso as ts

num_patches = 64
hidden_dims = 768
num_layer = 12
head = 12
mlp_hidden = 3072


def build_vit():
    graph = ts.new_graph()
    graph_in = graph.new_input(dims=(num_patches, hidden_dims))
    graph_in = graph.relu(graph_in)
    t = graph_in

    for m in range(num_layer):
        attn_out = _attention(graph, t, head)
        res_out = graph.add(t, attn_out)
        mlp_out = mlp_block(graph, res_out)
        t = graph.add(res_out, mlp_out)
    return graph


def mlp_block(graph, input):
    l1 = graph.new_weight(dims=(hidden_dims, mlp_hidden))
    l2 = graph.new_weight(dims=(mlp_hidden, hidden_dims))
    out = graph.matmul(input, l1)
    out = graph.matmul(out, l2)
    return out


def _attention(graph, input, heads):
    hidden_dims = input.dim(1)
    dk = hidden_dims // heads
    assert hidden_dims % heads == 0

    print(num_patches, heads, dk)

    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(hidden_dims, hidden_dims)))
    # compute query, key, value tensors
    print("try ")
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    print("key ok")
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(num_patches, heads, dk))
    k = graph.reshape(k, shape=(num_patches, heads, dk))
    v = graph.reshape(v, shape=(num_patches, heads, dk))
    print(q.nDim, k.nDim, v.nDim)
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(1, 0, 2), shuffle=True)
    k = graph.transpose(k, perm=(1, 0, 2), shuffle=True)
    v = graph.transpose(v, perm=(1, 0, 2), shuffle=True)
    # perform matrix multiplications
    logits = graph.matmul(q, k)
    output = graph.matmul(logits, v)
    print("attention ok")
    # transpose the output back
    output = graph.transpose(output, perm=(1, 0, 2), shuffle=True)
    output = graph.reshape(output, shape=(num_patches, hidden_dims))

    # a final linear layer
    linear = graph.new_weight(dims=(hidden_dims, hidden_dims))
    # output = graph.matmul(input, linear)  # as in taso's ae
    output = graph.matmul(output, linear)  # v2
    print("linear ok")
    return output


if __name__ == '__main__':
    # from xflowrl.graphs.util import export_onnx
    built_graph = build_vit()
    onnx_model = ts.export_onnx(built_graph)
    onnx.save(onnx_model, 'output-data/graphs/vit-base.onnx')
