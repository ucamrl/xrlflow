import onnx
import taso as ts

# https://github.com/kuprel/min-dalle
text_len = 16
hidden_dims = 768
num_layer = 8
head = 12
mlp_hidden = 2048


def build_dalle():
    graph = ts.new_graph()
    graph_in = graph.new_input(dims=(text_len, hidden_dims))
    graph_in = graph.relu(graph_in)
    t = graph_in

    masks = []
    for i in range(num_layer):
        masks.append(graph.new_input(dims=(head, text_len, text_len)))

    for m in range(num_layer):
        print(f"layer {m+1}")
        # text_len, hidden_dims
        attn_out = _self_attention(graph, t, head)
        res_out = graph.add(t, attn_out)
        mlp_out = ff(graph, res_out)
        t = graph.add(res_out, mlp_out)
    return graph


def ff(graph, input):
    l1 = graph.new_weight(dims=(hidden_dims, mlp_hidden))
    l2 = graph.new_weight(dims=(hidden_dims, mlp_hidden))
    l4 = graph.new_weight(dims=(mlp_hidden, hidden_dims))

    out1 = graph.matmul(input, l1)
    out1 = graph.relu(out1)
    out2 = graph.matmul(input, l2)
    out = graph.mul(out1, out2)
    out = graph.matmul(out, l4)
    return out


def _self_attention(graph, input, heads):
    embed = input.dim(1)
    dk = embed // heads
    assert embed % heads == 0

    print(text_len, heads, dk)

    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(embed, embed)))

    # compute query, key, value tensors
    print("try ")
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    print("key ok")
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(text_len, heads, dk))
    k = graph.reshape(k, shape=(text_len, heads, dk))
    v = graph.reshape(v, shape=(text_len, heads, dk))
    print(q.nDim, k.nDim, v.nDim)
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(1, 0, 2), shuffle=True)
    k = graph.transpose(k, perm=(1, 2, 0), shuffle=True)
    v = graph.transpose(v, perm=(1, 0, 2), shuffle=True)
    # perform matrix multiplications
    logits = graph.matmul(q, k)
    print("logit: ", logits.nDim)
    for i in range(logits.nDim):
        print(logits.dim(i))
    mask = graph.new_weight(dims=(heads, text_len, text_len))
    logits = graph.add(logits, mask)
    print("mask logit: ", logits.nDim)
    for i in range(logits.nDim):
        print(logits.dim(i))
    for i in range(v.nDim):
        print(v.dim(i))

    output = graph.matmul(logits, v)
    print("attention ok")
    # transpose the output back
    output = graph.transpose(output, perm=(1, 0, 2), shuffle=True)
    output = graph.reshape(output, shape=(text_len, embed))

    # a final linear layer
    linear = graph.new_weight(dims=(embed, embed))
    output = graph.matmul(output, linear)
    print("linear ok")
    return output


if __name__ == '__main__':
    # from xflowrl.graphs.util import export_onnx
    built_graph = build_dalle()
    onnx_model = ts.export_onnx(built_graph)
    onnx.save(onnx_model, 'output-data/graphs/mini-dalle-2.onnx')
