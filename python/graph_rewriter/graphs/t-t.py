import onnx
import taso as ts

# bs = 2 # set batch size = 1
seq_len = 80
input_dim = 16

audio_num_layer = 12
label_num_layer = 2

num_head = 8
# ff_dim = 2048
mlp_hidden = 2048
model_dim = 512

vocab_size = 10

"""transformer_transducer
as in https://arxiv.org/abs/2010.11395,
implementation adopted from
https://github.com/upskyy/Transformer-Transducer/tree/main/transformer_transducer
"""


def build_tt():
    graph = ts.new_graph()

    graph_in = graph.new_input(dims=(seq_len, model_dim))
    l1 = graph.new_weight(dims=(model_dim, input_dim))
    audio_in = graph.matmul(graph_in, l1)
    label_in = graph_in

    # out: [seq_len, model_dim]
    audio_out = audio_encoder(graph, audio_in, num_head, model_dim)
    for i in range(audio_out.nDim):
        print(audio_out.dim(i), end=", ")
    print("audio ok")
    # out: [seq_len, model_dim]
    label_out = label_encoder(graph, label_in, num_head, model_dim)
    for i in range(label_out.nDim):
        print(label_out.dim(i), end=", ")
    print("label ok")

    # _ = graph.add(audio_out, label_out)
    # out: [seq_len, vocab_size]
    _ = joint_net(graph, audio_out, label_out)

    return graph


def audio_encoder(graph, input, head, model_dim):
    l1 = graph.new_weight(dims=(input_dim, model_dim))
    out = graph.matmul(input, l1)
    t = out  # shape[seq_len, model_dim] or [bs, seq_len, model_dim]
    # dropout is ignored during inference

    for m in range(audio_num_layer):
        attn_out = _attention(graph, t, head, model_dim)
        res_out = graph.add(t, attn_out)
        mlp_out = mlp_block(graph, res_out)
        t = graph.add(res_out, mlp_out)
    return t


def label_encoder(graph, input, head, model_dim):
    t = input
    for m in range(label_num_layer):
        attn_out = _attention(graph, t, head, model_dim)
        # print("atten ok")
        res_out = graph.add(t, attn_out)
        mlp_out = mlp_block(graph, res_out)
        t = graph.add(res_out, mlp_out)
        # print(f"out ok")
    return t


def joint_net(graph, in1, in2):
    out = graph.concat(1, [in1, in2])  # shape: [seq_len, model_dim*2]
    l1 = graph.new_weight(dims=(model_dim*2, model_dim))
    out = graph.matmul(out, l1)
    out = graph.relu(out)
    l2 = graph.new_weight(dims=(model_dim, vocab_size))
    out = graph.matmul(out, l2)
    return out


def mlp_block(graph, input):
    l1 = graph.new_weight(dims=(model_dim, mlp_hidden))
    l2 = graph.new_weight(dims=(mlp_hidden, model_dim))
    out = graph.matmul(input, l1)
    out = graph.relu(out)
    out = graph.matmul(out, l2)
    return out


def _attention(graph, input, heads, model_dim):
    dk = model_dim // heads  # 64
    print("md: ", model_dim)
    assert model_dim % heads == 0

    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(model_dim, model_dim)))
    # compute query, key, value tensors
    # print("try ")
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    # print("key ok")
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(seq_len, heads, dk))
    k = graph.reshape(k, shape=(seq_len, heads, dk))
    v = graph.reshape(v, shape=(seq_len, heads, dk))
    # print(q.nDim, k.nDim, v.nDim)
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(1, 0, 2), shuffle=True)
    k = graph.transpose(k, perm=(1, 2, 0), shuffle=True)
    v = graph.transpose(v, perm=(1, 0, 2), shuffle=True)
    # perform matrix multiplications
    logits = graph.matmul(q, k)
    for i in range(logits.nDim):
        print(logits.dim(i), end=", ")
    print()
    output = graph.matmul(logits, v)
    # print("attention ok")
    # transpose the output back
    output = graph.transpose(output, perm=(1, 0, 2), shuffle=True)
    output = graph.reshape(output, shape=(seq_len, model_dim))

    # a final linear layer
    linear = graph.new_weight(dims=(model_dim, model_dim))
    output = graph.matmul(output, linear)  # v2
    # print("linear ok")
    return output


if __name__ == '__main__':
    # from xflowrl.graphs.util import export_onnx
    built_graph = build_tt()
    onnx_model = ts.export_onnx(built_graph)
    onnx.save(onnx_model, 'output-data/graphs/tt.onnx')
