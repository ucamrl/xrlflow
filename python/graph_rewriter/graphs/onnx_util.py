from taso import xf_operators, core, InputNotFoundError, _check_output
from onnx import numpy_helper


def load_onnx_model(model):
    graph = core.PyGraph()
    tensors = dict()
    for t in model.graph.input:
        dims = list()
        for d in t.type.tensor_type.shape.dim:
            dims.append(d.dim_value)
        weight_data = None
        for weight in model.graph.initializer:
            if (weight.name == t.name):
                weight_data = numpy_helper.to_array(weight)
        # We classify an input to be a pure input if we cannot find its weights
        if weight_data is None:
            tensors[t.name] = graph.new_input(dims=tuple(dims))
        else:
            tensors[t.name] = graph.new_weight(dims=tuple(dims),
                                               data=weight_data)

    # Add initializers not in the inputs
    for weight in model.graph.initializer:
        if weight.name not in tensors:
            if weight.dims:
                dims = list(weight.dims)
                weight_data = numpy_helper.to_array(weight)
                tensors[weight.name] = graph.new_weight(dims=tuple(dims),
                                                        data=weight_data)

    # Reorder nodes to satisfy data dependencies
    tensor_owner = dict()
    name_to_op = dict()
    idx = 0
    for op in model.graph.node:
        # Assign a name to the node if empty
        if len(op.name) == 0:
            op.name = op.op_type + '_' + str(idx)
        idx += 1
        name_to_op[op.name] = op
        for output in op.output:
            tensor_owner[output] = op.name
    out_edges = dict()
    dependents = dict()
    node_list = list()
    for op in model.graph.node:
        dependents[op.name] = 0
        for input in op.input:
            if input in tensor_owner:
                dependents[op.name] += 1
                input_node = tensor_owner[input]
                if input_node not in out_edges:
                    out_edges[input_node] = list()
                out_edges[input_node].append(op.name)
        if dependents[op.name] == 0:
            node_list.append(op.name)
    idx = 0
    while idx < len(node_list):
        opname = node_list[idx]
        if opname in out_edges:
            for e in out_edges[opname]:
                dependents[e] -= 1
                if dependents[e] == 0:
                    node_list.append(e)
        idx += 1
    assert len(node_list) == len(
        model.graph.node), "Internal error when reording ONNX operators"

    # Add nodse into TASO graph
    cnt = 0
    for opname in node_list:
        op = name_to_op[opname]
        # print(cnt, op.op_type, opname)
        cnt += 1
        if op.op_type in xf_operators:
            try:
                outputs = xf_operators[op.op_type](op, graph, tensors,
                                                   model.graph.initializer)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                assert len(outputs) == len(
                    op.output), "Number of output tensors mismatch"
                for i in range(len(outputs)):
                    assert _check_output(outputs[i], op.output[i])
                    tensors[op.output[i]] = outputs[i]
            except InputNotFoundError:
                print(
                    "Cannot find input tensor for operator: name({}) type({}) (Skipped)"
                    .format(opname, op.op_type))
                continue
        else:
            print("Found unsupported ONNX operator: {} (Skipped)".format(
                op.op_type))
            continue
    return graph
