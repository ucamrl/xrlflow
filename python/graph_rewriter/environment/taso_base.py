import numpy as np
from functools import partial
from typing import List

import jax.numpy as jnp
import jraph

# this needs to build TASO's python interface
from taso.core import op_table

from graph_rewriter.core import PyRLOptimizer

# Build op table
op_tbl = {}
for num, op_str in enumerate(sorted(op_table.values())):
    op_tbl[op_str] = num
op_tbl["Unknown"] = len(op_tbl.keys())
num_ops = len(op_tbl.keys())
inverse_tbl = {v: k for k, v in op_tbl.items()}


class _BaseEnvironment:

    def __init__(self, num_locations, real_measurements, reward_function,
                 node_cost_model):
        """
        Args:
            num_locations (int):
                number of possible locations to apply Graph Xfers
            real_measurements (bool): no use
            reward_function (Callable): A custom reward function
        """
        self.graph = None
        self.rl_opt = None

        self.locations = None
        self.xfer_graphs = None

        self.time_step = 0
        self.initial_runtime = 0.0
        self.last_runtime = 0.0
        self.measurement_info = dict()
        self.num_locations = num_locations
        self.last_costs = dict(runtime=0.0,
                               flops=0.0,
                               mem_acc=0.0,
                               num_kernels=0.0)

        self.real_measurements = real_measurements
        self.real_measurements_runtime = None
        self.last_measured_runtime = 0
        self.custom_reward = reward_function
        self.node_cost_model = node_cost_model

    def set_graph(self, graph):
        # this graph is TASO's graph, defined under the python folder
        self.graph = graph
        self.rl_opt = PyRLOptimizer(graph)

    def get_pre_process_graph(self):
        return self.rl_opt.get_pre_process_graph()

    def _eval_cur_graph(self, verbose=False):
        """only for testing"""
        return self.rl_opt.eval_cur_graph(verbose)

    def get_cost(self, real_measurement=None):
        """the same as self.graph.cost()"""
        if self.real_measurements or real_measurement:
            # return self.rl_opt.get_measured_runtime(self.graph)
            raise ValueError("not yet support")
        else:
            return self.rl_opt.get_cost()

    def eval_cur_graph_safe(self, verbose=False):
        return self.rl_opt.eval_cur_graph_safe(verbose)

    def eval_cur_no_pre_process_safe(self, verbose=False):
        return self.rl_opt.eval_cur_no_pre_process_safe(verbose)

    def reset(self):
        if not self.rl_opt:
            raise ValueError("Set graph first.")
        self.rl_opt.reset()
        # only run once
        if self.real_measurements_runtime is None:
            self.real_measurements_runtime = self.eval_cur_graph_safe()

        self.last_runtime = self.initial_runtime = self.get_cost()
        self.time_step = 0
        self.last_measured_runtime = self.real_measurements_runtime
        return self.build_state()

    def build_state(self):
        # list[int],
        # each int indicate number of locations that this xfer can apply
        self.locations = self.rl_opt.get_available_locations()

        # list[list[taso.core.PyGraph object]],
        # the result graph after applying the xfer
        self.xfer_graphs = self.rl_opt.get_xfer_graphs()

        assert len(self.locations) == len(
            self.xfer_graphs), "xfer_graphs len != locations len"

        # Xfer mask; list[bool]
        xfer_mask = jnp.asarray(self.locations).astype(bool).astype(int)
        xfer_mask = jnp.append(xfer_mask, 1)  # add one last action

        # Main graphnet tuple
        graph_tuple = graph_to_graphnet_tuple(
            self.graph,
            op_runtime_callback=self.rl_opt.get_op_runtime,
            node_cost_model=self.node_cost_model)

        # Sub graphnet tuple
        xfer_tuples = []
        location_masks = []
        # [[self.graph]] is the dummy no-op embedding
        for xfer in self.xfer_graphs + [[self.graph]]:
            len_xfer = len(xfer)
            assert (
                self.num_locations > len_xfer
            ), f"num_locations {self.num_locations} < len_xfer {len_xfer}"

            # xg is a mutated graph if apply xfer to a location index
            xfer_graphs = []
            for xg in xfer:
                g = graph_to_graphnet_tuple(
                    xg,
                    op_runtime_callback=partial(
                        self.rl_opt.get_op_runtime_for_graph, xg),
                    node_cost_model=self.node_cost_model)
                xfer_graphs.append(g)

            # embed the concat graph into a GraphTuple
            # if no, append None
            # if this_xfer_tuple has graph, pad to constant
            if len(xfer_graphs) == 0:
                xfer_tuples.append(None)
            else:
                this_xfer_tuple = jraph.batch(xfer_graphs)

                # pad to constant num_of_graph
                pad_nodes_to = jnp.sum(this_xfer_tuple.n_node) + 1
                # edge doesn't need + 1
                pad_edges_to = jnp.sum(this_xfer_tuple.n_edge)
                pad_graphs_to = self.num_locations
                # padded_graph has multiple graph instances
                padded_graph = jraph.pad_with_graphs(this_xfer_tuple,
                                                     pad_nodes_to,
                                                     pad_edges_to,
                                                     pad_graphs_to)
                xfer_tuples.append(padded_graph)

            num_locations = min(self.num_locations, len_xfer)

            # pad location_mask -> self.num_locations
            location_mask = [1] * num_locations + [0] * (self.num_locations -
                                                         num_locations)
            location_masks.append(location_mask)

        # location_masks:: type: np.ndarray -
        # shape: [len(self.xfer_graphs), self.num_locations]
        location_masks = jnp.asarray(location_masks)

        return {
            "graph": graph_tuple,
            "mask": xfer_mask,
            "xfers": xfer_tuples,  # list[GraphsTuple]; len == len(mask)
            "candidates_mask": location_masks
        }

    def get_num_actions(self):
        return self.rl_opt.get_num_xfers()

    def step(self, actions):
        pass

    def get_available_xfers(self) -> List[int]:
        return self.rl_opt.get_num_xfers()

    def get_available_locations(self) -> List[int]:
        return self.rl_opt.get_available_locations()

    def viz_dataflow_graph(self, graph: jraph.GraphsTuple, fn: str):
        import graphviz
        inf = float(graph.globals)
        inf = f"{inf:.4f}"
        g = graphviz.Digraph("dataflow",
                             graph_attr={
                                 "label": inf,
                                 "fontsize": "40"
                             },
                             format="png",
                             filename=fn)
        gid = 0  # global id

        # add node
        m = {}
        for node_idx, node in enumerate(graph.nodes):
            node_type_idx = int(node[0])
            if node_type_idx in inverse_tbl:
                node_type = inverse_tbl[node_type_idx]
            else:
                node_type = "Unknown"
            node_perf = float(node[1])
            node_perf = f"{node_perf:.2f}"
            node_name = f"gid-{gid}-{node_type}_{node_perf}"
            gid += 1
            m[node_idx] = node_name
            g.node(node_name, node_name, **{"shape": "circle"})

        # add edge
        for sender, receiver in zip(graph.senders, graph.receivers):
            int_sender = int(sender)
            int_receiver = int(receiver)
            # ignore self edges
            if int_sender == int_receiver:
                continue
            sender_name = m[int_sender]
            receiver_name = m[int_receiver]
            g.edge(sender_name, receiver_name)

        g.render()


def graph_to_graphnet_tuple(graph,
                            op_runtime_callback=lambda guid: 0.0,
                            max_input_dims=4,
                            node_cost_model=True):
    """
    Args:
        graph: TASO graph, defined under TASO's python folder
    Returns:
        GraphsTuple OR nodes, edges, globals, receivers, senders, n_node, n_edge
    """
    # this calls TASO's total_cost
    # return the sum over the runtime of all Ops of the graph
    if node_cost_model:
        globals = jnp.asarray([[graph.cost()]], dtype=np.float32)
    else:
        globals = jnp.asarray([[0.]], dtype=np.float32)

    guid_to_id = {}

    nodes = {}  # k: node id - v: [OpType index, Op runtime]
    edges = {}  # k: edge id - v: edge's shape up to max_input_dims
    receivers = {}  # k: edge id - v: node id
    senders = {}  # k: edge id - v: node id

    current_edge_id = 0
    # first populate guid_to_id
    # list[dict[str, int]]
    op_list = graph.get_operator_list()
    for current_node_id, node in enumerate(op_list):
        node_guid = node['guid']
        guid_to_id[node_guid] = current_node_id

    # build features
    for current_node_id, node in enumerate(op_list):
        try:  # e.g. enlarge op is missing from op table, catch assertion error
            # exception raised by taso core
            str_type = graph.get_operator_type(node)
            op_index_type = op_tbl[str_type]
        except AssertionError:
            str_type = "Unknown"
            op_index_type = num_ops - 1

        # node embedding
        # get the Op's runtime by its guid
        node_val = one_hot_encode_node(op_index_type)
        if node_cost_model:
            perf = op_runtime_callback(node)
            node_val.append(perf)
        nodes[current_node_id] = node_val

        # loop through input edges (edges are incoming tensors)
        for idx, edge in enumerate(graph.get_input_edges(node)):
            sender_node = edge['srcOp']
            sender_id = sender_node['guid']
            if sender_id not in guid_to_id:
                # input and weight are not counted
                continue

            # Edge embedding
            # use the node and input edge's idx
            # to get this node's [idx]-th incoming tensor's rank
            # in each dimension
            # input_dims: list[int],
            # e.g. [2, 2, 2] = 3 rank, each dim has 2 element
            input_dims = graph.get_input_dims(node, idx)
            edge_val = []
            # XXX normalise by max tensor 4096?
            for val in input_dims:
                edge_val.append(val / 4096)
            while len(edge_val) < max_input_dims:
                # left append
                edge_val.insert(0, 0)
            edges[current_edge_id] = edge_val

            # NOTE: This is a guid and has to be re-wired
            senders[current_edge_id] = sender_id
            receivers[current_edge_id] = current_node_id
            current_edge_id += 1

    # convert sender guid to id
    for edge_id, sender_id in senders.items():
        assert (sender_id in guid_to_id), f"{sender_id} not exists"
        senders[edge_id] = guid_to_id[sender_id]

    num_node = len(nodes)
    n_node = len(nodes)
    n_edge = len(edges)
    # convert to list
    sorted_node_id = sorted(nodes.keys())
    sorted_edge_id = sorted(edges.keys())
    # list[node_feat]
    nodes = [nodes[node_id] for node_id in sorted_node_id]
    # list[edge_feat]
    edges = [edges[edge_id] for edge_id in sorted_edge_id]
    # list[sender_id]
    senders = [senders[edge_id] for edge_id in sorted_edge_id]
    # list[receiver_id]
    receivers = [receivers[edge_id] for edge_id in sorted_edge_id]

    # to jnp
    senders = jnp.asarray(senders, dtype=jnp.int32)
    receivers = jnp.asarray(receivers, dtype=jnp.int32)

    # add self edges
    # NOTE: add self edge or not
    # for GAT and GCN, add_self_edges is recommanded
    # for generic GraphNetwork, add_self_edges also means add edge features
    receivers, senders = add_self_edges_fn(receivers, senders, n_node)

    # to jnp
    n_node = jnp.asarray([n_node], dtype=jnp.int32)
    n_edge = jnp.asarray([n_edge], dtype=jnp.int32)

    # node features
    nodes = jnp.asarray(nodes, dtype=jnp.float32)
    # each edge has 4 features; if add self edge,
    # also need to add edge feature
    edges = add_self_edges_feat_fn(edges, num_node, max_edge_features_dim=4)
    edges = jnp.asarray(edges, dtype=jnp.float32)

    return jraph.GraphsTuple(nodes=nodes,
                             edges=edges,
                             globals=globals,
                             receivers=receivers,
                             senders=senders,
                             n_node=n_node,
                             n_edge=n_edge)


def add_self_edges_fn(receivers: jnp.ndarray, senders: jnp.ndarray,
                      total_num_nodes: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Adds self edges. Assumes self edges are not in the graph yet."""
    assert (len(receivers.shape) == 1), "rank should be 1"
    assert (len(senders.shape) == 1), "rank should be 1"
    receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)),
                                axis=0)
    senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)), axis=0)
    return receivers, senders


def add_self_edges_feat_fn(
        edges: list[list[float]], total_num_nodes: int,
        max_edge_features_dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    # total_num_nodes self edge is added
    for i in range(total_num_nodes):
        edges.append([0.] * max_edge_features_dim)  # 0 for self edge?
    return edges


def one_hot_encode_node(node_idx: int) -> list[int]:
    """one-hot encoding"""
    feat = [0] * num_ops
    feat[node_idx] = 1
    return feat
