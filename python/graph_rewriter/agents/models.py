from typing import Sequence, Optional

import jax.numpy as jnp
from flax import linen as nn
import jraph
from jraph import GraphsTuple

from graph_rewriter.agents.encoder.gnn import GraphNetwork, GAT, GAT_with_global_update, multi_head_GAT, sum_global_layer, GraphNetwork_update_node


# ================================================
# generic graph networks Model
# ================================================
# NNs initilization may be useful
# e.g. last layer of policy net, people dont sigmoid
def default_conv_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.xavier_uniform()


def default_mlp_init(scale: Optional[float] = 0.01):
    return nn.initializers.orthogonal(scale)


def default_logits_init(scale: Optional[float] = 0.01):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    """A flax MLP."""
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate([
                nn.Dense(feat, kernel_init=default_mlp_init())
                for feat in self.features
        ]):
            x = lyr(x)
            # see: https://openreview.net/forum?id=r1etN1rtPB
            # x = nn.relu(x)
            # x = jax.nn.leaky_relu(x)
            x = jnp.tanh(x)
        return x


class MainTwinHeadModel_GraphNet(nn.Module):
    """Critic+Actor for PPO. the GNN is used to encode one-graph"""
    # gnn
    mlp_features: Sequence[int]
    latent_size: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: GraphsTuple):
        # for main model, graph_repr is the global features of that graph
        # the goal is to find the best agnet within the graph itself
        graph = GraphNetwork(self.mlp_features, self.latent_size)(x)
        # shape: [B, embedding size]
        graph_repr = graph.globals

        logits = MLP(self.policy_feature)(graph_repr)
        # shape: [B, num_actions]
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)
        vf = MLP(self.vf_feature)(graph_repr)
        # shape [B, 1];
        vf = nn.Dense(1)(vf)
        # shape [B, ]
        vf = jnp.squeeze(vf)
        return logits, vf


class SubTwinHeadModel_GraphNet(nn.Module):
    """Critic+Actor for PPO. the GNN is used to encode multi-graph"""
    # gnn
    mlp_features: Sequence[int]
    latent_size: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: GraphsTuple):
        graph = GraphNetwork(self.mlp_features, self.latent_size)(x)
        graph_repr = graph.globals
        final_embedding_size = self.mlp_features[-1]
        graph_repr = jnp.reshape(graph_repr,
                                 (-1, self.num_actions, final_embedding_size))

        # logits
        logits = MLP(self.policy_feature)(graph_repr)
        logits = nn.Dense(1, kernel_init=default_logits_init())(logits)

        # batch size is flatten
        # shape [B, num_action]
        logits = jnp.reshape(logits, (-1, self.num_actions))

        # shape [B, num_graph_in_GraphsTuple, 1];
        vf = MLP(self.vf_feature)(graph_repr)
        vf = nn.Dense(1)(vf)

        # flatten graph global features
        # shape [B, 1]
        vf = jnp.reshape(vf, (-1, self.num_actions))
        vf = nn.Dense(1)(vf)

        # shape [B, ]
        vf = jnp.squeeze(vf)
        return logits, vf


# ================================================
# GAT Model
# ================================================
def _gen_segment_ids(g: GraphsTuple) -> jnp.ndarray:
    """
    implement segment ids inside a forward maybe not a good idea
        because variable segment num means it is not JITtable

    Returns:
        1D jnp.ndarray represent which nodes are which graph
        [0, 0, 1, 1, 1, ...]
    """
    seg_ids = []
    n_node = g.n_node
    cnt = 0
    for i, num_node in enumerate(n_node):
        for _ in range(num_node):
            seg_ids.append(cnt)
        cnt += 1
    return jnp.array(seg_ids, dtype=jnp.int32)


class MainTwinHeadModel_GAT(nn.Module):
    """Critic+Actor for PPO. the GNN is used to encode one-graph"""
    # gnn
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: GraphsTuple):
        # for main model, graph_repr is the global features of that graph
        # the goal is to find the best agnet within the graph itself

        # GAT layer; only update node feat
        graph = GAT(self.gat_attn_mlp, self.gat_node_update_mlp,
                    self.message_passing_steps, self.gat_global_update_mlp)(x)

        # shape: [n_node, node embedding size];
        # where n_node is the num of node for all graphs in the batch
        graph_repr = graph.nodes

        # shape: [B, node embedding size];
        # NOTE: if want JIT, need to provide static segment num
        # see: https://github.com/deepmind/jraph/blob/master/jraph/_src/utils.py
        # this average node embedding to represent a graph
        segment_ids = _gen_segment_ids(x)
        graph_repr = jraph.segment_mean(graph_repr, segment_ids)

        # action logits
        logits = MLP(self.policy_feature)(graph_repr)
        # shape: [B, num_actions]
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)

        # value function
        vf = MLP(self.vf_feature)(graph_repr)
        # shape [B, 1];
        vf = nn.Dense(1)(vf)
        # shape [B, ]
        vf = jnp.squeeze(vf)
        return logits, vf


class SubTwinHeadModel_GAT(nn.Module):
    """Critic+Actor for PPO. the GNN is used to encode multi-graph"""
    # gnn
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: GraphsTuple):
        # for sub model, each graph in the batch has a multiple-graphs GraphsTuple
        # so the goal is pick the best among multiple graphs
        graph = GAT(self.gat_attn_mlp, self.gat_node_update_mlp,
                    self.message_passing_steps, self.gat_global_update_mlp)(x)
        # shape: [total n_node, node embedding size];
        # where n_node is the num of node for all graphs in the batch
        graph_repr = graph.nodes
        node_embedding_size = graph_repr.shape[1]
        segment_ids = _gen_segment_ids(x)

        # shape: [actual_candidate , node embedding size];
        # average node embedding whose are within the same graph
        # padded graph has 0 nodes, so segment_mean will not apply
        graph_repr = jraph.segment_mean(graph_repr, segment_ids)
        actual_candidate = graph_repr.shape[0]

        # Pad; only pad dim 0
        pad_len = self.num_actions - actual_candidate  # data-dependent
        graph_repr = jnp.pad(graph_repr, [(0, pad_len), (0, 0)],
                             constant_values=0)

        # DONT FLATTEN
        # shape: [B, num_candidate , node embedding size];
        # graph_repr = jnp.reshape(graph_repr,
        #                          (-1, self.num_actions, node_embedding_size))

        # JUST FLATTEN the node feat?
        # shape: [actual_candidate * node embedding size];
        graph_repr = jnp.reshape(graph_repr,
                                 (1, self.num_actions * node_embedding_size))

        # logits
        logits = MLP(self.policy_feature)(graph_repr)
        # shape: [B, num_candidate];
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)
        # batch size is flatten
        # shape [B, num_candidate]

        # value
        # shape [1, 1];
        vf = MLP(self.vf_feature)(graph_repr)
        vf = nn.Dense(1)(vf)
        # shape [1, ]
        vf = jnp.reshape(vf, (-1))

        return logits, vf


# ================================================
# GAT global Model
# ================================================
class MainTwinHeadModel_GAT_global(nn.Module):
    """Critic+Actor for PPO. the GNN is used to encode one-graph"""
    # gnn
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: GraphsTuple):
        # for main model, graph_repr is the global features of that graph
        # the goal is to find the best agnet within the graph itself
        # GAT layer; followed by global update
        graph = GAT_with_global_update(self.gat_attn_mlp,
                                       self.gat_node_update_mlp,
                                       self.message_passing_steps,
                                       self.gat_global_update_mlp)(x)

        # IF PAD
        # globals shape: [B * 2, global embedding size];
        # each graphsTuple has 2 graph, 1 real graph and 1 pad graph
        # graph_repr = graph.globals
        # global_embedding_size = graph_repr.shape[1]
        # # JUST flatten the padding
        # # globals shape: [B, 2 * global embedding size];
        # graph_repr = jnp.reshape(graph_repr, (-1, 2 * global_embedding_size))

        # IF NO PAD
        # globals shape: [B, global embedding size];
        graph_repr = graph.globals

        # action logits
        logits = MLP(self.policy_feature)(graph_repr)
        # shape: [B, num_actions]
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)

        # value function
        vf = MLP(self.vf_feature)(graph_repr)
        # shape [B, 1];
        vf = nn.Dense(1)(vf)
        # shape [B, ]
        vf = jnp.squeeze(vf)
        return logits, vf


class SubTwinHeadModel_GAT_global(nn.Module):
    """Critic+Actor for PPO. the GNN is used to encode multi-graph"""
    # gnn
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: GraphsTuple):
        # for sub model, each graph in the batch has a multiple-graphs GraphsTuple
        # so the goal is pick the best among multiple graphs
        graph = GAT_with_global_update(self.gat_attn_mlp,
                                       self.gat_node_update_mlp,
                                       self.message_passing_steps,
                                       self.gat_global_update_mlp)(x)
        # NOTE: for now, num_candidates = self.num_actions
        # shape: [B * num_candidates, globals embedding size];
        graph_repr = graph.globals
        global_embedding_size = graph.globals.shape[1]

        # JUST FLATTEN the global feat?
        # shape: [B,  num_candidate * global embedding size];
        graph_repr = jnp.reshape(
            graph_repr, (-1, self.num_actions * global_embedding_size))

        # logits
        logits = MLP(self.policy_feature)(graph_repr)
        # shape: [B, num_candidate];
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)
        # batch size is flatten
        # shape [B, num_candidate]

        # value
        # shape [1, 1];
        vf = MLP(self.vf_feature)(graph_repr)
        vf = nn.Dense(1)(vf)
        # shape [1, ]
        vf = jnp.reshape(vf, (-1))

        return logits, vf


class MainTwinHeadModel_GAT_global_v2(nn.Module):
    # gnn
    num_head: int
    hidden_dim: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: GraphsTuple, batch_size: int):
        """batch_size is not needed for main model,
        but to keep signature consistent"""
        # for main model, graph_repr is the global features of that graph
        # the goal is to find the best agnet within the graph itself

        # logits
        logits_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                      self.message_passing_steps)(x)
        logits_graph = sum_global_layer(
            self.gat_global_update_mlp)(logits_graph)
        # globals shape: [B, global embedding size];
        graph_repr = logits_graph.globals
        # action logits
        logits = MLP(self.policy_feature)(graph_repr)
        # shape: [B, num_actions]
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)

        # vf
        vf_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                  self.message_passing_steps)(x)
        vf_graph = sum_global_layer(self.gat_global_update_mlp)(vf_graph)
        # globals shape: [B, global embedding size];
        graph_repr = vf_graph.globals
        # value function
        vf = MLP(self.vf_feature)(graph_repr)
        # shape [B, 1];
        vf = nn.Dense(1, kernel_init=default_logits_init())(vf)
        # shape [B, ]
        vf = jnp.squeeze(vf)
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class SubTwinHeadModel_GAT_global_v2(nn.Module):
    # gnn
    num_head: int
    hidden_dim: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]

    @nn.compact
    def __call__(self, x: GraphsTuple, batch_size: int):
        # for sub model,
        # each graph in the batch has a multiple-graphs GraphsTuple
        # so the goal is pick the best among multiple graphs

        # logits
        logits_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                      self.message_passing_steps)(x)
        logits_graph = sum_global_layer(
            self.gat_global_update_mlp)(logits_graph)
        # globals shape: [B * num_candidate, global embedding size];
        embedding = logits_graph.globals
        # action logits
        # embedding = MLP(self.policy_feature)(embedding)
        # shape: [B * num_candidate, 1]
        logits = nn.Dense(1, kernel_init=default_logits_init())(embedding)
        # shape: [B, num_candidate]
        logits = jnp.reshape(logits, (batch_size, -1))

        # vf
        vf_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                  self.message_passing_steps)(x)
        vf_graph = sum_global_layer(self.gat_global_update_mlp)(vf_graph)
        # globals shape: [B, num_candidate * global embedding size];
        graph_repr = jnp.reshape(vf_graph.globals, (batch_size, -1))
        # value function
        # vf = MLP(self.vf_feature)(graph_repr)
        # shape [B, 1];
        vf = nn.Dense(1, kernel_init=default_logits_init())(graph_repr)
        # shape [B, ]
        vf = jnp.squeeze(vf)
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class PPO_GAT_model(nn.Module):
    # gnn
    num_head: int
    hidden_dim: int
    message_passing_steps: int
    gat_global_update_mlp: int

    policy_feature: list[int]
    vf_feature: list[int]

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers: GraphsTuple,
                 batch_size: int):
        # logits
        logits_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                      self.message_passing_steps)(xfers)
        logits_graph = sum_global_layer(
            self.gat_global_update_mlp)(logits_graph)
        # globals shape: [B * num_candidate, global embedding size];
        embedding = logits_graph.globals
        # action logits
        embedding = MLP(self.policy_feature)(embedding)
        # shape: [B * num_candidate, 1]
        logits = nn.Dense(1, kernel_init=default_logits_init())(embedding)
        # shape: [B, num_candidate]
        logits = jnp.reshape(logits, (batch_size, -1))

        # vf
        vf_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                  self.message_passing_steps)(graph)
        vf_graph = sum_global_layer(self.gat_global_update_mlp)(vf_graph)
        # globals shape: [B, global embedding size];
        vf = vf_graph.globals
        vf = MLP(self.vf_feature)(vf)
        # value function
        vf = nn.Dense(1, kernel_init=default_logits_init())(vf)
        # shape [B, ]
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class PPO_GN_model(nn.Module):
    # gnn
    mlp_features: Sequence[int]
    # GAT
    num_head: int
    hidden_dim: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers: GraphsTuple,
                 batch_size: int):
        # logits
        logits_graph = GraphNetwork_update_node(self.mlp_features)(xfers)
        logits_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                      self.message_passing_steps)(logits_graph)
        logits_graph = sum_global_layer(
            self.gat_global_update_mlp)(logits_graph)
        # globals shape: [B * num_candidate, global embedding size];
        embedding = logits_graph.globals
        # action logits
        embedding = MLP(self.policy_feature)(embedding)
        # shape: [B * num_candidate, 1]
        logits = nn.Dense(1, kernel_init=default_logits_init())(embedding)
        # shape: [B, num_candidate]
        logits = jnp.reshape(logits, (batch_size, -1))

        # vf
        vf_graph = GraphNetwork_update_node(self.mlp_features)(graph)
        vf_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                  self.message_passing_steps)(vf_graph)
        vf_graph = sum_global_layer(self.gat_global_update_mlp)(vf_graph)
        # globals shape: [B, global embedding size];
        vf = vf_graph.globals
        vf = MLP(self.vf_feature)(vf)
        # value function
        vf = nn.Dense(1, kernel_init=default_logits_init())(vf)
        # shape [B, ]
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class PPO_GAT_model_v2(nn.Module):
    # gnn
    num_head: int
    hidden_dim: int
    message_passing_steps: int
    gat_global_update_mlp: int

    policy_feature: list[int]
    vf_feature: list[int]

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers: GraphsTuple,
                 batch_size: int):
        """
        graph (the current dataflow graph) is not used
        """
        # logits
        logits_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                      self.message_passing_steps)(xfers)
        logits_graph = sum_global_layer(
            self.gat_global_update_mlp)(logits_graph)
        # globals shape: [B * num_candidate, global embedding size];
        embedding = logits_graph.globals
        # action logits
        embedding = MLP(self.policy_feature)(embedding)
        # shape: [B * num_candidate, 1]
        logits = nn.Dense(1, kernel_init=default_logits_init())(embedding)
        # shape: [B, num_candidate]
        logits = jnp.reshape(logits, (batch_size, -1))

        # vf
        vf_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                  self.message_passing_steps)(xfers)
        vf_graph = sum_global_layer(self.gat_global_update_mlp)(vf_graph)
        # globals shape: [B * num_candidate, global embedding size];
        vf = vf_graph.globals
        vf = MLP(self.vf_feature)(vf)
        # globals shape: [B * num_candidate, 1];
        vf = nn.Dense(1, kernel_init=default_logits_init())(vf)
        # globals shape: [B, num_candidate];
        vf = jnp.reshape(vf, (batch_size, -1))
        # shape [B, ]; TODO which aggregator?
        vf = jnp.sum(vf, axis=1)
        # shape [B, ]
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class PPO_GN_model_v2(nn.Module):
    # gnn
    num_head: int
    hidden_dim: int
    message_passing_steps: int
    gat_global_update_mlp: int

    policy_feature: list[int]
    vf_feature: list[int]

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers: GraphsTuple,
                 batch_size: int):
        """
        graph (the current dataflow graph) is not used
        """
        assert (False)
        # logits
        logits_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                      self.message_passing_steps)(xfers)
        logits_graph = sum_global_layer(
            self.gat_global_update_mlp)(logits_graph)
        # globals shape: [B * num_candidate, global embedding size];
        embedding = logits_graph.globals
        # action logits
        embedding = MLP(self.policy_feature)(embedding)
        # shape: [B * num_candidate, 1]
        logits = nn.Dense(1, kernel_init=default_logits_init())(embedding)
        # shape: [B, num_candidate]
        logits = jnp.reshape(logits, (batch_size, -1))

        # vf
        vf_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                  self.message_passing_steps)(xfers)
        vf_graph = sum_global_layer(self.gat_global_update_mlp)(vf_graph)
        # globals shape: [B * num_candidate, global embedding size];
        vf = vf_graph.globals
        vf = MLP(self.vf_feature)(vf)
        # globals shape: [B * num_candidate, 1];
        vf = nn.Dense(1, kernel_init=default_logits_init())(vf)
        # globals shape: [B, num_candidate];
        vf = jnp.reshape(vf, (batch_size, -1))
        # shape [B, ]; TODO which aggregator?
        vf = jnp.sum(vf, axis=1)
        # shape [B, ]
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class PPO_GAT_model_v3(nn.Module):
    # gnn
    num_head: int
    hidden_dim: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: list[int]
    vf_feature: list[int]
    num_action: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers, batch_size: int):
        # logits
        logits_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                      self.message_passing_steps)(graph)
        logits_graph = sum_global_layer(
            self.gat_global_update_mlp)(logits_graph)
        # globals shape: [B, global embedding size];
        embedding = logits_graph.globals
        # action logits
        embedding = MLP(self.policy_feature)(embedding)
        # shape: [B, num_action]
        logits = nn.Dense(self.num_action,
                          kernel_init=default_logits_init())(embedding)
        # shape: [B, num_action]
        # logits = jnp.reshape(logits, (batch_size, -1))

        # vf
        vf_graph = multi_head_GAT(self.num_head, self.hidden_dim,
                                  self.message_passing_steps)(graph)
        vf_graph = sum_global_layer(self.gat_global_update_mlp)(vf_graph)
        # globals shape: [B, global embedding size];
        vf = vf_graph.globals
        vf = MLP(self.vf_feature)(vf)
        # value function
        # shape: [B, 1]
        vf = nn.Dense(1, kernel_init=default_logits_init())(vf)
        # shape [B, ]
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class PPO_GAT_model_v4(nn.Module):
    # gnn
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers, batch_size: int):
        # each xfers in the batch has a multiple-graphs GraphsTuple
        # so the goal is pick the best among multiple graphs
        # ====================================================
        # NOTE: IMPLICITLY BATCH
        # e.g. batch size = 2, each GraphTuples has 10 graphs(num_candidate)
        # this gives globals: [2*10, globals embedding size]
        graph = GAT_with_global_update(self.gat_attn_mlp,
                                       self.gat_node_update_mlp,
                                       self.message_passing_steps,
                                       self.gat_global_update_mlp)(xfers)
        # NOTE: for now, num_candidates = self.num_actions
        # shape: [B * num_candidates, globals embedding size];
        graph_repr = graph.globals
        global_embedding_size = graph.globals.shape[1]

        # JUST FLATTEN the global feat?
        # shape: [B,  num_candidate * global embedding size];
        graph_repr = jnp.reshape(
            graph_repr, (-1, self.num_actions * global_embedding_size))

        # logits
        logits = MLP(self.policy_feature)(graph_repr)
        # shape: [B, num_candidate];
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)
        # batch size is flatten
        # shape [B, num_candidate]

        # value
        # shape [B, 1];
        vf = MLP(self.vf_feature)(graph_repr)
        vf = nn.Dense(1)(vf)
        # shape [B, ]
        vf = jnp.reshape(vf, (-1))

        return logits, vf


class PPO_GN_model_v4(nn.Module):
    # gnn
    mlp_features: list[int]
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers, batch_size: int):
        # NOTE: IMPLICITLY BATCH
        # e.g. batch size = 2, each GraphTuples has 10 graphs(num_candidate)
        # this gives globals: [2*10, globals embedding size]
        g = GraphNetwork_update_node(self.mlp_features)(xfers)
        g = GAT_with_global_update(self.gat_attn_mlp, self.gat_node_update_mlp,
                                   self.message_passing_steps,
                                   self.gat_global_update_mlp)(g)
        # NOTE: for now, num_candidates = self.num_actions
        # shape: [B * num_candidates, globals embedding size];
        graph_repr = g.globals
        global_embedding_size = g.globals.shape[1]

        # shape: [B,  num_candidate * global embedding size];
        graph_repr = jnp.reshape(
            graph_repr, (-1, self.num_actions * global_embedding_size))

        # logits
        logits = MLP(self.policy_feature)(graph_repr)
        # shape: [B, num_candidate];
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)
        # batch size is flatten
        # shape [B, num_candidate]

        # value
        # shape [B, 1];
        vf = MLP(self.vf_feature)(graph_repr)
        vf = nn.Dense(1)(vf)
        # shape [B, ]
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class PPO_GAT_model_v5(nn.Module):
    # gnn
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers, batch_size: int):
        # each xfers in the batch has a multiple-graphs GraphsTuple
        # so the goal is pick the best among multiple graphs
        # ====================================================
        # NOTE: IMPLICITLY BATCH
        # e.g. batch size = 2, each GraphTuples has 10 graphs(num_candidate)
        # this gives globals: [2*10, globals embedding size]
        g = GAT_with_global_update(self.gat_attn_mlp, self.gat_node_update_mlp,
                                   self.message_passing_steps,
                                   self.gat_global_update_mlp)(xfers)
        # shape: [B * num_candidates, globals embedding size];
        graph_repr = g.globals

        # logits
        logits = MLP(self.policy_feature)(graph_repr)
        logits = nn.Dense(1)(logits)
        # shape: [B, num_candidate];
        logits = jnp.reshape(logits, (batch_size, -1))
        # batch size is flatten
        # shape [B, num_candidate]

        # value
        # shape [B, 1];
        # shape: [B, num_candidates * globals embedding size];
        vf = jnp.reshape(graph_repr, (batch_size, -1))
        vf = MLP(self.vf_feature)(vf)
        vf = nn.Dense(1)(vf)
        # shape [B, ]
        vf = jnp.reshape(vf, (-1))

        return logits, vf


class PPO_GAT_model_v6(nn.Module):
    # gnn
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers, batch_size: int):
        # each xfers in the batch has a multiple-graphs GraphsTuple
        # so the goal is pick the best among multiple graphs
        # ====================================================
        # NOTE: IMPLICITLY BATCH
        # e.g. batch size = 2, each GraphTuples has 10 graphs(num_candidate)
        # this gives globals: [2*10, globals embedding size]
        g = GAT_with_global_update(self.gat_attn_mlp, self.gat_node_update_mlp,
                                   self.message_passing_steps,
                                   self.gat_global_update_mlp)(xfers)
        # shape: [B * num_candidates, globals embedding size];
        graph_repr = g.globals

        # logits
        # shape: [B * num_candidates, 1];
        logits = nn.Dense(1)(graph_repr)
        # shape: [B, num_candidates];
        logits = jnp.reshape(logits, (batch_size, -1))
        # 2 layer mlp as final policy
        logits = nn.Dense(self.num_actions)(logits)
        logits = jnp.tanh(logits)
        # shape: [B, num_candidate];
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)
        # shape: [B, num_candidate];
        # batch size is flatten
        # shape [B, num_candidate]

        # value
        # shape: [B * num_candidates, 1];
        vf = nn.Dense(1)(graph_repr)
        # shape: [B, num_candidates];
        vf = jnp.reshape(vf, (batch_size, -1))
        # 2 layer mlp as final value
        vf = nn.Dense(self.num_actions)(vf)
        vf = jnp.tanh(vf)
        # shape: [B, ];
        vf = nn.Dense(1)(vf)
        vf = jnp.reshape(vf, (-1))
        return logits, vf


class PPO_GN_model_v6(nn.Module):
    # gnn
    mlp_features: list[int]
    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int
    gat_global_update_mlp: int

    # agent
    policy_feature: Sequence[int]
    vf_feature: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, graph: GraphsTuple, xfers, batch_size: int):
        # NOTE: IMPLICITLY BATCH
        # e.g. batch size = 2, each GraphTuples has 10 graphs(num_candidate)
        # this gives globals: [2*10, globals embedding size]
        g = GraphNetwork_update_node(self.mlp_features)(xfers)
        g = GAT_with_global_update(self.gat_attn_mlp, self.gat_node_update_mlp,
                                   self.message_passing_steps,
                                   self.gat_global_update_mlp)(g)
        # shape: [B * num_candidates, globals embedding size];
        graph_repr = g.globals

        # logits
        # shape: [B * num_candidates, 1];
        logits = nn.Dense(1)(graph_repr)
        # shape: [B, num_candidates];
        logits = jnp.reshape(logits, (batch_size, -1))
        # 2 layer mlp as final policy
        logits = nn.Dense(self.num_actions)(logits)
        logits = jnp.tanh(logits)
        # shape: [B, num_candidate];
        logits = nn.Dense(self.num_actions,
                          kernel_init=default_logits_init())(logits)
        # shape: [B, num_candidate];
        # batch size is flatten
        # shape [B, num_candidate]

        # value
        # shape: [B * num_candidates, 1];
        vf = nn.Dense(1)(graph_repr)
        # shape: [B, num_candidates];
        vf = jnp.reshape(vf, (batch_size, -1))
        # 2 layer mlp as final value
        vf = nn.Dense(self.num_actions)(vf)
        vf = jnp.tanh(vf)
        # shape: [B, ];
        vf = nn.Dense(1)(vf)
        vf = jnp.reshape(vf, (-1))

        return logits, vf
