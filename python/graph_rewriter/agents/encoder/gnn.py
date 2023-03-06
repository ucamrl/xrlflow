import jraph
from jraph import segment_max, segment_sum
import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Sequence
"""
Think about if edge features OR node features should be used
    reference to: https://github.com/deepmind/jraph/blob/master/jraph/_src/models.py#L440#L501
"""


class ExplicitMLP(nn.Module):
    """A flax MLP."""
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate([nn.Dense(feat) for feat in self.features]):
            x = lyr(x)
            if i != len(self.features) - 1:
                x = jax.nn.leaky_relu(x)
        return x


# ================================================
# generic graph networks
# ================================================


# Functions must be passed to jraph GNNs, but pytype does not recognise
# linen Modules as callables to here we wrap in a function.
def make_embed_fn(latent_size):

    def embed(inputs):
        return nn.Dense(latent_size)(inputs)  # 1-layer mlp

    return embed


def make_mlp(features):

    @jraph.concatenated_args
    def update_fn(inputs):
        return ExplicitMLP(features)(inputs)

    return update_fn


# this is the most general form of GNNs
# This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261
class GraphNetwork(nn.Module):
    """A flax GraphNetwork."""
    mlp_features: Sequence[int]  # each layer size
    latent_size: int
    message_passing_steps: int

    @nn.compact
    def __call__(self, graph):
        # embed node, edge, global to latent size
        embedder = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.latent_size),
            embed_edge_fn=make_embed_fn(self.latent_size),
            embed_global_fn=make_embed_fn(self.latent_size))
        graph = embedder(graph)

        for _ in range(self.message_passing_steps):
            net = jraph.GraphNetwork(
                update_node_fn=make_mlp(self.mlp_features),
                update_edge_fn=make_mlp(self.mlp_features),
                update_global_fn=make_mlp(self.mlp_features))
            graph = net(graph)
        return graph


class GraphNetwork_update_node(nn.Module):
    """only do a node update"""
    mlp_features: list[int]

    @nn.compact
    def __call__(self, graph):
        net = jraph.GraphNetwork(update_node_fn=make_mlp(self.mlp_features),
                                 update_edge_fn=None,
                                 update_global_fn=None)
        graph = net(graph)
        return graph


# ================================================
# GCN
# ================================================
class GCN(nn.Module):
    mlp_features: Sequence[int]  # each layer size
    latent_size: int
    n_hop: int

    @nn.compact
    def __call__(self, graph):
        # n_hop gcn
        for i in range(self.n_hop):
            net = jraph.GraphConvolution(
                update_node_fn=lambda x: ExplicitMLP(self.mlp_features)(x),
                add_self_edges=False,
                symmetric_normalization=True)
            graph = net(graph)

        # update global after all gcn layers
        global_layer = jraph.GraphNetwork(
            update_node_fn=None,  # None means no update
            update_edge_fn=None,
            update_global_fn=update_globals_fn())
        graph = global_layer(graph)
        return graph


# ================================================
# GAT
# ================================================
def attention_logit_fn(sender_attr: jnp.ndarray, receiver_attr: jnp.ndarray,
                       edges: jnp.ndarray) -> jnp.ndarray:
    """Standard implementations of the GAT message passing"""

    # sender_attr shape: [num_edges, num_node_embedding]

    def embed(inputs):
        return nn.Dense(1)(inputs)  # reduce to attention logits

    del edges  # edge feat no need
    x = jnp.concatenate((sender_attr, receiver_attr), axis=1)
    x = embed(x)  # reduce to attention logits
    return x


class GAT(nn.Module):

    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int

    # no use, but to keep signature
    gat_global_update_mlp: int

    @nn.compact
    def __call__(self, graph):
        # Assume self edges are added

        # GAT layer (only udpate node attributes)
        for hop in range(self.message_passing_steps):
            net = jraph.GAT(
                attention_query_fn=make_embed_fn(
                    self.gat_attn_mlp),  # hidden dim
                attention_logit_fn=attention_logit_fn,
                node_update_fn=make_embed_fn(
                    self.gat_node_update_mlp),  # node out dim
            )
            graph = net(graph)
        """
        This problem is GAT only update nodes
            so if the input graph is IMPLICILTY BATCH
                need to unbatch here

        Use segment_sum related utility
        see: https://github.com/deepmind/jraph/blob/master/jraph/_src/utils.py
        """
        graph = jraph.zero_out_padding(graph)
        return graph


def make_global_update_fn(mlp_feat):
    """
    update global with aggregated_node_features and initial global
    """

    def update_globals_fn(aggregated_node_features, aggregated_edge_features,
                          globals_):
        del aggregated_edge_features

        def embed(inputs):
            return nn.Dense(mlp_feat)(inputs)  # 1-layer mlp

        agg = jnp.concatenate([aggregated_node_features, globals_], axis=1)
        return embed(agg)

    return update_globals_fn


class GAT_with_global_update(nn.Module):
    """GAT doesn't do global updates
    so instead we can do a global update after GAT layers
    and then use the globals as graph embeddings
    """

    gat_attn_mlp: int
    gat_node_update_mlp: int
    message_passing_steps: int

    gat_global_update_mlp: int

    @nn.compact
    def __call__(self, graph):
        # Assume self edges are added

        # GAT layer (only udpate node attributes)
        for hop in range(self.message_passing_steps):
            net = jraph.GAT(
                attention_query_fn=make_embed_fn(
                    self.gat_attn_mlp),  # hidden dim
                attention_logit_fn=attention_logit_fn,
                node_update_fn=None,  # by default, apply leaky rely
                # make_embed_fn(self.gat_node_update_mlp)
            )
            graph = net(graph)

        # zero out padded nodes, edges and globals
        # NOTE: if pad node only, they are not connected so no update in GAT
        # NOTE: if also pad edge, need more cautious
        # graph = jraph.zero_out_padding(graph)

        # ONLY update the gobal; can control how to aggregate node for this too
        net = jraph.GraphNetwork(update_node_fn=None,
                                 update_edge_fn=None,
                                 update_global_fn=make_global_update_fn(
                                     self.gat_global_update_mlp))
        graph = net(graph)
        return graph


# ===== Multi-head GAT =====
# GAT inter-layer, apply the leaky relu and then concatenate the heads on the
# feature axis.
def _inter_node_update_fn(x):
    return jnp.reshape(jax.nn.leaky_relu(x), (x.shape[0], -1))


# GAT final layer, average over heads
def _final_node_update_fn(x):
    # input x shape [num_node, num_head, hidden_dim]
    num_node = x.shape[0]
    return jnp.reshape(jax.nn.leaky_relu(jnp.mean(x, axis=1)), (num_node, -1))


def _multi_head_attention_logit_fn(sender_attr: jnp.ndarray,
                                   receiver_attr: jnp.ndarray,
                                   edges: jnp.ndarray) -> jnp.ndarray:
    # sender_attr/receiver_attr shape: [num_edges, num_head, hidden_dim]
    def embed(inputs):
        return nn.Dense(1)(inputs)  # reduce to attention logits

    del edges  # edge feat no need
    x = jnp.concatenate((sender_attr, receiver_attr), axis=2)
    x = embed(x)  # reduce to attention logits
    return x


# https://github.com/deepmind/jraph/blob/master/jraph/_src/models.py#L442.
def _multi_head_GAT(attention_query_fn: callable,
                    multi_head_attention_logit_fn: callable,
                    node_update_fn: callable, num_head: int,
                    hidden_dim: int) -> callable:

    def _ApplyGAT(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes, edges, receivers, senders, _, _, _ = graph
        # Equivalent to the sum of n_node, but statically known.
        try:
            sum_n_node = nodes.shape[0]
        except IndexError:
            raise IndexError('GAT requires node features')

        # Pass nodes through the mlp to transform node features
        # shape: [num_nodes, node_feat]
        nodes = attention_query_fn(nodes)
        total_num_edges = len(senders)
        # use index to retrieve feat
        # will have replicated feat
        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        # reshape to multi-head
        sent_attributes = jnp.reshape(
            sent_attributes, ((total_num_edges, num_head, hidden_dim)))
        received_attributes = jnp.reshape(
            received_attributes, ((total_num_edges, num_head, hidden_dim)))
        att_softmax_logits = multi_head_attention_logit_fn(
            sent_attributes, received_attributes, edges)

        # Compute the attention softmax weights on the entire tree.
        att_weights = jraph.segment_softmax(att_softmax_logits,
                                            segment_ids=receivers,
                                            num_segments=sum_n_node)

        # Apply attention weights.
        # shape: [num_edge, num_head, hidden_dim]
        messages = sent_attributes * att_weights
        # Aggregate messages to nodes.
        # shape: [num_node, num_head, hidden_dim]
        nodes = jax.ops.segment_sum(messages,
                                    receivers,
                                    num_segments=sum_n_node)

        # Apply an update function to the aggregated messages.
        nodes = node_update_fn(nodes)
        return graph._replace(nodes=nodes)

    return _ApplyGAT


class multi_head_GAT(nn.Module):
    num_head: int
    hidden_dim: int
    message_passing_steps: int

    @nn.compact
    def __call__(self, graph):
        # NOTE: Assume self edges are added

        # GAT layer (only udpate node attributes)
        # intermediate layer concat `heads`
        for hop in range(self.message_passing_steps - 1):
            net = _multi_head_GAT(
                attention_query_fn=make_embed_fn(
                    int(self.num_head * self.hidden_dim)),
                multi_head_attention_logit_fn=_multi_head_attention_logit_fn,
                node_update_fn=_inter_node_update_fn,
                num_head=self.num_head,
                hidden_dim=self.hidden_dim,
            )
            graph = net(graph)

        # last layer average heads
        net = _multi_head_GAT(
            attention_query_fn=make_embed_fn(
                int(self.num_head * self.hidden_dim)),
            multi_head_attention_logit_fn=_multi_head_attention_logit_fn,
            node_update_fn=_final_node_update_fn,
            num_head=self.num_head,
            hidden_dim=self.hidden_dim,
        )
        graph = net(graph)
        return graph


class max_global_layer(nn.Module):
    gat_global_update_mlp: int

    @nn.compact
    def __call__(self, graph):
        net = jraph.GraphNetwork(update_node_fn=None,
                                 update_edge_fn=None,
                                 update_global_fn=make_global_update_fn(
                                     self.gat_global_update_mlp),
                                 aggregate_nodes_for_globals_fn=segment_max)

        return net(graph)


class sum_global_layer(nn.Module):
    gat_global_update_mlp: int

    @nn.compact
    def __call__(self, graph):
        net = jraph.GraphNetwork(update_node_fn=None,
                                 update_edge_fn=None,
                                 update_global_fn=make_global_update_fn(
                                     self.gat_global_update_mlp),
                                 aggregate_nodes_for_globals_fn=segment_sum)

        return net(graph)
