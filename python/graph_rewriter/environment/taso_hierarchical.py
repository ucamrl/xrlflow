import numpy as np
from functools import partial
import jax.numpy as jnp
import jraph

from graph_rewriter.environment.taso_base import _BaseEnvironment, graph_to_graphnet_tuple


class HierarchicalEnvironment(_BaseEnvironment):

    def __init__(self,
                 num_locations=100,
                 real_measurements=False,
                 reward_function=None,
                 node_cost_model=True):
        super().__init__(num_locations, real_measurements, reward_function,
                         node_cost_model)

    def step(self, actions):
        xfer_id, location_id = actions
        self.time_step += 1

        terminate = False
        # graph_feedback's xfers.size()
        if xfer_id == self.rl_opt.get_num_xfers():
            # No-op action terminates the sequence
            new_graph = self.graph
            terminate = True
        elif xfer_id > self.rl_opt.get_num_xfers():
            assert (False), "xfer_id > self.rl_opt.get_num_xfers"
        else:
            new_graph = self.rl_opt.apply_xfer(xfer_id, location_id)

        if new_graph:
            self.graph = new_graph
            new_runtime = self.get_cost()
            assert (new_runtime > 0), f"new_runtime? {new_runtime:.2f}"

            if self.custom_reward is not None:
                reward = self.custom_reward(self.time_step, self.last_runtime,
                                            self.initial_runtime, new_runtime,
                                            terminate)
            else:
                raise ValueError("provide your own reward func!")

            self.last_runtime = new_runtime
        else:
            print("Invalid action: xfer {xfer_id} with location {location_id}")
            reward = -1000.

        new_state = self.build_state()

        terminal = False
        if np.sum(new_state['mask']) == 0 or terminate:
            terminal = True

        return new_state, reward, terminal, None

    def _normalize_measurements(self, x_key, x):
        if x_key not in self.measurement_info:
            self.measurement_info[x_key] = []
        self.measurement_info[x_key].append(x)

        val = self.measurement_info[x_key]
        x_min, x_max = np.min(val), np.max(val)
        return 2 * ((x - x_min) / (x_max - x_min + 1e-6)) - 1


class FlatEnvironment(_BaseEnvironment):

    def __init__(self,
                 num_locations=200,
                 real_measurements=False,
                 reward_function=None,
                 node_cost_model=True):
        super().__init__(num_locations, real_measurements, reward_function,
                         node_cost_model)

    def step(self, action):
        action = int(action)
        self.time_step += 1

        # NOTE: self.locations is up-to-date with xfers
        total_num_xfers = sum(self.locations)

        done = False
        if action == total_num_xfers:
            # No-op action terminates the sequence
            done = True
            xfer_id, location_id = 151, 0
        elif action > total_num_xfers:
            assert (False), "step fails {action} > {total_num_xfers}"
        else:
            xfer_id, location_id = _unfold_action(self.locations, action)
            new_graph = self.rl_opt.apply_xfer(xfer_id, location_id)
            assert (new_graph is not None), "apply fail"
            self.graph = new_graph

        # Make sure to only use the estimated cost before final eval
        new_runtime = self.get_cost()
        assert (new_runtime > 0), f"new_runtime? {new_runtime:.2f}"

        if self.custom_reward is not None:
            reward = self.custom_reward(self.time_step, self.last_runtime,
                                        self.initial_runtime, new_runtime,
                                        done)
        else:
            raise ValueError("provide your own reward func!")

        self.last_runtime = new_runtime

        new_state = self.build_state()
        if np.sum(new_state['mask']) == 0:
            done = True

        info = {
            "xfer_id": xfer_id,
            "location_id": location_id,
        }
        return new_state, reward, done, info

    def build_state(self):
        """flatten the hierarhical candidates"""
        # list[int], each int indicate number of locations
        # that this xfer can apply
        self.locations = self.rl_opt.get_available_locations()

        # list[list[taso.core.PyGraph object]], the result graph
        # after applying the xfer
        self.xfer_graphs = self.rl_opt.get_xfer_graphs()

        assert len(self.locations) == len(
            self.xfer_graphs), "xfer_graphs len != locations len"

        # Xfer mask; list[bool] NOTE should not be used
        xfer_mask = jnp.asarray(self.locations).astype(bool).astype(int)
        xfer_mask = jnp.append(xfer_mask, 1)  # add one last action

        # Main graphnet tuple
        graph_tuple = graph_to_graphnet_tuple(
            self.graph,
            self.rl_opt.get_op_runtime,
            node_cost_model=self.node_cost_model)

        # Sub graphnet tuple
        xfer_tuples = []
        # NOTE: [[self.graph]] is termination
        for xfer in self.xfer_graphs + [[self.graph]]:
            for xg in xfer:
                g = graph_to_graphnet_tuple(
                    xg,
                    op_runtime_callback=partial(
                        self.rl_opt.get_op_runtime_for_graph, xg),
                    node_cost_model=self.node_cost_model)
                xfer_tuples.append(g)

        this_xfer_tuple = jraph.batch(xfer_tuples)
        len_xfer = len(xfer_tuples)
        assert (self.num_locations > len_xfer
                ), f"num_locations {self.num_locations} < len_xfer {len_xfer}"

        # pad to constant num_of_graph
        pad_nodes_to = jnp.sum(this_xfer_tuple.n_node) + 1
        # edge doesn't need + 1
        pad_edges_to = jnp.sum(this_xfer_tuple.n_edge)
        pad_graphs_to = self.num_locations
        # padded_graph has multiple graph instances
        padded_graph = jraph.pad_with_graphs(this_xfer_tuple, pad_nodes_to,
                                             pad_edges_to, pad_graphs_to)

        # pad location_mask -> self.num_locations
        location_mask = [1] * len_xfer + [0] * (self.num_locations - len_xfer)
        # to jnp
        location_mask = jnp.array(location_mask, dtype=jnp.int32)

        return {
            "graph": graph_tuple,
            "mask": xfer_mask,
            "xfers": padded_graph,
            "candidates_mask": location_mask
        }


class FlatEnvironment_Masked(_BaseEnvironment):

    def __init__(self,
                 num_locations=200,
                 real_measurements=False,
                 reward_function=None,
                 node_cost_model=True):
        super().__init__(num_locations, real_measurements, reward_function,
                         node_cost_model)

    def step(self, action):
        action = int(action)
        self.time_step += 1

        # NOTE: self.locations is up-to-date with xfers
        total_num_xfers = sum(self.locations)

        done = False
        if action == self.num_locations:
            # No-op action terminates the sequence
            done = True
            xfer_id, location_id = 151, 0
        elif action >= total_num_xfers:
            assert (False), "step fails {action} >= {total_num_xfers}"
        else:
            # return nullptr if (xfer_id, location_id) is invalid
            xfer_id, location_id = _unfold_action(self.locations, action)
            new_graph = self.rl_opt.apply_xfer(xfer_id, location_id)
            assert (new_graph is not None), "apply fail"
            self.graph = new_graph

        # Make sure to only use the estimated cost before final eval
        new_runtime = self.get_cost()
        assert (new_runtime > 0), f"new_runtime? {new_runtime:.2f}"

        if self.custom_reward is not None:
            reward = self.custom_reward(self.time_step, self.last_runtime,
                                        self.initial_runtime, new_runtime,
                                        done)
        else:
            raise ValueError("provide your own reward func!")

        self.last_runtime = new_runtime

        new_state = self.build_state()
        if np.sum(new_state['mask']) == 0:
            done = True

        info = {
            "xfer_id": xfer_id,
            "location_id": location_id,
        }
        return new_state, reward, done, info

    def build_state(self):
        """flatten the hierarhical candidates"""
        # list[int], each int indicate number of locations
        self.locations = self.rl_opt.get_available_locations()

        # list[list[taso.core.PyGraph object]], the result graph
        self.xfer_graphs = self.rl_opt.get_xfer_graphs()

        assert len(self.locations) == len(
            self.xfer_graphs), "xfer_graphs len != locations len"

        xfer_mask = jnp.asarray(self.locations).astype(bool).astype(int)
        xfer_mask = jnp.append(xfer_mask, 1)  # add one last action

        # Main graphnet tuple
        graph_tuple = graph_to_graphnet_tuple(
            self.graph,
            op_runtime_callback=self.rl_opt.get_op_runtime,
            node_cost_model=self.node_cost_model)

        # Sub graphnet tuple
        xfer_tuples = []
        for xfer in self.xfer_graphs:
            for xg in xfer:
                g = graph_to_graphnet_tuple(
                    xg,
                    op_runtime_callback=partial(
                        self.rl_opt.get_op_runtime_for_graph, xg),
                    node_cost_model=self.node_cost_model)
                xfer_tuples.append(g)

        if len(xfer_tuples) == 0:
            padded_graph = None
        else:
            this_xfer_tuple = jraph.batch(xfer_tuples)

            # pad to constant num_of_graph
            pad_nodes_to = jnp.sum(this_xfer_tuple.n_node) + 1
            # edge doesn't need + 1
            pad_edges_to = jnp.sum(this_xfer_tuple.n_edge)
            pad_graphs_to = self.num_locations
            # padded_graph has multiple graph instances
            padded_graph = jraph.pad_with_graphs(this_xfer_tuple, pad_nodes_to,
                                                 pad_edges_to, pad_graphs_to)
        len_xfer = len(xfer_tuples)
        if self.num_locations < len_xfer:
            raise RuntimeError(
                f"num_locations {self.num_locations} < len_xfer {len_xfer}")

        # pad location_mask -> self.num_locations
        location_mask = [1] * len_xfer + [0] * (self.num_locations - len_xfer)
        # last one to mark stop
        location_mask.append(1)
        # to jnp
        location_mask = jnp.array(location_mask, dtype=jnp.int32)

        return {
            "graph": graph_tuple,
            "mask": xfer_mask,
            "xfers": padded_graph,
            "candidates_mask": location_mask
        }


class FlatEnvironment_sparse(_BaseEnvironment):

    def __init__(self,
                 num_locations=200,
                 real_measurements=False,
                 measure_interval=5,
                 reward_function=None,
                 node_cost_model=False):
        super().__init__(num_locations, real_measurements, None,
                         node_cost_model)
        self.measure_interval = measure_interval

    def step(self, action):
        action = int(action)
        self.time_step += 1

        # NOTE: self.locations is up-to-date with xfers
        total_num_xfers = sum(self.locations)

        done = False
        if action == total_num_xfers:
            # No-op action terminates the sequence
            done = True
            xfer_id, location_id = 151, 0
        elif action > total_num_xfers:
            assert (False), "step fails {action} > {total_num_xfers}"
        else:
            xfer_id, location_id = _unfold_action(self.locations, action)
            new_graph = self.rl_opt.apply_xfer(xfer_id, location_id)
            assert (new_graph is not None), "apply fail"
            self.graph = new_graph

        # the cost model has no use in sparse environment
        new_runtime = 1.
        assert (new_runtime > 0), f"new_runtime? {new_runtime:.2f}"

        # reward
        if self.time_step % self.measure_interval == 0 or done:
            measured_runtime = self.eval_cur_graph_safe()
            reward = (self.last_measured_runtime -
                      measured_runtime) / self.real_measurements_runtime
            reward *= 100
            self.last_measured_runtime = measured_runtime
        else:
            reward = 0.1

        self.last_runtime = new_runtime

        new_state = self.build_state()
        if np.sum(new_state['mask']) == 0:
            done = True

        info = {
            "xfer_id": xfer_id,
            "location_id": location_id,
        }
        return new_state, reward, done, info

    def build_state(self):
        """flatten the hierarhical candidates"""
        # list[int], each int indicate number of locations
        # that this xfer can apply
        self.locations = self.rl_opt.get_available_locations()

        # list[list[taso.core.PyGraph object]], the result graph
        # after applying the xfer
        self.xfer_graphs = self.rl_opt.get_xfer_graphs()

        assert len(self.locations) == len(
            self.xfer_graphs), "xfer_graphs len != locations len"

        # Xfer mask; list[bool] NOTE should not be used
        xfer_mask = jnp.asarray(self.locations).astype(bool).astype(int)
        xfer_mask = jnp.append(xfer_mask, 1)  # add one last action

        # Main graphnet tuple
        graph_tuple = graph_to_graphnet_tuple(
            self.graph,
            op_runtime_callback=self.rl_opt.get_op_runtime,
            node_cost_model=self.node_cost_model)

        # Sub graphnet tuple
        xfer_tuples = []
        # NOTE: [[self.graph]] is termination
        for xfer in self.xfer_graphs + [[self.graph]]:
            for xg in xfer:
                g = graph_to_graphnet_tuple(
                    xg,
                    op_runtime_callback=partial(
                        self.rl_opt.get_op_runtime_for_graph, xg),
                    node_cost_model=self.node_cost_model)
                xfer_tuples.append(g)

        this_xfer_tuple = jraph.batch(xfer_tuples)
        len_xfer = len(xfer_tuples)
        assert (self.num_locations > len_xfer
                ), f"num_locations {self.num_locations} < len_xfer {len_xfer}"

        # pad to constant num_of_graph
        pad_nodes_to = jnp.sum(this_xfer_tuple.n_node) + 1
        # edge doesn't need + 1
        pad_edges_to = jnp.sum(this_xfer_tuple.n_edge)
        pad_graphs_to = self.num_locations
        # padded_graph has multiple graph instances
        padded_graph = jraph.pad_with_graphs(this_xfer_tuple, pad_nodes_to,
                                             pad_edges_to, pad_graphs_to)

        # pad location_mask -> self.num_locations
        location_mask = [1] * len_xfer + [0] * (self.num_locations - len_xfer)
        # to jnp
        location_mask = jnp.array(location_mask, dtype=jnp.int32)

        return {
            "graph": graph_tuple,
            "mask": xfer_mask,
            "xfers": padded_graph,
            "candidates_mask": location_mask
        }


def _unfold_action(locations: list[int], action: int):
    assert (sum(locations) > action)
    cnt = 0
    for i, num in enumerate(locations):
        if num != 0:
            for j in range(num):
                if cnt == action:
                    return i, j
                else:
                    cnt += 1
    assert (False)
