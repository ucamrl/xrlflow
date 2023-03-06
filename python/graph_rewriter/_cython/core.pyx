from CCore cimport *
from CExt cimport *
import ctypes

from libc.stdint cimport uintptr_t
from libcpp.vector cimport vector

from taso.core import PyGraph, PyTensor

cdef class PyRLOptimizer:
    cdef RLOptimizer* rl_optimizer

    def __cinit__(self, pygraph):
        cdef uintptr_t ptr = pygraph.get_ptr_addr()

        c_graph = <Graph*>(ptr)
        self.rl_optimizer = new RLOptimizer(c_graph)

    def __dealloc__(self):
        del self.rl_optimizer

        #cdef Graph* c_graph = self.rl_optimizer.get_graph().get_ptr_addr()

    def get_pre_process_graph(self):
        cdef Graph_ptr g = self.rl_optimizer.get_pre_process_graph()
        pyptr = ctypes.cast(<unsigned long long>g, ctypes.c_void_p)
        return PyGraph(pyptr)

    def eval_cur_graph(self, verbose=False):
        return self.rl_optimizer.eval_cur_graph(verbose)

    def eval_cur_graph_safe(self, verbose=False):
        return self.rl_optimizer.eval_cur_graph_safe(verbose)

    def eval_cur_no_pre_process_safe(self, verbose=False):
        return self.rl_optimizer.eval_cur_no_pre_process_safe(verbose)

    def reset(self):
        return self.rl_optimizer.reset()

    def get_cost(self):
        return self.rl_optimizer.get_cost()

    def get_num_xfers(self):
        return self.rl_optimizer.get_num_xfers()

    def get_available_xfers(self):
        return self.rl_optimizer.get_available_xfers()

    def get_xfer_graphs(self):
        cdef vector[vector[Graph_ptr]] xfer_graphs = self.rl_optimizer.get_xfer_graphs()
        out_list = []
        for xfer in xfer_graphs:
            xfer_positions = []
            for position_graph in xfer:
                pyptr = ctypes.cast(<unsigned long long>position_graph, ctypes.c_void_p)
                pyobj = PyGraph(pyptr)
                xfer_positions.append(pyobj)
            out_list.append(xfer_positions)
        return out_list

    def get_xfer_inputs(self):
        assert(False), "should not use"
        cdef vector[vector[vector[Op]]] xfer_inputs = self.rl_optimizer.get_xfer_inputs()
        out_list = []
        for xfer in xfer_inputs:
            xfer_positions = []
            for position in xfer:
                position_inputs = []
                for in_tensor in position:
                    # pyptr = ctypes.cast(<unsigned long long>in_tensor, ctypes.c_void_p)
                    pyobj = in_tensor
                    position_inputs.append(pyobj)
                xfer_positions.append(position_inputs)
            out_list.append(xfer_positions)
        return out_list

    def get_xfer_outputs(self):
        assert(False), "should not use"
        cdef vector[vector[vector[Op]]] xfer_outputs = self.rl_optimizer.get_xfer_outputs()
        out_list = []
        for xfer in xfer_outputs:
            xfer_positions = []
            for position in xfer:
                position_outputs = []
                for out_tensor in position:
                    # pyptr = ctypes.cast(<unsigned long long>out_tensor, ctypes.c_void_p)
                    pyobj = out_tensor
                    position_outputs.append(pyobj)
                xfer_positions.append(position_outputs)
            out_list.append(xfer_positions)
        return out_list

    def get_available_locations(self):
        return self.rl_optimizer.get_available_locations()

    def get_op_runtime(self, Op op):
        cdef runtime = self.rl_optimizer.get_op_runtime(op.guid)
        return runtime

    def get_op_runtime_for_graph(self, pygraph, Op op):
        cdef uintptr_t ptr = pygraph.get_ptr_addr()
        c_graph = <Graph*>(ptr)

        cdef runtime = self.rl_optimizer.get_op_runtime_for_graph(c_graph, op.guid)
        return runtime

    def apply_xfer(self, xfer_id, location_id):
        cdef Graph_ptr new_graph = self.rl_optimizer.apply_xfer(xfer_id, location_id)
        pyptr = ctypes.cast(<unsigned long long>new_graph, ctypes.c_void_p)
        if not pyptr:
            return None
        pyobj = PyGraph(pyptr)
        return pyobj

    def get_measured_runtime(self, pygraph):
        cdef uintptr_t ptr = pygraph.get_ptr_addr()
        c_graph = <Graph*>(ptr)

        return c_graph.run_memorysafe()

    def reproduce_taso_optimize(self, alpha, budget, measure_interval):
        self.rl_optimizer.reproduce_taso_optimize(alpha, budget, measure_interval)
