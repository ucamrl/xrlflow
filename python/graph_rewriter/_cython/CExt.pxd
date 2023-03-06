# Copyright 2019 Stanford
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#ccore.pxd

from libcpp cimport bool
from libcpp.vector cimport vector
from CCore cimport Graph, Tensor, Op
import ctypes

ctypedef Graph* Graph_ptr
ctypedef Tensor* Tensor_ptr
ctypedef Op* Op_ptr

cdef extern from "graph_feedback.h" namespace "xflowrl":

    cdef cppclass RLOptimizer:
        RLOptimizer(Graph* graph);
        # ~RLOptimizer();

        void free_xfer();
        void free_xfer_graphs();
        void set_graph(Graph* graph)
        Graph* get_pre_process_graph()
        float eval_cur_graph(bool verbose)
        float eval_cur_no_pre_process_safe(bool verbose)
        float eval_cur_graph_safe(bool verbose)

        bool reset()
        int get_num_xfers()

        vector[int] get_available_xfers()
        vector[int] get_available_locations()

        vector[vector[Graph_ptr]] get_xfer_graphs()
        vector[vector[vector[Op]]] get_xfer_inputs()
        vector[vector[vector[Op]]] get_xfer_outputs()

        float get_op_runtime(size_t guid)
        float get_op_runtime_for_graph(Graph* graph, size_t guid)

        Graph* apply_xfer(int xfer_id, int location_id)

        float get_cost()
        float get_measured_runtime(Graph* graph)
        void reproduce_taso_optimize(float alpha, int budget, int measure_interval)
