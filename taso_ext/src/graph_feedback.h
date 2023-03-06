#ifndef GRAPH_FEEDBACK_H
#define GRAPH_FEEDBACK_H

#include "taso/ops.h"
#include "taso/substitution.h"
#include <iostream>
// #include <cuda.h>  // to use cuda free
#include <cuda_runtime.h>

using namespace taso;

namespace xflowrl {


class DummyGraphCompare : public GraphCompare {
public:
  bool operator() (Graph* lhs, Graph* rhs) {
    return false;
  }
};

class RLOptimizer {
  protected:
    Graph* graph;
    std::vector<GraphXfer*> xfers;
    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates_{};

    std::vector<std::vector<Graph*>> xfer_graphs; // each xfer has a vector of graphs
    std::vector<std::vector<std::vector<Op>>> xfer_inputs;  // each xfer has a vector of a list of inputs
    std::vector<std::vector<std::vector<Op>>> xfer_outputs; // and outputs

    std::set<size_t> hashmap;
    Graph* bestGraph;
    float bestCost;

  public:
    RLOptimizer(Graph* graph);
    ~RLOptimizer();

    void free_xfer();
    void free_xfer_graphs();

    void set_graph(Graph* graph);
    Graph* get_pre_process_graph();
    float eval_cur_graph(bool verbose);
    float eval_cur_no_pre_process_safe(bool verbose);
    float eval_cur_graph_safe(bool verbose);

    //
    bool reset();
    int get_num_xfers();

    // state
    std::vector<int> get_available_xfers();
    std::vector<int> get_available_locations();

    std::vector<std::vector<Graph*>> get_xfer_graphs();

    std::vector<std::vector<std::vector<Op>>> get_xfer_inputs();
    std::vector<std::vector<std::vector<Op>>> get_xfer_outputs();

    void get_xfer_locations(
                    GraphXfer* xfer,
                    int depth, Graph* graph,
                    std::vector<Graph*>& graphs,
                    std::vector<std::vector<Op>>& xfer_inputs,
                    std::vector<std::vector<Op>>& xfer_outputs,
                    std::set<size_t>& hashmap, int maxNumOps);

    float get_op_runtime(size_t guid);
    float get_op_runtime_for_graph(Graph* graph, size_t guid);

    // action
    Graph* apply_xfer(int xfer_id, int location_id);

    // reward
    float get_cost();
    float get_measured_runtime(Graph* graph);

    void reproduce_taso_optimize(float alpha, int budget, int measure_interval);
    void custom_run(GraphXfer* xfer,
                    int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::vector<std::vector<Op>>& this_xfer_inputs,
                    std::vector<std::vector<Op>>& this_xfer_outputs,
                    std::set<size_t>& hashmap, float threshold,  int maxNumOps);

    // viz xfers
    void viz_xfers(int idx);
};

}
#endif // GRAPH_FEEDBACK_H
