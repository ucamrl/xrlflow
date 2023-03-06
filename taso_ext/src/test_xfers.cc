#include <algorithm>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

#include "taso/ops.h"
#include "taso/substitution.h"
#include "graph_feedback.h"
#include <iostream>
#include <limits>

using namespace taso;
using namespace std;
using namespace xflowrl;

// utils
typedef float DATATYPE;

DATATYPE* new_random_data(size_t size) {
  // Random generator.
  static std::random_device r;
  static std::default_random_engine e(r());
  static std::uniform_real_distribution<DATATYPE> dist;
  auto gen = [&]() { return dist(e); };

  auto data = new DATATYPE[size];
  std::generate(data, data + size, gen);
  return data;
}

size_t dims2size(const std::vector<int>& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

inline TensorHandle new_input(Graph* graph, const std::vector<int>& dims) {
  return graph->new_input(dims.size(), dims.data());
}

inline TensorHandle new_weight(Graph* graph, const std::vector<int>& dims, const DATATYPE* data) {
  return graph->new_weight(dims.size(), dims.data(), data);
}

inline TensorHandle new_random_weight(Graph* graph, const std::vector<int>& dims) {
  return new_weight(graph, dims, new_random_data(dims2size(dims)));
}

TensorHandle attention_layer(Graph* graph, const TensorHandle input, int heads) {
  int d_model = input->dim[1];
  int d_k = d_model / heads;
  assert(input->dim[1] % heads == 0);
  TensorHandle weights[3];
  for (int i = 0; i < 3; i++) {
    weights[i] = new_random_weight(graph, { d_model, d_model });
  }
  // compute query, key, value tensors
  auto q = graph->matmul(input, weights[0]);
  auto k = graph->matmul(input, weights[1]);
  auto v = graph->matmul(input, weights[2]);
  // reshape query, key, value to multiple heads
  q = graph->reshape(q, { -1, heads, d_k });
  k = graph->reshape(k, { -1, heads, d_k });
  v = graph->reshape(v, { -1, heads, d_k });
  // transpose query, key, value for batched matmul
  q = graph->transpose(q, { 1, 0, 2 }, true);
  k = graph->transpose(k, { 1, 2, 0 }, true);
  v = graph->transpose(v, { 1, 0, 2 }, true);
  // perform matrix multiplications
  auto logits = graph->matmul(q, k);
  auto output = graph->matmul(logits, v);
  // transpose the output back
  output = graph->transpose(output, { 1, 0, 2 }, true);
  output = graph->reshape(output, { input->dim[0], input->dim[1] });

  // a final linear layer
  auto linear = new_random_weight(graph, { d_model, d_model });
  output = graph->matmul(output, linear);
  return output;
}

int main() {

  cout <<" build a graph for RLOptimizer" << endl;
  const int seq_length = 64;
  const int hidden_dims = 1024;
  Graph *graph = new Graph();

  auto inp = new_input(graph, { seq_length, hidden_dims });
  inp = graph->relu(inp);
  auto t = inp;
  for (int i = 0; i < 8; i++) {
  // for (int i = 0; i < 1; i++)
    t = attention_layer(graph, t, 16);
  }

  cout <<"start testing RLOptimizer" << endl;
  RLOptimizer opt(graph);
  opt.reset();

  auto num_loc = opt.get_available_locations();
  cout << "num of locations: " << num_loc.size() << endl;
  int cnt = 0;
  for (auto &g: num_loc) {
    if (g != 0) {
      cout << "loc: " << cnt << " appicable: " << g << endl;
    }
    cnt++;
  }

  auto num_xfers = opt.get_available_xfers();
  cout << "num of xfers: " << num_xfers.size() << endl;

  auto gs = opt.get_xfer_graphs();
  cout << "xfer rules: " << gs.size() << endl;
  cnt = 0;
  for (auto &g: gs) {
    if (g.size() != 0) {
      cout << "rule idx: " << cnt <<  " candidates: " << g.size() << endl;
    }
    cnt++;
  }

  cout <<"viz xfers" << endl;
  cout << endl;


  cout <<"xfers 18::" << endl;
  opt.viz_xfers(18);

  cout <<"xfers 95::" << endl;
  opt.viz_xfers(95);

  delete graph;
}


