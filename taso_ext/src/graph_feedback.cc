#include "taso/ops.h"
#include "taso/substitution.h"
#include "graph_feedback.h"
#include <iostream>
#include <limits>
#include <queue>


using namespace taso;
using namespace xflowrl;

// RLOptimizer interfaces with TASO
// TASO has memory leak so each env.step() causes a leak
// this should not be a problem in general, as long as we checkoutpoint the RL agent
// and resume training if we need to train a large number of episodes

RLOptimizer::RLOptimizer(Graph* graph)
  : graph(graph) , xfers(), xfer_graphs(), xfer_inputs(), xfer_outputs() {}

RLOptimizer::~RLOptimizer() {
  free_xfer();
  free_xfer_graphs();
}

void RLOptimizer::free_xfer() {
  if (xfers.size() != 0) {
    for (auto &x: xfers) {
      for(auto &opX: x->srcOps) {
        if (opX) {
          delete opX;
          opX = nullptr;
        }
      }
      for(auto &opX: x->dstOps) {
        if (opX) {
          delete opX;
          opX = nullptr;
        }
      }
      delete x;
    }
  }
  xfers.clear();
}

void RLOptimizer::free_xfer_graphs() {

  if (xfer_graphs.size() != 0) {
    for (auto &gg: xfer_graphs) {
      for (auto &g: gg) {
        // FIXME insides the graph, there are other memory leak
        // free Ops
        //std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::iterator opIt;
        //for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); ++opIt) {
        //  if (opIt->first.ptr)
        //    delete opIt->first.ptr;
        //    opIt->first.ptr = nullptr;  // this seems like a read-only object
        //}

        // free graph
        if (g) {
          delete g;
          g = nullptr;
        }
      }
    }
  }
  xfer_graphs.clear();
}


void RLOptimizer::set_graph(Graph* graph) {
  this->graph = graph;
}

Graph* RLOptimizer::get_pre_process_graph() {
  // need to free by caller
  return this->graph->preprocess_weights();
}

float RLOptimizer::eval_cur_graph(bool verbose) {
  printf("[RLOptimizer] [Warning] eval_cur_graph memory not safe");
  Graph* new_g = this->graph->preprocess_weights();
  float ret =new_g->run();
  if (verbose) {
    printf("        ===== Eval =====\n\n");
    printf("Cost model = %.4lf\n", new_g->total_cost());
    printf("End-to-end inference time =%.8lf ms (average of 100 runs)\n", ret);
  }
  delete new_g;
  return ret;
}

float RLOptimizer::eval_cur_no_pre_process_safe(bool verbose) {
  printf("[RLOptimizer] [Warning] no pre-processed graph");
  float ret = this->graph->run_memorysafe();
  if (verbose) {
    printf("        ===== Eval =====\n\n");
    printf("Cost model = %.4lf\n", this->graph->total_cost());
    printf("End-to-end inference time =%.8lf ms (average of 100 runs)\n", ret);
  }
  return ret;
}

float RLOptimizer::eval_cur_graph_safe(bool verbose) {
  Graph* newGraph = this->graph->preprocess_weights();
  float ret = newGraph->run_memorysafe();
  if (verbose) {
    printf("        ===== Eval =====\n\n");
    printf("Cost model = %.4lf\n", newGraph->total_cost());
    printf("End-to-end inference time =%.8lf ms (average of 100 runs)\n", ret);
  }
  // free newGraph; can run rules in 8GB GPU now!!
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::iterator opIt;
  for (opIt = newGraph->inEdges.begin(); opIt != newGraph->inEdges.end(); ++opIt) {
    const Op& ret = opIt->first;
    if (ret.ptr->type == OP_WEIGHT) {
      if (ret.ptr->numInputs == 1 && ret.ptr->numOutputs == 1) {
        Tensor& tensor = ret.ptr->outputs[0];
        if (tensor.op.guid == GUID_FREE && tensor.op.ptr == NULL && tensor.idx == 0 && tensor.data_ptr != NULL) {
          cudaFree(tensor.data_ptr);
          tensor.data_ptr = NULL;
        }
      }
    }
  }

  delete newGraph;
  return ret;
}

bool RLOptimizer::reset() {
  Graph* g = this->graph;

  // prevent mem leak;
  free_xfer();
  //printf("Resetting Xfers...\n");
  xfers = std::vector<GraphXfer*>();

  for (int i = 1; i < 3; i++)
    for (int j = 0; j < 2; j++) {
      PaddingMode pad_mode = (j == 0) ? PD_MODE_SAME : PD_MODE_VALID;
      xfers.push_back(GraphXfer::create_conv_relu(g->model, i, i, pad_mode));
      xfers.push_back(GraphXfer::create_conv_batch(g->model, i, i, pad_mode));
      xfers.push_back(GraphXfer::create_conv_mul(g->model, i, i, pad_mode));
      //xfers.push_back(GraphXfer::create_conv_add(g->model, i, i, pad_mode));
    }
  xfers.push_back(GraphXfer::create_enlarge_merge_convs(g->model, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_enlarge_merge_convs(g->model, AC_MODE_RELU));
  xfers.push_back(GraphXfer::create_merge_group_convs(g->model, 1, 1, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_merge_group_convs(g->model, 1, 1, AC_MODE_RELU));
  xfers.push_back(GraphXfer::create_merge_group_convs(g->model, 2, 2, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_merge_group_convs(g->model, 2, 2, AC_MODE_RELU));

  //xfers.push_back(create_avg_pool_conv(g->model));
  //xfers.push_back(create_two_pools(g->model));
  //xfers.push_back(create_merge_seperable_convs(g->model));
  char* taso_path = getenv("TASO_HOME");
  if (taso_path == NULL) {
    fprintf(stderr, "Error: environment variable TASO_HOME is not set. "
           "Please set TASO_HOME to the home directory of TASO source code.\n");
    assert(false);
  }
  std::string graph_subst_file = std::string(taso_path) + "/graph_subst.pb";
  GraphXfer::load_graph_xfer_from_pb_file(g->model, xfers, graph_subst_file);
  //xfers.push_back(create_fuse_conv_batch_xfer(g->model));
  //xfers.push_back(create_fuse_conv_relu_xfer(g->model));
  //xfers.push_back(create_merge_conv_xfer(g->model));
  //xfers.push_back(create_exclusive_concat_xfer(g->model));
  //xfers.push_back(create_enlarge_conv_xfer(g->model));
  //xfers.push_back(create_resnet_merge_xfer(g->model));

  //printf("Resetting stats...\n");

  bestGraph = g;
  bestCost = g->total_cost();

  hashmap.clear();
  hashmap.insert(g->hash());
  return true;
}

int RLOptimizer::get_num_xfers() {
  return static_cast<int>(xfers.size());
}

// state
std::vector<int> RLOptimizer::get_available_xfers() {
  std::vector<int> available_xfers{};
  for (int i = 0; i < xfers.size(); i++) {
    available_xfers.push_back(1);
  }
  return available_xfers;
}

std::vector<int> RLOptimizer::get_available_locations() {
  // called by env.reset()
  // this try to apply each xfer to each node in the graph
  // as a result xfer_graphs & xfer_inputs & xfer_outputs are populated
  //
  hashmap.clear();
  hashmap.insert(graph->hash());

  //auto available_xfers = new std::vector<int>();
  std::vector<int> available_xfers{};

  // clear xfer graphs and maps
  // free_xfer_graphs();  // free_xfer_graphs cannot free entirely
  xfer_graphs = std::vector<std::vector<Graph*>>();
  xfer_inputs = std::vector<std::vector<std::vector<Op>>>();
  xfer_outputs = std::vector<std::vector<std::vector<Op>>>();

  int maxNumOps = graph->inEdges.size() * 2;

  // loop through all xfers
  for (int i = 0; i < xfers.size(); i++) {
    std::vector<Graph*> this_xfer_graphs;
    std::vector<std::vector<Op>> this_xfer_inputs;
    std::vector<std::vector<Op>> this_xfer_outputs;

    get_xfer_locations(xfers[i], 0, graph, this_xfer_graphs, this_xfer_inputs, this_xfer_outputs, hashmap, maxNumOps);

    int num_locations = (int)this_xfer_graphs.size();
    // cout << "id " << i << " num_loc: " << num_locations << endl;
    //assert((int)this_xfer_inputs.size() == num_locations);
    //assert((int)this_xfer_outputs.size() == num_locations);

    xfer_graphs.push_back(this_xfer_graphs);
    xfer_inputs.push_back(this_xfer_inputs);
    xfer_outputs.push_back(this_xfer_outputs);

    available_xfers.push_back(num_locations);
  }
  return available_xfers;
}

std::vector<std::vector<Graph*>> RLOptimizer::get_xfer_graphs() {
  return xfer_graphs;
}

std::vector<std::vector<std::vector<Op>>> RLOptimizer::get_xfer_inputs() {
  return xfer_inputs;
}

std::vector<std::vector<std::vector<Op>>> RLOptimizer::get_xfer_outputs() {
  return xfer_outputs;
}

void RLOptimizer::get_xfer_locations(
                    GraphXfer* xfer,
                    int depth, Graph* graph,
                    std::vector<Graph*>& this_xfer_graphs,
                    std::vector<std::vector<Op>>& this_xfer_inputs,
                    std::vector<std::vector<Op>>& this_xfer_outputs,
                    std::set<size_t>& hashmap, int maxNumOps) {

  // try to match xfer to each Op in the graph
  // this populates this_xfer_graphs, this_xfer_inputs, this_xfer_outputs
  if (depth >= (int)xfer->srcOps.size()) {
    // this is run once all srcOps have been mapped
    // Create dst operators
    bool pass = true;
    std::vector<OpX*>::const_iterator dstIt;
    for (dstIt = xfer->dstOps.begin(); dstIt != xfer->dstOps.end(); dstIt++)
      if (pass) {
        OpX* dstOp = *dstIt;
        pass = (pass & xfer->create_new_operator(dstOp, dstOp->mapOp));
      }
    if (!pass) {
        return;
    }
    // Check that output tensors with external edges are mapped
    std::map<Op, OpX*, OpCompare>::const_iterator opIt;
    for (opIt = xfer->mappedOps.begin(); opIt != xfer->mappedOps.end(); opIt++) {
      // loop through all mapped ops Op -> OpX
      const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        // loop through all output edges of the mapped ops
        if (xfer->mappedOps.find(it->dstOp) == xfer->mappedOps.end()) {
          // only check this if the dstOp is not in the mapped Ops ("dstOp is external", i.e. not in the Xfer)
          // dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
          TensorX srcTen;
          srcTen.op = opIt->second;
          srcTen.idx = it->srcIdx;
          if (xfer->mappedOutputs.find(srcTen) == xfer->mappedOutputs.end()) {
            pass = false;
            return;
          }
        }
    }
    // Generate a new graph by applying xfer rule
    Graph* newGraph = xfer->create_new_graph(graph);
    // Check that the new graph should not have any loop
    if (newGraph->has_loop()) {
      delete newGraph;
      return;
    }
    // TODO: remove me for better performance
    assert(newGraph->check_correctness());
    if ((int)newGraph->inEdges.size() < maxNumOps) {
      if (hashmap.find(newGraph->hash()) == hashmap.end()) {

        // add graph
        hashmap.insert(newGraph->hash());
        this_xfer_graphs.push_back(newGraph);

        // add inputs
        // std::vector<Op> input_ops;
        // for (std::vector<OpX*>::iterator srcOpsIt = xfer->srcOps.begin(); srcOpsIt != xfer->srcOps.end(); srcOpsIt++) {
        //   OpX* srcOp = *srcOpsIt;

        //   input_ops.push_back(srcOp->mapOp);
        // }
        // this_xfer_inputs.push_back(input_ops);

        // // add outputs
        // std::vector<Op> output_ops;
        // for (std::vector<OpX*>::iterator dstOpsIt = xfer->dstOps.begin(); dstOpsIt != xfer->dstOps.end(); dstOpsIt++) {
        //   OpX* dstOp = *dstOpsIt;

        //   output_ops.push_back(dstOp->mapOp);
        // }
        // this_xfer_outputs.push_back(output_ops);
      }
    } else {
      printf("[Graph feedback][Warning] hit maxNumOps\n");
      delete newGraph;
    }
  } else {
    // This is called as long as depth < srcOps.size(), so all srcOps must be accounted for
    OpX* srcOp = xfer->srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      // Here we iterate over all Ops in the graph (we don't care for the edges right now)
      //printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
      if (xfer->can_match(srcOp, it->first, graph)
      // if the srcOpX matches the Op
      && (xfer->mappedOps.find(it->first) == xfer->mappedOps.end())) {
        // and the Op has not been mapped yet
        Op op = it->first;
        // Check mapOutput
        xfer->match(srcOp, op, graph);
        get_xfer_locations(xfer, depth + 1, graph, this_xfer_graphs, this_xfer_inputs, this_xfer_outputs, hashmap, maxNumOps);
        xfer->unmatch(srcOp, op, graph);
      }
    }
  }
}


float RLOptimizer::get_op_runtime(size_t guid) {
  Op op = this->graph->find_op_or_fail(guid);
  if (op.ptr == NULL)
      return 0.0f;
  return op.ptr->runtime;
}

float RLOptimizer::get_op_runtime_for_graph(Graph* graph, size_t guid) {
  Op op = graph->find_op_or_fail(guid);
  if (op.ptr == NULL)
      return 0.0f;
  return op.ptr->runtime;
}


// action
Graph* RLOptimizer::apply_xfer(int xfer_id, int location_id) {
  if (xfer_id < 0 || xfer_id >= xfer_graphs.size()) {
    printf("Invalid xfer ID: %u\n", xfer_id);
    return NULL;
  }

  std::vector<Graph*> &this_xfer_graphs = xfer_graphs[xfer_id];

  if (location_id < 0 || location_id >= this_xfer_graphs.size()) {
    printf("Invalid location ID: %u\n", location_id);
    return NULL;
  }

  this->graph = this_xfer_graphs[location_id];
  return this->graph;
}

// reward
float RLOptimizer::get_cost() {
  return this->graph->total_cost();
}

float RLOptimizer::get_measured_runtime(Graph* graph)
{
  std::map<Op, int, OpCompare> todos{};
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;

  int op_size = static_cast<int> (graph->inEdges.size());
  std::vector<Op> opList(op_size);
  size_t op_idx = 0;
  std::vector<OpBase*> opBaseList{};

  for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) {
        cnt++;
        // printf("srcOp.guid %d\n", it2->srcOp.guid);
      }
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      // opList.push_back(it->first);
       opList[op_idx++] = it->first;
  }
  std::cout << "Oplist size: " << opList.size() << ", inEdges: " << graph->inEdges.size() << std::endl;

  size_t i = 0;
  // while (i < opList.size()) {
  while (i < graph->inEdges.size()) {
    std::cout << "Op id: " << i << "\t";

    Op op = opList[i++];
    std::set<Edge, EdgeCompare> outList = graph->outEdges[op];
    std::set<Edge, EdgeCompare> inList = graph->inEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    assert(inList.size() > 0);
    OpBase* opPtr = NULL;
    // Step 1: prepare inputs
    Tensor inputs[MAX_NUM_INPUTS];
    if ((op.ptr->type == OP_INPUT) || (op.ptr->type == OP_WEIGHT)) {
      assert(inList.size() == 1);
      //Edge e = *inList.begin();
      //assert(e.srcOp.ptr == NULL); // NoOp's input must not be any Op
      Tensor t = op.ptr->inputs[0];
      size_t size = sizeof(DATATYPE);
      for (int j = 0; j < t.numDim; j++)
        size *= t.dim[j];

      // OP_INPUT
      if (op.ptr->type == OP_INPUT) {
        std::cout << "model malloc" << "\t";
        assert(t.data_ptr == NULL);
        t.data_ptr = (DATATYPE*) graph->model->allocate_memory(size);
        std::cout << "malloc OK" << "\t";
      } else {
        // OP_WEIGHT
        assert(t.data_ptr != NULL);
      }
      inputs[0] = t;
    } else {
      for (it2 = inList.begin(); it2 != inList.end(); it2++) {
        size_t idx2 = 0;
        for (idx2 = 0; idx2 < opList.size(); idx2++) {
          if (opList[idx2].guid == it2->srcOp.guid) break;
        }
        assert(idx2 < i);
        assert(inputs[it2->dstIdx].data_ptr == NULL); // No duplicated dstIdxes
        inputs[it2->dstIdx] = opBaseList[idx2]->outputs[it2->srcIdx];
      }
    }

    // Step 2: create Ops
    std::cout << "switch" << "\t";
    switch (op.ptr->type) {
      case OP_CONV2D:
      {
        //Conv2D* conv = (Conv2D*) op.ptr;
        Conv2D* conv = static_cast<Conv2D*>(op.ptr);
        assert(inList.size() == 2);
        printf("Padding: %d\n", conv->padding);
        opPtr = new Conv2D(graph->model, inputs[0], inputs[1],
                           conv->strideH, conv->strideW,
                           conv->padding, conv->activation);
#ifdef USE_CUDNN
        ((Conv2D*)opPtr)->fwdAlgo = conv->fwdAlgo;
#endif
        break;
      }
      case OP_MATMUL:
      {
        Matmul* matmul = (Matmul*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Matmul(graph->model, inputs[0], inputs[1], matmul->activation);
        break;
      }
      case OP_RESHAPE:
      {
        Reshape* reshape = (Reshape*) op.ptr;
        assert(inList.size() == 1);
        std::vector<int> shape;
        for (int i = 0; i < reshape->outputs[0].numDim; i++)
          shape.push_back(reshape->outputs[0].dim[i]);
        opPtr = new Reshape(graph->model, inputs[0], shape);
        break;
      }
      case OP_TRANSPOSE:
      {
        Transpose* transpose = (Transpose*) op.ptr;
        assert(inList.size() == 1);
        int ndim = inputs[0].numDim, permIdx = transpose->permIdx;
        std::vector<int> permVec;
        int permArray[MAX_DIM];
        for (int i = ndim - 1; i >= 0; i--) {
          permArray[i] = permIdx % ndim;
          permIdx = permIdx / ndim;
        }
        assert(permIdx == 0);
        for (int i = 0; i < ndim; i++)
          for (int j = i + 1; j < ndim; j++)
            assert(permArray[i] != permArray[j]);
        for (int i = 0; i < ndim; i++)
          permVec.push_back(permArray[i]);
        opPtr = new Transpose(graph->model, inputs[0], permVec, transpose->shuffle);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_MUL:
      {
        //Element* element = (Element*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Element(graph->model, op.ptr->type, inputs[0], inputs[1]);
        break;
      }
      case OP_ENLARGE:
      {
        //Enlarge* enlarge = (Enlarge*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Enlarge(graph->model, inputs[0], inputs[1]);
        break;
      }
      case OP_MERGE_GCONV:
      {
        MergeGConv* merge = (MergeGConv*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new MergeGConv(graph->model, inputs[0], merge->count);
        break;
      }
      case OP_POOL2D_MAX:
      case OP_POOL2D_AVG:
      {
        Pool2D* pool = (Pool2D*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Pool2D(graph->model, inputs[0], inputs[1], pool->type,
                           pool->kernelH, pool->kernelW,
                           pool->strideH, pool->strideW,
                           pool->padding, pool->activation);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      {
        Activation* act = (Activation*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Activation(graph->model, inputs[0], act->type, act->inPlace);
        break;
      }
      case OP_BATCHNORM:
      {
        BatchNorm* batchnorm = (BatchNorm*) op.ptr;
        assert(inList.size() == 5);
        opPtr = new BatchNorm(graph->model, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], batchnorm->epsilon);
        break;
      }
      case OP_SPLIT:
      {
        Split* split = (Split*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Split(graph->model, inputs[0], split->axis, split->sizes);
        break;
      }
      case OP_INPUT:
      case OP_WEIGHT:
      case OP_DROPOUT:
      {
        assert(inList.size() == 1);
        opPtr = new NoOp(graph->model, inputs[0], op.ptr->type);
        break;
      }
      case OP_CONCAT:
      {
        Concat* concat = (Concat*) op.ptr;
        opPtr = new Concat(graph->model, concat->axis, inList.size(), inputs, concat->needCopy);
        break;
      }
      default:
        printf("op.type = %d\n", op.ptr->type);
        assert(false);
    }
    // Step 3: map new Op
    std::cout << "CUDA malloc" << "\t";
    opPtr->map();
    std::cout << "CUDA malloc OK" << "\t";
    opBaseList.push_back(opPtr);
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      std::cout << "string: " << it2->dstOp.guid << "\t";
      // printf("myOp(%zu) dstOp(%zu) dstType(%d) dstTodos(%d)\n",
      //     it2->srcOp.guid, it2->dstOp.guid,
      //     it2->dstOp.ptr->type, todos[it2->dstOp]);
      if (todos[it2->dstOp] == 0) {
        // opList.push_back(it2->dstOp);
        opList[op_idx++] = it2->dstOp;
      }
    }
    std::cout << "size: " << opList.size() << "\t";
    std::cout << "done" << endl;
  }

  // #ifdef VERBOSE_PRINTS
  //   std::cout << "verbose" << endl;
  //   for (int i =0; i < opList.size(); i++) {
  //     printf("opList[%d]: guid(%zu) type(%d)\n", i, opList[i].guid,
  //            opList[i].ptr->type);
  //   }
  //   for (it = inEdges.begin(); it != inEdges.end(); it++) {
  //     printf("op: guid(%zu) type(%d)\n", it->first.guid, it->first.ptr->type);
  //     std::set<Edge, EdgeCompare> inList = it->second;
  //     std::set<Edge, EdgeCompare>::const_iterator it2;
  //     int cnt = 0;
  //     for (it2 = inList.begin(); it2 != inList.end(); it2++) {
  //       printf("    inEdge[%d]: srcOp(%zu) srcIdx(%d) dstOp(%zu) dstIdx(%d)\n", cnt++, it2->srcOp.guid, it2->srcIdx, it2->dstOp.guid, it2->dstIdx);
  //     }
  //   }
  // #endif

  std::cout << "assert" << endl;
  assert(opList.size() == graph->inEdges.size());
  assert(opList.size() == opBaseList.size());

  float result = graph->model->measure_oplist_runtime(opBaseList);
  // Now free GPU memory from the opList
  for (int i = 0; i < opBaseList.size(); i++) {
    OpBase* opBase = opBaseList[i];
    opBase->unmap();
    delete opBaseList[i];
    // free(opBase);
    opBase = nullptr;
  }

  return result;
}

void RLOptimizer::reproduce_taso_optimize(float alpha, int budget, int measure_interval)
{
  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
  Graph *best_g = bestGraph;
  candidates.push(best_g);
  float best_c = bestCost;
  printf("MetaFlow Cost = %.4lfms\n", best_c);
  printf("Input graph: end-to-end execution time =\n"
         "%.8lf ms (average of 100 runs)\n", best_g->run());
  printf("hash map size: %zu\n", hashmap.size());
  hashmap.clear();
  hashmap.insert(best_g->hash());

  int counter = 0;
  int maxNumOps = best_g->inEdges.size() * 2;
  //long long start_time = microsecond_timer();
  printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
  while (!candidates.empty()) {
    Graph *subGraph = candidates.top();
    candidates.pop();
    bool sub = false;
    if (subGraph->total_cost() < best_c) {
      delete best_g;
      best_c = subGraph->total_cost();
      best_g = subGraph;
      sub = true;
    }
    if (counter > budget) {
      // TODO: free all remaining candidates when budget exhausted
      break;
    }
    printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu sub: %d \n", counter, subGraph->total_cost(), best_c, candidates.size(), sub);
    if (counter != 0 && counter % measure_interval == 0 && counter > 198) {
      printf("        [%d] cost = %.4lf bestCost = %.4lf, measure_cost = %4lf, candidates.size() = %zu\n", counter, subGraph->total_cost(), best_c, subGraph->run(), candidates.size());
    }
    counter++;

    // ====== this matches TASO's call and thus can reproduce TASO optimization =====
    // for (size_t i = 0; i < xfers.size(); i++) {
    //   //for (size_t j = 0; j < xfers[i]->srcOps.size(); j++) {
    //   //  printf("srcOps[%zu]: type(%d)\n", j, xfers[i]->srcOps[j]->type);
    //   //}
    //   //for (size_t j = 0; j < xfers[i]->dstOps.size(); j++) {
    //   //  printf("dstOps[%zu]: type(%d)\n", j, xfers[i]->dstOps[j]->type);
    //   //}
    //   xfers[i]->run(0, subGraph, candidates, hashmap, best_c * alpha, maxNumOps);
    // }

    // ====== this implements our own candidate logic =====
    for (int i = 0; i < xfers.size(); i++) {
      // std::vector<Graph*> this_xfer_graphs;
      std::vector<std::vector<Op>> this_xfer_inputs;
      std::vector<std::vector<Op>> this_xfer_outputs;

      custom_run(xfers[i], 0, subGraph, candidates,
          this_xfer_inputs, this_xfer_outputs, hashmap,
          best_c * alpha, maxNumOps);
    }

    if (best_g != subGraph) {
      delete subGraph;
    }
  }
  best_g = best_g->preprocess_weights();
  printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");
  printf("bestCost = %.4lf\n", best_g->total_cost());
  printf("Optimized graph: end-to-end execution time =\n");
  printf("%.8lf ms (average of 100 runs)\n", best_g->run());
}


void RLOptimizer::custom_run(
                    GraphXfer* xfer,
                    int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::vector<std::vector<Op>>& this_xfer_inputs,
                    std::vector<std::vector<Op>>& this_xfer_outputs,
                    std::set<size_t>& hashmap, float threshold, int maxNumOps) {

  // try to match xfer to each Op in the graph
  if (depth >= (int)xfer->srcOps.size()) {
    // this is run once all srcOps have been mapped
    // Create dst operators
    bool pass = true;
    std::vector<OpX*>::const_iterator dstIt;
    for (dstIt = xfer->dstOps.begin(); dstIt != xfer->dstOps.end(); dstIt++)
      if (pass) {
        OpX* dstOp = *dstIt;
        pass = (pass & xfer->create_new_operator(dstOp, dstOp->mapOp));
      }
    if (!pass) {
        return;
    }
    // Check that output tensors with external edges are mapped
    std::map<Op, OpX*, OpCompare>::const_iterator opIt;
    for (opIt = xfer->mappedOps.begin(); opIt != xfer->mappedOps.end(); opIt++) {
      // loop through all mapped ops Op -> OpX
      const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        // loop through all output edges of the mapped ops
        if (xfer->mappedOps.find(it->dstOp) == xfer->mappedOps.end()) {
          // only check this if the dstOp is not in the mapped Ops ("dstOp is external", i.e. not in the Xfer)
          // dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
          TensorX srcTen;
          srcTen.op = opIt->second;
          srcTen.idx = it->srcIdx;
          if (xfer->mappedOutputs.find(srcTen) == xfer->mappedOutputs.end()) {
            pass = false;
            return;
          }
        }
    }
    // Generate a new graph by applying xfer rule
    Graph* newGraph = xfer->create_new_graph(graph);
    // Check that the new graph should not have any loop
    if (newGraph->has_loop()) {
      delete newGraph;
      return;
    }
    // TODO: remove me for better performance
    assert(newGraph->check_correctness());
    bool has_threshold = (threshold > 0 ) ? true:false;
    bool ok = true;
    if ((int)newGraph->inEdges.size() >= maxNumOps || (has_threshold && newGraph->total_cost() >= threshold)) {
      ok = false;
    }
    // append
    if (ok) {
      if (hashmap.find(newGraph->hash()) == hashmap.end()) {

        // add graph
        hashmap.insert(newGraph->hash());
        candidates.push(newGraph);
      }
    } else {
      if (!has_threshold) {
        printf("[Graph feedback][Warning] hit maxNumOps\n");
      }
      delete newGraph;
    }
  } else {
    // This is called as long as depth < srcOps.size(), so all srcOps must be accounted for
    OpX* srcOp = xfer->srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      // Here we iterate over all Ops in the graph (we don't care for the edges right now)
      //printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
      if (xfer->can_match(srcOp, it->first, graph)
      // if the srcOpX matches the Op
      && (xfer->mappedOps.find(it->first) == xfer->mappedOps.end())) {
        // and the Op has not been mapped yet
        Op op = it->first;
        // Check mapOutput
        xfer->match(srcOp, op, graph);
        custom_run(xfer, depth + 1, graph, candidates,
            this_xfer_inputs, this_xfer_outputs,
            hashmap, threshold, maxNumOps);
        xfer->unmatch(srcOp, op, graph);
      }
    }
  }
}


void RLOptimizer::viz_xfers(int idx) {
  auto& xfer = xfers[idx];
  auto src = xfer->get_srcOps();
  auto dst = xfer->get_dstOps();

  std::cout << "src Op" << std::endl;
  for (auto& i: src) {
    std::cout << "src " << i->type << std::endl;
    auto& ins = i->inputs;
    auto& outs = i->outputs;
    for (auto &j: ins) {
      std::cout << "ins " << j.idx << "\t";
    }
    for (auto &j: outs) {
      std::cout << "outs " << j.idx << "\t";
    }
    std::cout << std::endl;
  }


  std::cout << "dst Op" << std::endl;
  for (auto& i: dst) {
    std::cout << "dst " << i->type << std::endl;

    auto& ins = i->inputs;
    auto& outs = i->outputs;
    for (auto &j: ins) {
      std::cout << "ins " << j.idx << "\t";
    }
    for (auto &j: outs) {
      std::cout << "outs " << j.idx << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
