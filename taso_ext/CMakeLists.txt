cmake_minimum_required(VERSION 3.2)
project(TASORL LANGUAGES CXX CUDA)


# file(GLOB_RECURSE TASORL_SRCS
#   src/*.cc
#   )
set(TASORL_SRCS src/graph_feedback.cc)

# to use cuda
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

#find_package(Protobuf REQUIRED)
#message(STATUS "PROTOBUF=" ${PROTOBUF_LIBRARY})
#include_directories(${PROTOBUF_INCLUDE_DIR})
#protobuf_generate_cpp(PROTO_CPPS PROTO_HDRS src/rules.proto)
#list(APPEND TASORL_LINK_LIBS ${PROTOBUF_LIBRARY})

list(APPEND TASORL_LINK_LIBS "/usr/local/lib/libtaso_runtime.so")

# compile taso_rl as a shared lib
add_library(taso_rl SHARED ${TASORL_SRCS})
target_link_libraries(taso_rl ${TASORL_LINK_LIBS})
target_link_libraries(taso_rl ${CUDA_LIBRARIES})
set_target_properties(taso_rl
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_mem_leak src/test_graph_feedback.cc)
target_link_libraries(test_mem_leak ${TASORL_LINK_LIBS})
target_link_libraries(test_mem_leak taso_rl)
target_link_libraries(test_mem_leak ${CUDA_LIBRARIES})
set_target_properties(test_mem_leak
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_taso_opt src/test_taso_opt.cc)
target_link_libraries(test_taso_opt ${TASORL_LINK_LIBS})
target_link_libraries(test_taso_opt taso_rl)
target_link_libraries(test_taso_opt ${CUDA_LIBRARIES})
set_target_properties(test_taso_opt
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_xfers src/test_xfers.cc)
target_link_libraries(test_xfers ${TASORL_LINK_LIBS})
target_link_libraries(test_xfers taso_rl)
target_link_libraries(test_xfers ${CUDA_LIBRARIES})
set_target_properties(test_xfers
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


install(TARGETS taso_rl
    LIBRARY DESTINATION lib)
