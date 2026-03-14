cmake_minimum_required(VERSION 3.22)
project(chessengine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CHESSENGINE_BUILD_PYTHON "Build pybind11 extension" ON)

include(FetchContent)
FetchContent_Declare(
  chesslib
  GIT_REPOSITORY https://github.com/Disservin/chess-library.git
  GIT_TAG v0.3.4
)
FetchContent_MakeAvailable(chesslib)

find_package(Torch REQUIRED)

add_library(chessengine_core
  src/torch_evaluator.cpp
  src/alpha_beta_engine.cpp
  src/mcts_engine.cpp
)

target_include_directories(chessengine_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(chessengine_core
  PUBLIC
    chess::chess
    ${TORCH_LIBRARIES}
)

target_compile_options(chessengine_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)

if(CHESSENGINE_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(chessengine_cpp MODULE python/bindings.cpp)
  target_link_libraries(chessengine_cpp PRIVATE chessengine_core)

  if(MSVC)
    target_compile_options(chessengine_cpp PRIVATE /EHsc)
  endif()
endif()
