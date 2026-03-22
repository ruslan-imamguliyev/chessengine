#include <pybind11/pybind11.h>

#include "chessengine/alpha_beta_engine.hpp"
#include "chessengine/mcts_engine.hpp"
#include "chessengine/torch_evaluator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chessengine_cpp, m) {
    m.doc() = "C++ alpha-beta and MCTS engines backed by TorchScript";

    py::class_<chessengine::TorchEvaluator>(m, "TorchEvaluator")
        .def(py::init([](const std::string& model_path, const std::string& device) {
            torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
            return std::make_unique<chessengine::TorchEvaluator>(model_path, dev);
        }), py::arg("model_path"), py::arg("device") = "cpu");

    py::class_<chessengine::ABConfig>(m, "ABConfig")
        .def(py::init<>())
        .def_readwrite("depth", &chessengine::ABConfig::depth)
        .def_readwrite("time_limit_sec", &chessengine::ABConfig::time_limit_sec)
        .def_readwrite("quiescence_depth", &chessengine::ABConfig::quiescence_depth)
        .def_readwrite("tt_size", &chessengine::ABConfig::tt_size);

    py::class_<chessengine::AlphaBetaEngine>(m, "AlphaBetaEngine")
        .def(py::init<chessengine::ABConfig, chessengine::TorchEvaluator&>(), py::arg("config"),
             py::arg("evaluator"), py::keep_alive<1, 3>())
        .def("get_best_move_uci", &chessengine::AlphaBetaEngine::get_best_move_uci, py::arg("fen"));

    py::class_<chessengine::MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("num_simulations", &chessengine::MCTSConfig::num_simulations)
        .def_readwrite("leaf_parallelism", &chessengine::MCTSConfig::leaf_parallelism)
        .def_readwrite("puct_c", &chessengine::MCTSConfig::puct_c)
        .def_readwrite("prior_temperature", &chessengine::MCTSConfig::prior_temperature)
        .def_readwrite("time_limit_sec", &chessengine::MCTSConfig::time_limit_sec);

    py::class_<chessengine::MCTSEngine>(m, "MCTSEngine")
        .def(py::init<chessengine::MCTSConfig, chessengine::TorchEvaluator&>(), py::arg("config"),
             py::arg("evaluator"), py::keep_alive<1, 3>())
        .def("get_best_move_uci", &chessengine::MCTSEngine::get_best_move_uci, py::arg("fen"));
}
