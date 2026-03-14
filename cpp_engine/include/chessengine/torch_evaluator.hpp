#pragma once

#include <string>
#include <vector>

#include <chess.hpp>
#include <torch/script.h>

namespace chessengine {

class TorchEvaluator {
public:
    explicit TorchEvaluator(const std::string& model_path, torch::Device device = torch::kCPU);

    float evaluate(const chess::Board& board);
    std::vector<float> evaluate_batch(const std::vector<chess::Board>& boards);

private:
    torch::jit::script::Module module_;
    torch::Device device_;

    torch::Tensor board_to_tensor(const chess::Board& board) const;
};

}  // namespace chessengine
