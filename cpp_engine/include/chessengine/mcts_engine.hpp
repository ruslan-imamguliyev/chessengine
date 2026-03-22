#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <chess.hpp>

#include "chessengine/torch_evaluator.hpp"

namespace chessengine {

struct MCTSConfig {
    int num_simulations = 800;
    int leaf_parallelism = 16;
    float puct_c = 1.4f;
    float prior_temperature = 1.0f;
    double time_limit_sec = 0.0;
};

struct MCTSNode {
    chess::Board board;
    MCTSNode* parent = nullptr;
    chess::Move move = chess::Move::NO_MOVE;
    float prior = 0.0f;
    int visits = 0;
    float value_sum = 0.0f;
    std::vector<std::unique_ptr<MCTSNode>> children;

    [[nodiscard]] float q_value() const;
    [[nodiscard]] bool is_expanded() const;
};

class MCTSEngine {
public:
    MCTSEngine(MCTSConfig config, TorchEvaluator& evaluator);

    std::optional<chess::Move> get_best_move(const chess::Board& board);
    std::string get_best_move_uci(const std::string& fen);

private:
    MCTSConfig config_;
    TorchEvaluator& evaluator_;

    static std::vector<float> softmax(const std::vector<float>& values, float temperature);
    static std::optional<float> terminal_value(const chess::Board& board);

    MCTSNode* select_child(MCTSNode* node) const;
    MCTSNode* select_leaf(MCTSNode* root) const;

    std::vector<std::pair<MCTSNode*, float>> expand_batch(const std::vector<MCTSNode*>& leaves);
    static void backup(MCTSNode* node, float value);
};

}  // namespace chessengine
