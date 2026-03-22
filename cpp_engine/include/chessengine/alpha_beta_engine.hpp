#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include <chess.hpp>

#include "chessengine/torch_evaluator.hpp"

namespace chessengine {

struct ABConfig {
    int depth = 4;
    double time_limit_sec = 0.0;
    int quiescence_depth = 4;
    std::size_t tt_size = 500000;
};

class AlphaBetaEngine {
public:
    AlphaBetaEngine(ABConfig config, TorchEvaluator& evaluator);

    std::optional<chess::Move> get_best_move(const chess::Board& board);
    std::string get_best_move_uci(const std::string& fen);

private:
    enum TTFlag : std::uint8_t { EXACT = 0, LOWERBOUND = 1, UPPERBOUND = 2 };

    struct TTEntry {
        int depth = -1;
        float score = 0.0f;
        chess::Move move = chess::Move::NO_MOVE;
        TTFlag flag = EXACT;
    };

    ABConfig config_;
    TorchEvaluator& evaluator_;
    std::unordered_map<std::uint64_t, TTEntry> tt_;

    std::int64_t nodes_searched_ = 0;

    std::vector<chess::Move> order_moves(const chess::Board& board, const chess::Movelist& moves,
                                         chess::Move tt_best = chess::Move::NO_MOVE) const;

    float quiescence(chess::Board& board, float alpha, float beta, int depth);
    std::pair<float, chess::Move> alpha_beta(chess::Board& board, int depth, float alpha, float beta, bool root);

    static bool is_terminal(const chess::Board& board, float& score);
};

}  // namespace chessengine
