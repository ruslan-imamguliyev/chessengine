#include "chessengine/alpha_beta_engine.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>

namespace chessengine {

namespace {
int piece_value_from_promo_char(char promo) {
    switch (promo) {
        case 'n':
        case 'N':
            return 320;
        case 'b':
        case 'B':
            return 330;
        case 'r':
        case 'R':
            return 500;
        case 'q':
        case 'Q':
            return 900;
        default:
            return 0;
    }
}

}  // namespace

AlphaBetaEngine::AlphaBetaEngine(ABConfig config, TorchEvaluator& evaluator)
    : config_(config), evaluator_(evaluator) {
    tt_.reserve(config_.tt_size);
}

std::vector<chess::Move> AlphaBetaEngine::order_moves(const chess::Board& board, const chess::Movelist& moves,
                                                      chess::Move tt_best) const {
    std::vector<std::pair<int, chess::Move>> scored;
    scored.reserve(moves.size());

    for (const auto& move : moves) {
        int score = 0;
        if (tt_best != chess::Move::NO_MOVE && move == tt_best) {
            score = 1'000'000;
        } else {
            if (board.isCapture(move)) {
                score += 5'000;
            }
            const std::string uci = chess::uci::moveToUci(move);
            if (uci.size() == 5) {
                score += piece_value_from_promo_char(uci[4]);
            }
            chess::Board copy = board;
            copy.makeMove(move);
            if (copy.inCheck()) score += 50;
        }
        scored.emplace_back(score, move);
    }

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<chess::Move> ordered;
    ordered.reserve(scored.size());
    for (const auto& entry : scored) ordered.push_back(entry.second);
    return ordered;
}

bool AlphaBetaEngine::is_terminal(const chess::Board& board, float& score) {
    chess::Movelist legal;
    chess::movegen::legalmoves(legal, board);
    if (!legal.empty()) return false;

    if (board.inCheck()) {
        score = -1.0f;
        return true;
    }

    score = 0.0f;
    return true;
}

float AlphaBetaEngine::quiescence(chess::Board& board, float alpha, float beta, int depth) {
    if (depth == 0) return evaluator_.evaluate(board);

    float stand_pat = evaluator_.evaluate(board);
    if (stand_pat >= beta) return beta;
    alpha = std::max(alpha, stand_pat);

    chess::Movelist legal;
    chess::movegen::legalmoves(legal, board);

    chess::Movelist tactical;
    for (const auto& move : legal) {
        bool is_tactical = board.isCapture(move);
        if (!is_tactical) {
            chess::Board copy = board;
            copy.makeMove(move);
            is_tactical = copy.inCheck();
        }
        if (is_tactical) tactical.add(move);
    }

    if (tactical.empty()) return stand_pat;

    const auto ordered = order_moves(board, tactical);
    for (const auto& move : ordered) {
        board.makeMove(move);
        float score = -quiescence(board, -beta, -alpha, depth - 1);
        board.unmakeMove(move);

        if (score >= beta) return beta;
        alpha = std::max(alpha, score);
    }

    return alpha;
}

std::pair<float, chess::Move> AlphaBetaEngine::alpha_beta(chess::Board& board, int depth, float alpha, float beta,
                                                           bool root) {
    ++nodes_searched_;

    const auto hash = board.hash();
    auto tt_it = tt_.find(hash);
    chess::Move tt_move = chess::Move::NO_MOVE;
    if (tt_it != tt_.end()) {
        const auto& entry = tt_it->second;
        tt_move = entry.move;
        if (!root && entry.depth >= depth) {
            if (entry.flag == EXACT) return {entry.score, entry.move};
            if (entry.flag == LOWERBOUND && entry.score >= beta) return {entry.score, entry.move};
            if (entry.flag == UPPERBOUND && entry.score <= alpha) return {entry.score, entry.move};
        }
    }

    float terminal_score = 0.0f;
    if (is_terminal(board, terminal_score)) return {terminal_score, chess::Move::NO_MOVE};

    if (depth == 0) return {quiescence(board, alpha, beta, config_.quiescence_depth), chess::Move::NO_MOVE};

    chess::Movelist legal;
    chess::movegen::legalmoves(legal, board);
    if (legal.empty()) return {0.0f, chess::Move::NO_MOVE};

    auto ordered = order_moves(board, legal, tt_move);

    chess::Move best = ordered.front();
    float best_score = -std::numeric_limits<float>::infinity();
    TTFlag flag = UPPERBOUND;

    for (const auto& move : ordered) {
        board.makeMove(move);
        auto [child_score, _] = alpha_beta(board, depth - 1, -beta, -alpha, false);
        const float score = -child_score;
        board.unmakeMove(move);

        if (score > best_score) {
            best_score = score;
            best = move;
        }
        if (score > alpha) {
            alpha = score;
            flag = EXACT;
        }
        if (alpha >= beta) {
            flag = LOWERBOUND;
            break;
        }
    }

    tt_[hash] = TTEntry{depth, best_score, best, flag};
    return {best_score, best};
}

std::optional<chess::Move> AlphaBetaEngine::get_best_move(const chess::Board& board) {
    chess::Board copy = board;
    std::optional<chess::Move> best;
    float best_score = 0.0f;

    const auto start = std::chrono::steady_clock::now();
    for (int depth = 1; depth <= config_.depth; ++depth) {
        if (config_.time_limit_sec > 0.0) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed > config_.time_limit_sec) break;
        }

        auto [score, move] = alpha_beta(copy, depth, -std::numeric_limits<float>::infinity(),
                                        std::numeric_limits<float>::infinity(), true);
        if (move != chess::Move::NO_MOVE) {
            best = move;
            best_score = score;
        }
    }

    (void)best_score;
    return best;
}

std::string AlphaBetaEngine::get_best_move_uci(const std::string& fen) {
    chess::Board board(fen);
    auto move = get_best_move(board);
    if (!move.has_value()) return "0000";
    return chess::uci::moveToUci(move.value());
}

}  // namespace chessengine
