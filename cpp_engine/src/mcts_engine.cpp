#include "chessengine/mcts_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace chessengine {

float MCTSNode::q_value() const {
    if (visits == 0) return 0.0f;
    return value_sum / static_cast<float>(visits);
}

bool MCTSNode::is_expanded() const { return !children.empty(); }

MCTSEngine::MCTSEngine(MCTSConfig config, TorchEvaluator& evaluator) : config_(config), evaluator_(evaluator) {}

std::vector<float> MCTSEngine::softmax(const std::vector<float>& values, float temperature) {
    if (values.empty()) return {};

    const float t = std::max(1e-3f, temperature);
    const float max_v = *std::max_element(values.begin(), values.end());

    std::vector<float> exps(values.size(), 0.0f);
    float total = 0.0f;
    for (std::size_t i = 0; i < values.size(); ++i) {
        exps[i] = std::exp((values[i] - max_v) / t);
        total += exps[i];
    }

    if (total <= 0.0f) return std::vector<float>(values.size(), 1.0f / values.size());
    for (auto& v : exps) v /= total;
    return exps;
}

std::optional<float> MCTSEngine::terminal_value(const chess::Board& board) {
    const auto state = board.isGameOver();
    if (state.second == chess::GameResultReason::CHECKMATE) return -1.0f;
    if (state.first != chess::GameResult::NONE) return 0.0f;
    return std::nullopt;
}

MCTSNode* MCTSEngine::select_child(MCTSNode* node) const {
    const float sqrt_parent = std::sqrt(static_cast<float>(std::max(1, node->visits)));
    float best_score = -std::numeric_limits<float>::infinity();
    MCTSNode* best = nullptr;

    for (auto& child : node->children) {
        const float q = -child->q_value();
        const float u = config_.puct_c * child->prior * sqrt_parent / (1.0f + static_cast<float>(child->visits));
        const float score = q + u;
        if (score > best_score) {
            best_score = score;
            best = child.get();
        }
    }

    return best;
}

MCTSNode* MCTSEngine::select_leaf(MCTSNode* root) const {
    MCTSNode* node = root;
    while (node != nullptr && node->is_expanded()) node = select_child(node);
    return node == nullptr ? root : node;
}

std::vector<std::pair<MCTSNode*, float>> MCTSEngine::expand_batch(const std::vector<MCTSNode*>& leaves) {
    std::vector<std::pair<MCTSNode*, float>> outcomes;
    std::vector<chess::Board> boards_for_eval;
    std::vector<std::pair<MCTSNode*, std::vector<chess::Move>>> meta;

    for (auto* leaf : leaves) {
        if (!leaf) continue;

        if (auto term = terminal_value(leaf->board); term.has_value()) {
            outcomes.emplace_back(leaf, term.value());
            continue;
        }

        chess::Movelist legal;
        chess::movegen::legalmoves(legal, leaf->board);
        if (legal.empty()) {
            outcomes.emplace_back(leaf, 0.0f);
            continue;
        }

        std::vector<chess::Move> moves;
        moves.reserve(legal.size());
        for (const auto& move : legal) {
            moves.push_back(move);
            chess::Board child = leaf->board;
            child.makeMove(move);
            boards_for_eval.push_back(child);
        }

        meta.emplace_back(leaf, std::move(moves));
    }

    if (boards_for_eval.empty()) return outcomes;

    auto child_scores = evaluator_.evaluate_batch(boards_for_eval);

    std::size_t idx = 0;
    for (auto& [leaf, moves] : meta) {
        std::vector<float> parent_scores;
        parent_scores.reserve(moves.size());

        for (std::size_t i = 0; i < moves.size(); ++i) {
            parent_scores.push_back(-child_scores[idx++]);
        }

        auto priors = softmax(parent_scores, config_.prior_temperature);

        for (std::size_t i = 0; i < moves.size(); ++i) {
            chess::Board child = leaf->board;
            child.makeMove(moves[i]);

            auto node = std::make_unique<MCTSNode>();
            node->board = child;
            node->parent = leaf;
            node->move = moves[i];
            node->prior = priors[i];
            leaf->children.push_back(std::move(node));
        }

        const float leaf_value = parent_scores.empty() ? 0.0f : *std::max_element(parent_scores.begin(), parent_scores.end());
        outcomes.emplace_back(leaf, leaf_value);
    }

    return outcomes;
}

void MCTSEngine::backup(MCTSNode* node, float value) {
    auto* cur = node;
    float v = value;
    while (cur != nullptr) {
        cur->visits += 1;
        cur->value_sum += v;
        cur = cur->parent;
        v = -v;
    }
}

std::optional<chess::Move> MCTSEngine::get_best_move(const chess::Board& board) {
    auto root = std::make_unique<MCTSNode>();
    root->board = board;
    root->prior = 1.0f;

    const auto start = std::chrono::steady_clock::now();
    for (int sim = 0; sim < std::max(1, config_.num_simulations); ++sim) {
        if (config_.time_limit_sec > 0.0) {
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (elapsed > config_.time_limit_sec) break;
        }

        std::vector<MCTSNode*> leaves;
        leaves.reserve(std::max(1, config_.leaf_parallelism));
        for (int i = 0; i < std::max(1, config_.leaf_parallelism); ++i) {
            leaves.push_back(select_leaf(root.get()));
        }

        auto outcomes = expand_batch(leaves);
        for (auto& [leaf, value] : outcomes) backup(leaf, value);
    }

    if (root->children.empty()) {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, root->board);
        if (legal.empty()) return std::nullopt;
        return legal.front();
    }

    auto best_it = std::max_element(root->children.begin(), root->children.end(),
                                    [](const auto& a, const auto& b) { return a->visits < b->visits; });

    return (*best_it)->move;
}

std::string MCTSEngine::get_best_move_uci(const std::string& fen) {
    chess::Board board(fen);
    auto move = get_best_move(board);
    if (!move.has_value()) return "0000";
    return chess::uci::moveToUci(move.value());
}

}  // namespace chessengine
