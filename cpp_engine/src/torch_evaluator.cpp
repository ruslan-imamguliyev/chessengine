#include "chessengine/torch_evaluator.hpp"

#include <array>
#include <stdexcept>

namespace chessengine {

namespace {
constexpr int kPlanes = 18;
constexpr int kBoardSize = 8;

int square_to_file(int square) { return square % 8; }
int square_to_rank(int square) { return square / 8; }

int piece_plane(chess::Piece piece) {
    using chess::PieceType;
    if (piece == chess::Piece::NONE) return -1;

    const bool white = piece.color() == chess::Color::WHITE;
    switch (piece.type()) {
        case PieceType::PAWN:
            return white ? 0 : 6;
        case PieceType::KNIGHT:
            return white ? 1 : 7;
        case PieceType::BISHOP:
            return white ? 2 : 8;
        case PieceType::ROOK:
            return white ? 3 : 9;
        case PieceType::QUEEN:
            return white ? 4 : 10;
        case PieceType::KING:
            return white ? 5 : 11;
        default:
            return -1;
    }
}
}  // namespace

TorchEvaluator::TorchEvaluator(const std::string& model_path, torch::Device device) : device_(std::move(device)) {
    module_ = torch::jit::load(model_path, device_);
    module_.eval();
}

torch::Tensor TorchEvaluator::board_to_tensor(const chess::Board& board) const {
    auto tensor = torch::zeros({kPlanes, kBoardSize, kBoardSize}, torch::TensorOptions().dtype(torch::kFloat32));
    auto acc = tensor.accessor<float, 3>();

    for (int sq = 0; sq < 64; ++sq) {
        const auto piece = board.at(chess::Square(sq));
        const int plane = piece_plane(piece);
        if (plane < 0) continue;

        const int rank = 7 - square_to_rank(sq);
        const int file = square_to_file(sq);
        acc[plane][rank][file] = 1.0f;
    }

    if (board.sideToMove() == chess::Color::WHITE) {
        for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) acc[12][r][f] = 1.0f;
    }

    if (board.castlingRights().has(chess::Color::WHITE, chess::CastleSide::KING_SIDE)) {
        for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) acc[13][r][f] = 1.0f;
    }
    if (board.castlingRights().has(chess::Color::WHITE, chess::CastleSide::QUEEN_SIDE)) {
        for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) acc[14][r][f] = 1.0f;
    }
    if (board.castlingRights().has(chess::Color::BLACK, chess::CastleSide::KING_SIDE)) {
        for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) acc[15][r][f] = 1.0f;
    }
    if (board.castlingRights().has(chess::Color::BLACK, chess::CastleSide::QUEEN_SIDE)) {
        for (int r = 0; r < 8; ++r) for (int f = 0; f < 8; ++f) acc[16][r][f] = 1.0f;
    }

    auto ep = board.enpassantSq();
    if (ep != chess::Square::NO_SQ) {
        const int sq = static_cast<int>(ep);
        acc[17][7 - square_to_rank(sq)][square_to_file(sq)] = 1.0f;
    }

    return tensor;
}

float TorchEvaluator::evaluate(const chess::Board& board) {
    auto input = board_to_tensor(board).unsqueeze(0).to(device_);
    std::vector<torch::jit::IValue> inputs{input};
    auto output = module_.forward(inputs).toTensor().item<float>();

    if (board.sideToMove() == chess::Color::BLACK) output = -output;
    return output;
}

std::vector<float> TorchEvaluator::evaluate_batch(const std::vector<chess::Board>& boards) {
    if (boards.empty()) return {};

    std::vector<torch::Tensor> encoded;
    encoded.reserve(boards.size());
    for (const auto& b : boards) encoded.push_back(board_to_tensor(b));

    auto input = torch::stack(encoded).to(device_);
    std::vector<torch::jit::IValue> inputs{input};
    auto out = module_.forward(inputs).toTensor().to(torch::kCPU).squeeze(1);

    std::vector<float> scores(boards.size(), 0.0f);
    for (std::size_t i = 0; i < boards.size(); ++i) {
        scores[i] = out[i].item<float>();
        if (boards[i].sideToMove() == chess::Color::BLACK) scores[i] = -scores[i];
    }
    return scores;
}

}  // namespace chessengine
