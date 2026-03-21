#include "chessengine/torch_evaluator.hpp"

#include <sstream>
#include <stdexcept>

namespace chessengine {

namespace {
constexpr int kPlanes = 18;
constexpr int kBoardSize = 8;

int piece_plane_from_fen_char(char piece) {
    switch (piece) {
        case 'P':
            return 0;
        case 'N':
            return 1;
        case 'B':
            return 2;
        case 'R':
            return 3;
        case 'Q':
            return 4;
        case 'K':
            return 5;
        case 'p':
            return 6;
        case 'n':
            return 7;
        case 'b':
            return 8;
        case 'r':
            return 9;
        case 'q':
            return 10;
        case 'k':
            return 11;
        default:
            return -1;
    }
}
}  // namespace

TorchEvaluator::TorchEvaluator(const std::string& model_path, torch::Device device) : device_(std::move(device)) {
    std::cout << "Loading model..." << std::endl;
    module_ = torch::jit::load(model_path, device_);
    std::cout << "Loaded model!" << std::endl;
    module_.eval();
}

torch::Tensor TorchEvaluator::board_to_tensor(const chess::Board& board) const {
    auto tensor = torch::zeros({kPlanes, kBoardSize, kBoardSize}, torch::TensorOptions().dtype(torch::kFloat32));
    auto acc = tensor.accessor<float, 3>();

    const std::string fen = board.getFen();
    std::istringstream iss(fen);
    std::string board_part, side, castling, ep;
    iss >> board_part >> side >> castling >> ep;

    int r = 0;
    int c = 0;
    for (char ch : board_part) {
        if (ch == '/') {
            ++r;
            c = 0;
            continue;
        }
        if (ch >= '1' && ch <= '8') {
            c += (ch - '0');
            continue;
        }
        const int plane = piece_plane_from_fen_char(ch);
        if (plane >= 0 && r >= 0 && r < 8 && c >= 0 && c < 8) {
            acc[plane][r][c] = 1.0f;
        }
        ++c;
    }

    if (side == "w") {
        for (int rr = 0; rr < 8; ++rr) {
            for (int cc = 0; cc < 8; ++cc) {
                acc[12][rr][cc] = 1.0f;
            }
        }
    }

    auto fill_plane = [&](int plane) {
        for (int rr = 0; rr < 8; ++rr) {
            for (int cc = 0; cc < 8; ++cc) {
                acc[plane][rr][cc] = 1.0f;
            }
        }
    };

    if (castling.find('K') != std::string::npos) fill_plane(13);
    if (castling.find('Q') != std::string::npos) fill_plane(14);
    if (castling.find('k') != std::string::npos) fill_plane(15);
    if (castling.find('q') != std::string::npos) fill_plane(16);

    if (ep.size() == 2 && ep[0] >= 'a' && ep[0] <= 'h' && ep[1] >= '1' && ep[1] <= '8') {
        const int file = ep[0] - 'a';
        const int rank = 8 - (ep[1] - '0');
        if (rank >= 0 && rank < 8 && file >= 0 && file < 8) {
            acc[17][rank][file] = 1.0f;
        }
    }

    return tensor;
}

float TorchEvaluator::evaluate(const chess::Board& board) {
    auto input = board_to_tensor(board).unsqueeze(0).to(device_);
    std::vector<torch::jit::IValue> inputs{input};
    auto output = module_.forward(inputs).toTensor().item<float>();

    if (board.getFen().find(" b ") != std::string::npos) output = -output;
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
        if (boards[i].getFen().find(" b ") != std::string::npos) scores[i] = -scores[i];
    }
    return scores;
}

}  // namespace chessengine
