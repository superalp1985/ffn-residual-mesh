#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    fs::path model;
    fs::path out_dir;
    std::string prompt;
    int n_gpu_layers = 0;
    int n_threads = 8;
    bool capture_ffn = false;
};

struct CapturedTensor {
    std::string name;
    std::string type;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

struct CaptureState {
    std::vector<CapturedTensor> tensors;
};

static void usage(const char * argv0) {
    std::cerr << "usage: " << argv0 << " --model MODEL --out DIR --prompt TEXT [--ngl N] [--threads N] [--capture-ffn]\n";
}

static bool parse_args(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--model" && i + 1 < argc) args.model = argv[++i];
        else if (key == "--out" && i + 1 < argc) args.out_dir = argv[++i];
        else if (key == "--prompt" && i + 1 < argc) args.prompt = argv[++i];
        else if (key == "--ngl" && i + 1 < argc) args.n_gpu_layers = std::stoi(argv[++i]);
        else if (key == "--threads" && i + 1 < argc) args.n_threads = std::stoi(argv[++i]);
        else if (key == "--capture-ffn") args.capture_ffn = true;
        else {
            usage(argv[0]);
            return false;
        }
    }
    if (args.model.empty() || args.out_dir.empty() || args.prompt.empty()) {
        usage(argv[0]);
        return false;
    }
    return true;
}

static bool eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<CaptureState *>(user_data);
    const std::string name = tensor->name;
    // attn_post_norm is the exact vector consumed by the dense FFN in Qwen3.5.
    const bool selected = name.find("ffn_") != std::string::npos || name.rfind("attn_post_norm-", 0) == 0;
    if (ask) {
        return selected;
    }
    if (!selected) {
        return true;
    }

    CapturedTensor item;
    item.name = name;
    item.type = ggml_type_name(tensor->type);
    const int n_dims = ggml_n_dims(tensor);
    item.shape.reserve(n_dims);
    for (int i = 0; i < n_dims; ++i) {
        item.shape.push_back(tensor->ne[i]);
    }
    const size_t n_bytes = ggml_nbytes(tensor);
    item.data.resize(n_bytes);
    ggml_backend_tensor_get(tensor, item.data.data(), 0, n_bytes);
    capture->tensors.push_back(std::move(item));
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    fs::create_directories(args.out_dir);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = args.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(args.model.string().c_str(), model_params);
    if (!model) {
        std::cerr << "failed to load model\n";
        llama_backend_free();
        return 3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, args.prompt.c_str(), args.prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        std::cerr << "failed to tokenize prompt\n";
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, args.prompt.c_str(), args.prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
        std::cerr << "failed to tokenize prompt\n";
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = std::max<uint32_t>(n_prompt + 8, 256);
    ctx_params.n_batch = std::max<uint32_t>(n_prompt, 1);
    ctx_params.n_ubatch = std::min<uint32_t>(ctx_params.n_batch, 128);
    ctx_params.n_threads = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads;
    ctx_params.no_perf = false;

    CaptureState capture;
    if (args.capture_ffn) {
        ctx_params.cb_eval = eval_callback;
        ctx_params.cb_eval_user_data = &capture;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "failed to create context\n";
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    const uint32_t n_layer = llama_model_n_layer(model);
    const uint32_t n_embd = llama_model_n_embd(model);
    for (uint32_t layer = 0; layer < n_layer; ++layer) {
        llama_set_embeddings_layer_inp(ctx, layer, true);
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    const int decode_result = llama_decode(ctx, batch);
    if (decode_result != 0) {
        std::cerr << "llama_decode failed: " << decode_result << "\n";
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 7;
    }

    {
        std::ofstream meta(args.out_dir / "manifest.json", std::ios::binary);
        meta << "{\n"
             << "  \"model\": \"" << args.model.generic_string() << "\",\n"
             << "  \"tokens\": " << tokens.size() << ",\n"
             << "  \"hidden\": " << n_embd << ",\n"
             << "  \"layers\": " << n_layer << ",\n"
             << "  \"ngl\": " << args.n_gpu_layers << ",\n"
             << "  \"threads\": " << args.n_threads << "\n"
             << "}\n";
        std::ofstream token_file(args.out_dir / "tokens.i32", std::ios::binary);
        token_file.write(reinterpret_cast<const char *>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(llama_token)));
    }

    for (uint32_t layer = 0; layer < n_layer; ++layer) {
        float * values = llama_get_embeddings_layer_inp(ctx, layer);
        if (!values) {
            std::cerr << "missing layer input: " << layer << "\n";
            continue;
        }
        const size_t count = static_cast<size_t>(tokens.size()) * n_embd;
        const std::string filename = std::string("layer_") + (layer < 10 ? "0" : "") + std::to_string(layer) + "_input.f32";
        std::ofstream output(args.out_dir / filename, std::ios::binary);
        output.write(reinterpret_cast<const char *>(values), static_cast<std::streamsize>(count * sizeof(float)));
    }

    if (args.capture_ffn) {
        std::ofstream manifest(args.out_dir / "ffn_tensors.json", std::ios::binary);
        manifest << "{\n  \"count\": " << capture.tensors.size() << ",\n  \"tensors\": [\n";
        for (size_t i = 0; i < capture.tensors.size(); ++i) {
            const auto & item = capture.tensors[i];
            const std::string filename = "tensor_" + std::to_string(i) + ".bin";
            std::ofstream data_file(args.out_dir / filename, std::ios::binary);
            data_file.write(reinterpret_cast<const char *>(item.data.data()), static_cast<std::streamsize>(item.data.size()));
            manifest << "    {\"name\": \"" << item.name << "\", \"type\": \"" << item.type << "\", \"shape\": [";
            for (size_t j = 0; j < item.shape.size(); ++j) {
                if (j > 0) manifest << ", ";
                manifest << item.shape[j];
            }
            manifest << "], \"bytes\": " << item.data.size() << ", \"file\": \"" << filename << "\"}";
            manifest << (i + 1 == capture.tensors.size() ? "\n" : ",\n");
        }
        manifest << "  ]\n}\n";
        std::cout << "captured ffn tensors=" << capture.tensors.size() << "\n";
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::cout << "captured layers=" << n_layer << " tokens=" << tokens.size() << " hidden=" << n_embd << "\n";
    return 0;
}
