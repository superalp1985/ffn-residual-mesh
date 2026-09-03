#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char * expr, const char * file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << expr << " failed at " << file << ':' << line << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

#pragma pack(push, 1)
struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t layer;
    std::uint32_t tokens;
    std::uint32_t hidden;
    std::uint32_t rank;
    std::uint32_t keep;
    std::uint32_t reserved;
};
#pragma pack(pop)

struct Artifact {
    Header header{};
    std::vector<__half> basis;
    std::vector<__half> mean;
    std::vector<__half> base;
    std::vector<__half> coeff;
    std::vector<std::uint16_t> indices;
    std::vector<float> target;
    std::vector<float> reference;
};

template <typename T>
std::vector<T> read_vector(std::ifstream & input, std::size_t count) {
    std::vector<T> values(count);
    input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!input) {
        throw std::runtime_error("artifact ended before all tensors were read");
    }
    return values;
}

Artifact read_artifact(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open artifact: " + path);
    }
    Artifact artifact;
    input.read(reinterpret_cast<char *>(&artifact.header), sizeof(artifact.header));
    if (!input || std::memcmp(artifact.header.magic, "FFNRES01", 8) != 0 || artifact.header.version != 1) {
        throw std::runtime_error("invalid artifact header");
    }
    const auto & h = artifact.header;
    artifact.basis = read_vector<__half>(input, static_cast<std::size_t>(h.rank) * h.hidden);
    artifact.mean = read_vector<__half>(input, h.hidden);
    artifact.base = read_vector<__half>(input, static_cast<std::size_t>(h.tokens) * h.hidden);
    artifact.coeff = read_vector<__half>(input, static_cast<std::size_t>(h.tokens) * h.keep);
    artifact.indices = read_vector<std::uint16_t>(input, static_cast<std::size_t>(h.tokens) * h.keep);
    artifact.target = read_vector<float>(input, static_cast<std::size_t>(h.tokens) * h.hidden);
    artifact.reference = read_vector<float>(input, static_cast<std::size_t>(h.tokens) * h.hidden);
    return artifact;
}

__global__ void merge_kernel(
        const __half * base,
        const __half * coeff,
        const std::uint16_t * indices,
        const __half * basis,
        const __half * mean,
        float * output,
        int tokens,
        int hidden,
        int rank,
        int keep) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden;
    if (flat >= total) {
        return;
    }
    const int token = flat / hidden;
    const int feature = flat - token * hidden;
    float value = __half2float(base[flat]) + __half2float(mean[feature]);
    for (int slot = 0; slot < keep; ++slot) {
        const int route = indices[token * keep + slot];
        if (route < rank) {
            value += __half2float(coeff[token * keep + slot])
                * __half2float(basis[static_cast<std::size_t>(route) * hidden + feature]);
        }
    }
    output[flat] = value;
}

struct Metrics {
    double mean_rel_l2 = 0.0;
    double p95_rel_l2 = 0.0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
};

Metrics compare(const std::vector<float> & output, const std::vector<float> & target, int tokens, int hidden) {
    std::vector<double> rel(tokens);
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (int token = 0; token < tokens; ++token) {
        double error_sq = 0.0;
        double target_sq = 0.0;
        for (int feature = 0; feature < hidden; ++feature) {
            const std::size_t index = static_cast<std::size_t>(token) * hidden + feature;
            const double error = static_cast<double>(output[index]) - target[index];
            error_sq += error * error;
            target_sq += static_cast<double>(target[index]) * target[index];
            const double abs_error = std::abs(error);
            max_abs = std::max(max_abs, abs_error);
            sum_abs += abs_error;
        }
        rel[token] = std::sqrt(error_sq) / std::max(std::sqrt(target_sq), 1.0e-12);
    }
    std::vector<double> sorted = rel;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t p95_index = static_cast<std::size_t>(std::floor(0.95 * (sorted.size() - 1)));
    return {
        std::accumulate(rel.begin(), rel.end(), 0.0) / rel.size(),
        sorted[p95_index],
        max_abs,
        sum_abs / output.size(),
    };
}

double event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float value = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&value, start, stop));
    return value;
}

struct Benchmark {
    int tokens = 0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double pipeline_ms = 0.0;
    std::size_t payload_bytes = 0;
};

__global__ void merge_packed_kernel(
        const std::uint8_t * packet,
        const __half * basis,
        const __half * mean,
        float * output,
        int tokens,
        int hidden,
        int rank,
        int keep,
        std::size_t base_bytes,
        std::size_t coeff_bytes) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden;
    if (flat >= total) {
        return;
    }
    const auto * base = reinterpret_cast<const __half *>(packet);
    const auto * coeff = reinterpret_cast<const __half *>(packet + base_bytes);
    const auto * indices = reinterpret_cast<const std::uint16_t *>(packet + base_bytes + coeff_bytes);
    const int token = flat / hidden;
    const int feature = flat - token * hidden;
    float value = __half2float(base[flat]) + __half2float(mean[feature]);
    for (int slot = 0; slot < keep; ++slot) {
        const int route = indices[token * keep + slot];
        if (route < rank) {
            value += __half2float(coeff[token * keep + slot])
                * __half2float(basis[static_cast<std::size_t>(route) * hidden + feature]);
        }
    }
    output[flat] = value;
}

__global__ void merge_packed_half2_kernel(
        const std::uint8_t * packet,
        const __half * basis,
        const __half * mean,
        float * output,
        int tokens,
        int hidden,
        int rank,
        int keep,
        std::size_t base_bytes,
        std::size_t coeff_bytes) {
    const int pairs_per_token = hidden / 2;
    const int pair_flat = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = tokens * pairs_per_token;
    if (pair_flat >= total_pairs) {
        return;
    }
    const auto * base = reinterpret_cast<const __half *>(packet);
    const auto * coeff = reinterpret_cast<const __half *>(packet + base_bytes);
    const auto * indices = reinterpret_cast<const std::uint16_t *>(packet + base_bytes + coeff_bytes);
    const int token = pair_flat / pairs_per_token;
    const int pair_in_token = pair_flat - token * pairs_per_token;
    const int feature = pair_in_token * 2;
    const auto base_pair = reinterpret_cast<const __half2 *>(base)[pair_flat];
    const auto mean_pair = reinterpret_cast<const __half2 *>(mean)[feature / 2];
    float2 value = __half22float2(base_pair);
    const float2 mean_value = __half22float2(mean_pair);
    value.x += mean_value.x;
    value.y += mean_value.y;
    for (int slot = 0; slot < keep; ++slot) {
        const int route = indices[token * keep + slot];
        if (route < rank) {
            const float scale = __half2float(coeff[token * keep + slot]);
            const auto basis_pair = reinterpret_cast<const __half2 *>(
                basis + static_cast<std::size_t>(route) * hidden + feature)[0];
            const float2 basis_value = __half22float2(basis_pair);
            value.x += scale * basis_value.x;
            value.y += scale * basis_value.y;
        }
    }
    const int output_index = token * hidden + feature;
    output[output_index] = value.x;
    output[output_index + 1] = value.y;
}

Benchmark benchmark_packed(
        const Artifact & artifact,
        int tokens,
        __half * device_basis,
        __half * device_mean,
        int iterations,
        int warmup,
        std::vector<float> * validation_output = nullptr,
        bool vectorized = false) {
    const auto & h = artifact.header;
    const std::size_t base_count = static_cast<std::size_t>(tokens) * h.hidden;
    const std::size_t coeff_count = static_cast<std::size_t>(tokens) * h.keep;
    const std::size_t base_bytes = base_count * sizeof(__half);
    const std::size_t coeff_bytes = coeff_count * sizeof(__half);
    const std::size_t index_bytes = coeff_count * sizeof(std::uint16_t);
    const std::size_t payload_bytes = base_bytes + coeff_bytes + index_bytes;

    std::uint8_t * host_packet = nullptr;
    std::uint8_t * device_packet = nullptr;
    float * device_output = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&host_packet), payload_bytes));
    std::memcpy(host_packet, artifact.base.data(), base_bytes);
    std::memcpy(host_packet + base_bytes, artifact.coeff.data(), coeff_bytes);
    std::memcpy(host_packet + base_bytes + coeff_bytes, artifact.indices.data(), index_bytes);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_packet), payload_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_output), base_count * sizeof(float)));

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t copy_stop = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&copy_stop));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int work_items = vectorized
        ? tokens * (static_cast<int>(h.hidden) / 2)
        : tokens * static_cast<int>(h.hidden);
    const int blocks = (work_items + 255) / 256;
    auto enqueue = [&]() {
        CUDA_CHECK(cudaMemcpyAsync(device_packet, host_packet, payload_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(copy_stop, stream));
        if (vectorized) {
            merge_packed_half2_kernel<<<blocks, 256, 0, stream>>>(device_packet, device_basis, device_mean,
                device_output, tokens, h.hidden, h.rank, h.keep, base_bytes, coeff_bytes);
        } else {
            merge_packed_kernel<<<blocks, 256, 0, stream>>>(device_packet, device_basis, device_mean,
                device_output, tokens, h.hidden, h.rank, h.keep, base_bytes, coeff_bytes);
        }
        CUDA_CHECK(cudaGetLastError());
    };
    for (int i = 0; i < warmup; ++i) {
        enqueue();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (validation_output != nullptr) {
        validation_output->resize(base_count);
        CUDA_CHECK(cudaMemcpy(validation_output->data(), device_output,
            base_count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double pipeline_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        enqueue();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        const double copy = event_ms(start, copy_stop);
        const double pipeline = event_ms(start, stop);
        h2d_ms += copy;
        pipeline_ms += pipeline;
        kernel_ms += pipeline - copy;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(copy_stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(host_packet));
    CUDA_CHECK(cudaFree(device_packet));
    CUDA_CHECK(cudaFree(device_output));
    return {tokens, h2d_ms / iterations, kernel_ms / iterations, pipeline_ms / iterations, payload_bytes};
}

Benchmark benchmark(
        const Artifact & artifact,
        int tokens,
        __half * device_basis,
        __half * device_mean,
        int iterations,
        int warmup) {
    const auto & h = artifact.header;
    const std::size_t base_count = static_cast<std::size_t>(tokens) * h.hidden;
    const std::size_t coeff_count = static_cast<std::size_t>(tokens) * h.keep;
    const std::size_t base_bytes = base_count * sizeof(__half);
    const std::size_t coeff_bytes = coeff_count * sizeof(__half);
    const std::size_t index_bytes = coeff_count * sizeof(std::uint16_t);
    const std::size_t payload_bytes = base_bytes + coeff_bytes + index_bytes;

    __half * host_base = nullptr;
    __half * host_coeff = nullptr;
    std::uint16_t * host_indices = nullptr;
    __half * device_base = nullptr;
    __half * device_coeff = nullptr;
    std::uint16_t * device_indices = nullptr;
    float * device_output = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&host_base), base_bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&host_coeff), coeff_bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&host_indices), index_bytes));
    std::copy_n(artifact.base.begin(), base_count, host_base);
    std::copy_n(artifact.coeff.begin(), coeff_count, host_coeff);
    std::copy_n(artifact.indices.begin(), coeff_count, host_indices);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_base), base_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_coeff), coeff_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_indices), index_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_output), base_count * sizeof(float)));

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t copy_stop = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&copy_stop));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int blocks = (tokens * static_cast<int>(h.hidden) + 255) / 256;
    auto enqueue = [&]() {
        CUDA_CHECK(cudaMemcpyAsync(device_base, host_base, base_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_coeff, host_coeff, coeff_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_indices, host_indices, index_bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(copy_stop, stream));
        merge_kernel<<<blocks, 256, 0, stream>>>(device_base, device_coeff, device_indices,
            device_basis, device_mean, device_output, tokens, h.hidden, h.rank, h.keep);
        CUDA_CHECK(cudaGetLastError());
    };
    for (int i = 0; i < warmup; ++i) {
        enqueue();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double pipeline_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        enqueue();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        const double copy = event_ms(start, copy_stop);
        const double pipeline = event_ms(start, stop);
        h2d_ms += copy;
        pipeline_ms += pipeline;
        kernel_ms += pipeline - copy;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(copy_stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(host_base));
    CUDA_CHECK(cudaFreeHost(host_coeff));
    CUDA_CHECK(cudaFreeHost(host_indices));
    CUDA_CHECK(cudaFree(device_base));
    CUDA_CHECK(cudaFree(device_coeff));
    CUDA_CHECK(cudaFree(device_indices));
    CUDA_CHECK(cudaFree(device_output));
    return {tokens, h2d_ms / iterations, kernel_ms / iterations, pipeline_ms / iterations, payload_bytes};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: ffn_artifact_probe ARTIFACT OUTPUT_JSON\n";
            return 2;
        }
        const Artifact artifact = read_artifact(argv[1]);
        const auto & h = artifact.header;
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

        __half * device_basis = nullptr;
        __half * device_mean = nullptr;
        __half * device_base = nullptr;
        __half * device_coeff = nullptr;
        std::uint16_t * device_indices = nullptr;
        float * device_output = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_basis), artifact.basis.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_mean), artifact.mean.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_base), artifact.base.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_coeff), artifact.coeff.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_indices), artifact.indices.size() * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_output), artifact.reference.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(device_basis, artifact.basis.data(), artifact.basis.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_mean, artifact.mean.data(), artifact.mean.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_base, artifact.base.data(), artifact.base.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_coeff, artifact.coeff.data(), artifact.coeff.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_indices, artifact.indices.data(), artifact.indices.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
        const int blocks = (static_cast<int>(h.tokens * h.hidden) + 255) / 256;
        merge_kernel<<<blocks, 256>>>(device_base, device_coeff, device_indices, device_basis, device_mean,
            device_output, h.tokens, h.hidden, h.rank, h.keep);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> output(artifact.reference.size());
        CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const Metrics implementation = compare(output, artifact.reference, h.tokens, h.hidden);
        const Metrics approximation = compare(output, artifact.target, h.tokens, h.hidden);

        const std::vector<int> requested = {1, 16, 64};
        std::vector<Benchmark> benchmarks;
        std::vector<Benchmark> packed_benchmarks;
        std::vector<Benchmark> packed_half2_benchmarks;
        std::vector<float> packed_output_64;
        std::vector<float> packed_half2_output_64;
        if ((h.hidden % 2) != 0) {
            throw std::runtime_error("half2 benchmark requires an even hidden size");
        }
        for (int tokens : requested) {
            if (tokens <= static_cast<int>(h.tokens)) {
                benchmarks.push_back(benchmark(artifact, tokens, device_basis, device_mean, 200, 20));
                packed_benchmarks.push_back(benchmark_packed(artifact, tokens, device_basis, device_mean, 200, 20,
                    tokens == 64 ? &packed_output_64 : nullptr));
                packed_half2_benchmarks.push_back(benchmark_packed(artifact, tokens, device_basis, device_mean, 200, 20,
                    tokens == 64 ? &packed_half2_output_64 : nullptr, true));
            }
        }
        std::vector<float> reference_64(artifact.reference.begin(), artifact.reference.begin() + packed_output_64.size());
        const Metrics packed_implementation = compare(packed_output_64, reference_64, 64, h.hidden);
        const Metrics packed_half2_implementation = compare(packed_half2_output_64, reference_64, 64, h.hidden);

        CUDA_CHECK(cudaFree(device_basis));
        CUDA_CHECK(cudaFree(device_mean));
        CUDA_CHECK(cudaFree(device_base));
        CUDA_CHECK(cudaFree(device_coeff));
        CUDA_CHECK(cudaFree(device_indices));
        CUDA_CHECK(cudaFree(device_output));

        std::ofstream json(argv[2], std::ios::binary);
        if (!json) {
            throw std::runtime_error("cannot open output JSON");
        }
        json << std::setprecision(9);
        json << "{\n";
        json << "  \"experiment\": \"real_ffn_artifact_cuda_validation\",\n";
        json << "  \"device\": \"" << properties.name << "\",\n";
        json << "  \"layer\": " << h.layer << ",\n";
        json << "  \"tokens\": " << h.tokens << ",\n";
        json << "  \"hidden\": " << h.hidden << ",\n";
        json << "  \"rank\": " << h.rank << ",\n";
        json << "  \"keep\": " << h.keep << ",\n";
        json << "  \"resident_fp16_bytes\": " << (artifact.basis.size() + artifact.mean.size()) * sizeof(__half) << ",\n";
        json << "  \"implementation_vs_fp16_reference\": {\"mean_rel_l2\": " << implementation.mean_rel_l2
             << ", \"p95_rel_l2\": " << implementation.p95_rel_l2
             << ", \"max_abs_error\": " << implementation.max_abs
             << ", \"mean_abs_error\": " << implementation.mean_abs << "},\n";
        json << "  \"packed_implementation_vs_fp16_reference\": {\"mean_rel_l2\": " << packed_implementation.mean_rel_l2
             << ", \"p95_rel_l2\": " << packed_implementation.p95_rel_l2
             << ", \"max_abs_error\": " << packed_implementation.max_abs
             << ", \"mean_abs_error\": " << packed_implementation.mean_abs << "},\n";
        json << "  \"packed_half2_implementation_vs_fp16_reference\": {\"mean_rel_l2\": " << packed_half2_implementation.mean_rel_l2
             << ", \"p95_rel_l2\": " << packed_half2_implementation.p95_rel_l2
             << ", \"max_abs_error\": " << packed_half2_implementation.max_abs
             << ", \"mean_abs_error\": " << packed_half2_implementation.mean_abs << "},\n";
        json << "  \"approximation_vs_capture\": {\"mean_rel_l2\": " << approximation.mean_rel_l2
             << ", \"p95_rel_l2\": " << approximation.p95_rel_l2
             << ", \"max_abs_error\": " << approximation.max_abs
             << ", \"mean_abs_error\": " << approximation.mean_abs << "},\n";
        json << "  \"benchmarks\": [\n";
        for (std::size_t i = 0; i < benchmarks.size(); ++i) {
            const auto & row = benchmarks[i];
            json << "    {\"tokens\": " << row.tokens
                 << ", \"payload_bytes\": " << row.payload_bytes
                 << ", \"payload_bytes_per_token\": " << row.payload_bytes / row.tokens
                 << ", \"h2d_ms\": " << row.h2d_ms
                 << ", \"kernel_ms\": " << row.kernel_ms
                 << ", \"pipeline_ms\": " << row.pipeline_ms
                 << ", \"pipeline_ms_per_token\": " << row.pipeline_ms / row.tokens << "}";
            if (i + 1 != benchmarks.size()) json << ',';
            json << '\n';
        }
        json << "  ],\n";
        json << "  \"packed_benchmarks\": [\n";
        for (std::size_t i = 0; i < packed_benchmarks.size(); ++i) {
            const auto & row = packed_benchmarks[i];
            json << "    {\"tokens\": " << row.tokens
                 << ", \"payload_bytes\": " << row.payload_bytes
                 << ", \"payload_bytes_per_token\": " << row.payload_bytes / row.tokens
                 << ", \"h2d_ms\": " << row.h2d_ms
                 << ", \"kernel_ms\": " << row.kernel_ms
                 << ", \"pipeline_ms\": " << row.pipeline_ms
                 << ", \"pipeline_ms_per_token\": " << row.pipeline_ms / row.tokens << "}";
            if (i + 1 != packed_benchmarks.size()) json << ',';
            json << '\n';
        }
        json << "  ],\n";
        json << "  \"packed_half2_benchmarks\": [\n";
        for (std::size_t i = 0; i < packed_half2_benchmarks.size(); ++i) {
            const auto & row = packed_half2_benchmarks[i];
            json << "    {\"tokens\": " << row.tokens
                 << ", \"payload_bytes\": " << row.payload_bytes
                 << ", \"payload_bytes_per_token\": " << row.payload_bytes / row.tokens
                 << ", \"h2d_ms\": " << row.h2d_ms
                 << ", \"kernel_ms\": " << row.kernel_ms
                 << ", \"pipeline_ms\": " << row.pipeline_ms
                 << ", \"pipeline_ms_per_token\": " << row.pipeline_ms / row.tokens << "}";
            if (i + 1 != packed_half2_benchmarks.size()) json << ',';
            json << '\n';
        }
        json << "  ]\n";
        json << "}\n";
        std::cout << "wrote " << argv[2] << "\n";
        std::cout << "implementation mean_rel_l2=" << implementation.mean_rel_l2
                  << " approximation mean_rel_l2=" << approximation.mean_rel_l2 << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ffn_artifact_probe: " << error.what() << '\n';
        return 1;
    }
}
