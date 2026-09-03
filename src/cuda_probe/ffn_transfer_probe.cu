#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kHidden = 2048;
constexpr int kTopK = 4;
constexpr int kResidualRank = 64;
constexpr int kRingSlots = 4;
constexpr std::size_t kFallbackWeightBytes = 10321920;

enum class Mode {
    BaseOnly,
    BasePlusResidual,
    BasePlusResidualOverlap,
    ExactFallbackReuse,
    ExactFallbackEvicted,
};

const char * mode_name(Mode mode) {
    switch (mode) {
        case Mode::BaseOnly: return "base_only";
        case Mode::BasePlusResidual: return "base_plus_residual";
        case Mode::BasePlusResidualOverlap: return "base_plus_residual_overlap";
        case Mode::ExactFallbackReuse: return "exact_fallback_reuse";
        case Mode::ExactFallbackEvicted: return "exact_fallback_evicted";
    }
    return "unknown";
}

void check_cuda(cudaError_t status, const char * expr, const char * file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream msg;
        msg << expr << " failed at " << file << ':' << line << ": " << cudaGetErrorString(status);
        throw std::runtime_error(msg.str());
    }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

std::size_t round_up(std::size_t value, std::size_t block) {
    return ((value + block - 1) / block) * block;
}

std::size_t base_bytes(int tokens) {
    return static_cast<std::size_t>(tokens) * kHidden * sizeof(__half);
}

std::size_t residual_bytes(int tokens) {
    return static_cast<std::size_t>(tokens) * kTopK * (sizeof(__half) + sizeof(std::uint16_t));
}

std::size_t approx_payload_bytes(int tokens) {
    return base_bytes(tokens) + residual_bytes(tokens);
}

__global__ void copy_base_kernel(const __half * packet, float * output, int tokens, int hidden) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden;
    if (index >= total) {
        return;
    }
    output[index] = __half2float(packet[index]);
}

__global__ void residual_merge_kernel(
        const std::uint8_t * packet,
        const __half * basis,
        const __half * residual_mean,
        float * output,
        int tokens,
        int hidden,
        int top_k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden;
    if (index >= total) {
        return;
    }
    const int token = index / hidden;
    const int feature = index - token * hidden;
    const std::size_t base_offset = static_cast<std::size_t>(token) * hidden * sizeof(__half);
    const std::size_t coeff_offset = static_cast<std::size_t>(tokens) * hidden * sizeof(__half)
        + static_cast<std::size_t>(token) * top_k * sizeof(__half);
    const std::size_t index_offset = static_cast<std::size_t>(tokens) * hidden * sizeof(__half)
        + static_cast<std::size_t>(tokens) * top_k * sizeof(__half)
        + static_cast<std::size_t>(token) * top_k * sizeof(std::uint16_t);

    const auto * base = reinterpret_cast<const __half *>(packet + base_offset);
    const auto * coeff = reinterpret_cast<const __half *>(packet + coeff_offset);
    const auto * ids = reinterpret_cast<const std::uint16_t *>(packet + index_offset);
    float value = __half2float(base[feature]) + __half2float(residual_mean[feature]);
    for (int i = 0; i < top_k; ++i) {
        const std::uint16_t basis_row = ids[i];
        value += __half2float(coeff[i]) * __half2float(basis[static_cast<std::size_t>(basis_row) * hidden + feature]);
    }
    output[index] = value;
}

__global__ void fallback_checksum_kernel(const std::uint8_t * weights, float * output, int token) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const auto * values = reinterpret_cast<const float *>(weights);
        const std::size_t count = kFallbackWeightBytes / sizeof(float);
        output[token] = values[0] + values[count / 2] + values[count - 1];
    }
}

struct Timings {
    double h2d_ms = 0.0;
    double pipeline_ms = 0.0;
    double kernel_ms = 0.0;
    std::size_t transferred_bytes = 0;
};

struct ProbeBuffers {
    std::uint8_t * host_ring = nullptr;
    std::uint8_t * host_fallback = nullptr;
    std::uint8_t * device_packet = nullptr;
    std::uint8_t * device_fallback = nullptr;
    __half * device_basis = nullptr;
    __half * device_residual_mean = nullptr;
    float * device_output = nullptr;
    std::size_t ring_slot_bytes = 0;
    std::size_t fallback_transfer_bytes = 0;
};

void free_buffers(ProbeBuffers & buffers) {
    if (buffers.host_ring) cudaFreeHost(buffers.host_ring);
    if (buffers.host_fallback) cudaFreeHost(buffers.host_fallback);
    if (buffers.device_packet) cudaFree(buffers.device_packet);
    if (buffers.device_fallback) cudaFree(buffers.device_fallback);
    if (buffers.device_basis) cudaFree(buffers.device_basis);
    if (buffers.device_residual_mean) cudaFree(buffers.device_residual_mean);
    if (buffers.device_output) cudaFree(buffers.device_output);
    buffers = {};
}

ProbeBuffers make_buffers(int tokens, std::size_t block_bytes) {
    ProbeBuffers buffers;
    const std::size_t approx_transfer = round_up(approx_payload_bytes(tokens), block_bytes);
    const std::size_t base_transfer = round_up(base_bytes(tokens), block_bytes);
    buffers.ring_slot_bytes = std::max(approx_transfer, base_transfer);
    buffers.fallback_transfer_bytes = round_up(kFallbackWeightBytes, block_bytes);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&buffers.host_ring), buffers.ring_slot_bytes * kRingSlots));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&buffers.host_fallback), buffers.fallback_transfer_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.device_packet), buffers.ring_slot_bytes * kRingSlots));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.device_fallback), buffers.fallback_transfer_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.device_basis),
        static_cast<std::size_t>(kResidualRank) * kHidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.device_residual_mean), kHidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.device_output),
        static_cast<std::size_t>(tokens) * kHidden * sizeof(float)));
    return buffers;
}

void fill_inputs(ProbeBuffers & buffers, int tokens, std::mt19937 & rng) {
    std::uniform_real_distribution<float> base_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> coeff_dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> basis_dist(-0.05f, 0.05f);
    std::fill(buffers.host_ring, buffers.host_ring + buffers.ring_slot_bytes * kRingSlots, 0);
    std::fill(buffers.host_fallback, buffers.host_fallback + buffers.fallback_transfer_bytes, 0);

    auto * first_packet = buffers.host_ring;
    auto * base = reinterpret_cast<__half *>(first_packet);
    for (int i = 0; i < tokens * kHidden; ++i) {
        base[i] = __float2half(base_dist(rng));
    }
    auto * coeff = reinterpret_cast<__half *>(first_packet + base_bytes(tokens));
    for (int i = 0; i < tokens * kTopK; ++i) {
        coeff[i] = __float2half(coeff_dist(rng));
    }
    auto * ids = reinterpret_cast<std::uint16_t *>(first_packet + base_bytes(tokens) + tokens * kTopK * sizeof(__half));
    for (int i = 0; i < tokens * kTopK; ++i) {
        ids[i] = static_cast<std::uint16_t>((i * 13 + 7) % kResidualRank);
    }
    for (int slot = 1; slot < kRingSlots; ++slot) {
        std::copy(first_packet, first_packet + buffers.ring_slot_bytes,
            buffers.host_ring + static_cast<std::size_t>(slot) * buffers.ring_slot_bytes);
    }
    for (std::size_t i = 0; i < kFallbackWeightBytes / sizeof(float); ++i) {
        const int pattern = static_cast<int>(i % 17) - 8;
        reinterpret_cast<float *>(buffers.host_fallback)[i] = 0.001f * static_cast<float>(pattern);
    }
}

void upload_basis(ProbeBuffers & buffers, std::mt19937 & rng) {
    std::uniform_real_distribution<float> basis_dist(-0.05f, 0.05f);
    std::vector<__half> basis(static_cast<std::size_t>(kResidualRank) * kHidden);
    std::vector<__half> mean(kHidden);
    for (auto & value : basis) {
        value = __float2half(basis_dist(rng));
    }
    for (auto & value : mean) {
        value = __float2half(basis_dist(rng));
    }
    CUDA_CHECK(cudaMemcpy(buffers.device_basis, basis.data(), basis.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.device_residual_mean, mean.data(), mean.size() * sizeof(__half), cudaMemcpyHostToDevice));
}

void enqueue_work(Mode mode, ProbeBuffers & buffers, int tokens, std::size_t transfer_bytes, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (tokens * kHidden + threads - 1) / threads;
    if (mode == Mode::BaseOnly) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.device_packet, buffers.host_ring, transfer_bytes,
            cudaMemcpyHostToDevice, stream));
        copy_base_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half *>(buffers.device_packet), buffers.device_output, tokens, kHidden);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (mode == Mode::BasePlusResidual || mode == Mode::BasePlusResidualOverlap) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.device_packet, buffers.host_ring, transfer_bytes,
            cudaMemcpyHostToDevice, stream));
        residual_merge_kernel<<<blocks, threads, 0, stream>>>(
            buffers.device_packet, buffers.device_basis, buffers.device_residual_mean,
            buffers.device_output, tokens, kHidden, kTopK);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (mode == Mode::ExactFallbackReuse) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.device_fallback, buffers.host_fallback, transfer_bytes,
            cudaMemcpyHostToDevice, stream));
        for (int token = 0; token < tokens; ++token) {
            fallback_checksum_kernel<<<1, 1, 0, stream>>>(buffers.device_fallback, buffers.device_output, token);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    for (int token = 0; token < tokens; ++token) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.device_fallback, buffers.host_fallback, transfer_bytes,
            cudaMemcpyHostToDevice, stream));
        fallback_checksum_kernel<<<1, 1, 0, stream>>>(buffers.device_fallback, buffers.device_output, token);
        CUDA_CHECK(cudaGetLastError());
    }
}

void enqueue_copies_only(Mode mode, ProbeBuffers & buffers, int tokens, std::size_t transfer_bytes, cudaStream_t stream) {
    if (mode == Mode::ExactFallbackEvicted) {
        for (int token = 0; token < tokens; ++token) {
            CUDA_CHECK(cudaMemcpyAsync(buffers.device_fallback, buffers.host_fallback, transfer_bytes,
                cudaMemcpyHostToDevice, stream));
        }
    } else if (mode == Mode::ExactFallbackReuse) {
        CUDA_CHECK(cudaMemcpyAsync(buffers.device_fallback, buffers.host_fallback, transfer_bytes,
            cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(buffers.device_packet, buffers.host_ring, transfer_bytes,
            cudaMemcpyHostToDevice, stream));
    }
}

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float value = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&value, start, stop));
    return static_cast<double>(value);
}

Timings benchmark_overlap(ProbeBuffers & buffers, int tokens, std::size_t block_bytes, int iterations, int warmup) {
    const int chunk_tokens = std::min(16, tokens);
    const int chunks = (tokens + chunk_tokens - 1) / chunk_tokens;
    const std::size_t chunk_payload = approx_payload_bytes(chunk_tokens);
    const std::size_t chunk_transfer = round_up(chunk_payload, block_bytes);
    const std::size_t total_transfer = chunk_transfer * static_cast<std::size_t>(chunks);
    cudaStream_t copy_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t copy_start = nullptr;
    cudaEvent_t copy_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    std::vector<cudaEvent_t> ready_events(chunks);
    std::vector<cudaEvent_t> done_events(chunks);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&copy_start));
    CUDA_CHECK(cudaEventCreate(&copy_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    for (int chunk = 0; chunk < chunks; ++chunk) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ready_events[chunk], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&done_events[chunk], cudaEventDisableTiming));
    }

    auto enqueue_chunks = [&](bool timed) {
        if (timed) {
            CUDA_CHECK(cudaEventRecord(start, copy_stream));
            CUDA_CHECK(cudaEventRecord(copy_start, copy_stream));
            CUDA_CHECK(cudaEventRecord(kernel_start, compute_stream));
        }
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int token_offset = chunk * chunk_tokens;
            const int local_tokens = std::min(chunk_tokens, tokens - token_offset);
            const std::size_t base_offset = static_cast<std::size_t>(token_offset) * kHidden * sizeof(__half);
            const std::size_t coeff_offset = base_bytes(tokens)
                + static_cast<std::size_t>(token_offset) * kTopK * sizeof(__half);
            const std::size_t index_offset = base_bytes(tokens)
                + static_cast<std::size_t>(tokens) * kTopK * sizeof(__half)
                + static_cast<std::size_t>(token_offset) * kTopK * sizeof(std::uint16_t);
            const int slot = chunk % kRingSlots;
            auto * device_slot = buffers.device_packet + static_cast<std::size_t>(slot) * buffers.ring_slot_bytes;
            if (chunk >= kRingSlots) {
                CUDA_CHECK(cudaStreamWaitEvent(copy_stream, done_events[chunk - kRingSlots], 0));
            }
            CUDA_CHECK(cudaMemcpyAsync(device_slot, buffers.host_ring + base_offset,
                static_cast<std::size_t>(local_tokens) * kHidden * sizeof(__half), cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_slot + static_cast<std::size_t>(local_tokens) * kHidden * sizeof(__half),
                buffers.host_ring + coeff_offset, static_cast<std::size_t>(local_tokens) * kTopK * sizeof(__half),
                cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaMemcpyAsync(device_slot + static_cast<std::size_t>(local_tokens) * kHidden * sizeof(__half)
                    + static_cast<std::size_t>(local_tokens) * kTopK * sizeof(__half),
                buffers.host_ring + index_offset, static_cast<std::size_t>(local_tokens) * kTopK * sizeof(std::uint16_t),
                cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaEventRecord(ready_events[chunk], copy_stream));
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, ready_events[chunk], 0));
            const int blocks = (local_tokens * kHidden + 255) / 256;
            residual_merge_kernel<<<blocks, 256, 0, compute_stream>>>(
                device_slot, buffers.device_basis, buffers.device_residual_mean,
                buffers.device_output + static_cast<std::size_t>(token_offset) * kHidden,
                local_tokens, kHidden, kTopK);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(done_events[chunk], compute_stream));
        }
        if (timed) {
            CUDA_CHECK(cudaEventRecord(copy_stop, copy_stream));
            CUDA_CHECK(cudaEventRecord(kernel_stop, compute_stream));
            CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        }
    };

    for (int i = 0; i < warmup; ++i) {
        enqueue_chunks(false);
    }
    CUDA_CHECK(cudaStreamSynchronize(copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    double total_h2d = 0.0;
    double total_pipeline = 0.0;
    double total_kernel = 0.0;
    for (int i = 0; i < iterations; ++i) {
        enqueue_chunks(true);
        CUDA_CHECK(cudaEventSynchronize(stop));
        total_h2d += elapsed_ms(copy_start, copy_stop);
        total_kernel += elapsed_ms(kernel_start, kernel_stop);
        total_pipeline += elapsed_ms(start, stop);
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
        CUDA_CHECK(cudaEventDestroy(ready_events[chunk]));
        CUDA_CHECK(cudaEventDestroy(done_events[chunk]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(copy_start));
    CUDA_CHECK(cudaEventDestroy(copy_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    return {total_h2d / iterations, total_pipeline / iterations, total_kernel / iterations, total_transfer};
}

Timings benchmark(Mode mode, ProbeBuffers & buffers, int tokens, std::size_t block_bytes, int iterations, int warmup) {
    const std::size_t payload = mode == Mode::BaseOnly ? base_bytes(tokens)
        : (mode == Mode::BasePlusResidual || mode == Mode::BasePlusResidualOverlap) ? approx_payload_bytes(tokens)
        : mode == Mode::ExactFallbackReuse ? kFallbackWeightBytes
        : kFallbackWeightBytes * static_cast<std::size_t>(tokens);
    const std::size_t transfer = mode == Mode::ExactFallbackEvicted
        ? buffers.fallback_transfer_bytes * static_cast<std::size_t>(tokens)
        : mode == Mode::ExactFallbackReuse ? buffers.fallback_transfer_bytes
        : round_up(payload, block_bytes);
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t copy_start = nullptr;
    cudaEvent_t copy_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&copy_start));
    CUDA_CHECK(cudaEventCreate(&copy_stop));
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    for (int i = 0; i < warmup; ++i) {
        enqueue_work(mode, buffers, tokens,
            mode == Mode::ExactFallbackEvicted ? buffers.fallback_transfer_bytes : mode == Mode::ExactFallbackReuse
                ? buffers.fallback_transfer_bytes : round_up(
                    mode == Mode::BaseOnly ? base_bytes(tokens) : approx_payload_bytes(tokens), block_bytes), stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double total_h2d = 0.0;
    double total_pipeline = 0.0;
    double total_kernel = 0.0;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaEventRecord(copy_start, stream));
        enqueue_copies_only(mode, buffers, tokens,
            mode == Mode::ExactFallbackEvicted ? buffers.fallback_transfer_bytes : mode == Mode::ExactFallbackReuse
                ? buffers.fallback_transfer_bytes : round_up(
                    mode == Mode::BaseOnly ? base_bytes(tokens) : approx_payload_bytes(tokens), block_bytes), stream);
        CUDA_CHECK(cudaEventRecord(copy_stop, stream));
        CUDA_CHECK(cudaEventRecord(kernel_start, stream));
        if (mode == Mode::BaseOnly) {
            const int blocks = (tokens * kHidden + 255) / 256;
            copy_base_kernel<<<blocks, 256, 0, stream>>>(
                reinterpret_cast<const __half *>(buffers.device_packet), buffers.device_output, tokens, kHidden);
        } else if (mode == Mode::BasePlusResidual || mode == Mode::BasePlusResidualOverlap) {
            const int blocks = (tokens * kHidden + 255) / 256;
            residual_merge_kernel<<<blocks, 256, 0, stream>>>(
                buffers.device_packet, buffers.device_basis, buffers.device_residual_mean,
                buffers.device_output, tokens, kHidden, kTopK);
        } else {
            for (int token = 0; token < tokens; ++token) {
                fallback_checksum_kernel<<<1, 1, 0, stream>>>(buffers.device_fallback, buffers.device_output, token);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kernel_stop, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        total_h2d += elapsed_ms(copy_start, copy_stop);
        total_kernel += elapsed_ms(kernel_start, kernel_stop);
        total_pipeline += elapsed_ms(start, stop);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(copy_start));
    CUDA_CHECK(cudaEventDestroy(copy_stop));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    Timings result;
    result.h2d_ms = total_h2d / iterations;
    result.kernel_ms = total_kernel / iterations;
    result.pipeline_ms = total_pipeline / iterations;
    result.transferred_bytes = transfer;
    return result;
}

struct Verification {
    double max_abs = 0.0;
    double mean_abs = 0.0;
};

Verification verify_approx(ProbeBuffers & buffers, int tokens) {
    const std::size_t payload = approx_payload_bytes(tokens);
    CUDA_CHECK(cudaMemcpy(buffers.device_packet, buffers.host_ring, payload, cudaMemcpyHostToDevice));
    const int blocks = (tokens * kHidden + 255) / 256;
    residual_merge_kernel<<<blocks, 256>>>(buffers.device_packet, buffers.device_basis,
        buffers.device_residual_mean, buffers.device_output, tokens, kHidden, kTopK);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> output(static_cast<std::size_t>(tokens) * kHidden);
    CUDA_CHECK(cudaMemcpy(output.data(), buffers.device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const auto * packet = buffers.host_ring;
    const auto * base = reinterpret_cast<const __half *>(packet);
    const auto * coeff = reinterpret_cast<const __half *>(packet + base_bytes(tokens));
    const auto * ids = reinterpret_cast<const std::uint16_t *>(packet + base_bytes(tokens) + tokens * kTopK * sizeof(__half));
    std::vector<__half> basis(static_cast<std::size_t>(kResidualRank) * kHidden);
    std::vector<__half> mean(kHidden);
    CUDA_CHECK(cudaMemcpy(basis.data(), buffers.device_basis, basis.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mean.data(), buffers.device_residual_mean, mean.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (int token = 0; token < tokens; ++token) {
        for (int feature = 0; feature < kHidden; ++feature) {
            float reference = __half2float(base[token * kHidden + feature]) + __half2float(mean[feature]);
            for (int i = 0; i < kTopK; ++i) {
                reference += __half2float(coeff[token * kTopK + i])
                    * __half2float(basis[static_cast<std::size_t>(ids[token * kTopK + i]) * kHidden + feature]);
            }
            const double error = std::abs(static_cast<double>(output[token * kHidden + feature] - reference));
            max_abs = std::max(max_abs, error);
            sum_abs += error;
        }
    }
    return {max_abs, sum_abs / static_cast<double>(tokens * kHidden)};
}

std::string json_escape(const std::string & value) {
    std::string output;
    for (char c : value) {
        if (c == '\\' || c == '"') {
            output.push_back('\\');
        }
        output.push_back(c);
    }
    return output;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const std::string output_path = argc > 1 ? argv[1] : "results/cuda_transfer_probe_20260903.json";
        int device = 0;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        int runtime_version = 0;
        int driver_version = 0;
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        CUDA_CHECK(cudaDriverGetVersion(&driver_version));

        std::mt19937 rng(12345);
        const std::vector<std::size_t> blocks = {65536, 262144, 1048576};
        const std::vector<int> windows = {1, 16, 64};
        const std::vector<Mode> modes = {Mode::BaseOnly, Mode::BasePlusResidual, Mode::ExactFallbackReuse,
            Mode::ExactFallbackEvicted};
        const int iterations = 100;
        const int warmup = 10;

        std::ostringstream json;
        json << std::setprecision(9);
        json << "{\n";
        json << "  \"experiment\": \"cuda_ffn_transfer_probe\",\n";
        json << "  \"device\": \"" << json_escape(properties.name) << "\",\n";
        json << "  \"compute_capability\": \"" << properties.major << '.' << properties.minor << "\",\n";
        json << "  \"cuda_runtime_version\": " << runtime_version << ",\n";
        json << "  \"cuda_driver_version\": " << driver_version << ",\n";
        json << "  \"hidden\": " << kHidden << ",\n";
        json << "  \"residual_rank\": " << kResidualRank << ",\n";
        json << "  \"top_k\": " << kTopK << ",\n";
        json << "  \"fallback_weight_bytes\": " << kFallbackWeightBytes << ",\n";
        json << "  \"iterations\": " << iterations << ",\n";
        json << "  \"warmup\": " << warmup << ",\n";
        json << "  \"rows\": [\n";

        bool first = true;
        Verification verification{};
        for (std::size_t block_bytes : blocks) {
            for (int tokens : windows) {
                ProbeBuffers buffers = make_buffers(tokens, block_bytes);
                fill_inputs(buffers, tokens, rng);
                upload_basis(buffers, rng);
                if (block_bytes == blocks.front() && tokens == windows.back()) {
                    verification = verify_approx(buffers, tokens);
                }
                for (Mode mode : modes) {
                    const std::size_t payload = mode == Mode::BaseOnly ? base_bytes(tokens)
                        : mode == Mode::BasePlusResidual ? approx_payload_bytes(tokens)
                        : mode == Mode::ExactFallbackReuse ? kFallbackWeightBytes
                        : kFallbackWeightBytes * static_cast<std::size_t>(tokens);
                    const std::size_t transferred = mode == Mode::ExactFallbackEvicted
                        ? buffers.fallback_transfer_bytes * static_cast<std::size_t>(tokens)
                        : mode == Mode::ExactFallbackReuse ? buffers.fallback_transfer_bytes
                        : round_up(payload, block_bytes);
                    const Timings timings = benchmark(mode, buffers, tokens, block_bytes, iterations, warmup);
                    if (!first) {
                        json << ",\n";
                    }
                    first = false;
                    const double h2d_gbps = timings.h2d_ms > 0.0
                        ? static_cast<double>(transferred) / (timings.h2d_ms * 1.0e6)
                        : 0.0;
                    json << "    {\"mode\": \"" << mode_name(mode)
                         << "\", \"superblock_bytes\": " << block_bytes
                         << ", \"window_tokens\": " << tokens
                         << ", \"payload_bytes\": " << payload
                         << ", \"transfer_bytes_per_window\": " << transferred
                         << ", \"transfer_bytes_per_token\": " << (transferred / static_cast<std::size_t>(tokens))
                         << ", \"h2d_ms_per_window\": " << timings.h2d_ms
                         << ", \"kernel_ms_per_window\": " << timings.kernel_ms
                         << ", \"pipeline_ms_per_window\": " << timings.pipeline_ms
                         << ", \"h2d_gbps\": " << h2d_gbps
                         << ", \"pipeline_ms_per_token\": " << timings.pipeline_ms / tokens
                         << "}";
                }
                const Timings overlap = benchmark_overlap(buffers, tokens, block_bytes, iterations, warmup);
                const std::size_t overlap_payload = approx_payload_bytes(tokens);
                const std::size_t chunk_tokens = std::min(16, tokens);
                const std::size_t overlap_transfer = round_up(approx_payload_bytes(static_cast<int>(chunk_tokens)), block_bytes)
                    * static_cast<std::size_t>((tokens + static_cast<int>(chunk_tokens) - 1) / static_cast<int>(chunk_tokens));
                if (!first) {
                    json << ",\n";
                }
                first = false;
                const double overlap_h2d_gbps = overlap.h2d_ms > 0.0
                    ? static_cast<double>(overlap_transfer) / (overlap.h2d_ms * 1.0e6) : 0.0;
                json << "    {\"mode\": \"base_plus_residual_overlap\", \"superblock_bytes\": " << block_bytes
                     << ", \"window_tokens\": " << tokens
                     << ", \"payload_bytes\": " << overlap_payload
                     << ", \"transfer_bytes_per_window\": " << overlap_transfer
                     << ", \"transfer_bytes_per_token\": " << (overlap_transfer / static_cast<std::size_t>(tokens))
                     << ", \"h2d_ms_per_window\": " << overlap.h2d_ms
                     << ", \"kernel_ms_per_window\": " << overlap.kernel_ms
                     << ", \"pipeline_ms_per_window\": " << overlap.pipeline_ms
                     << ", \"h2d_gbps\": " << overlap_h2d_gbps
                     << ", \"pipeline_ms_per_token\": " << overlap.pipeline_ms / tokens
                     << "}";
                free_buffers(buffers);
            }
        }
        json << "\n  ],\n";
        json << "  \"verification\": {\"tokens\": " << windows.back()
             << ", \"max_abs_error\": " << verification.max_abs
             << ", \"mean_abs_error\": " << verification.mean_abs << "}\n";
        json << "}\n";

        std::ofstream output(output_path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot open output file: " + output_path);
        }
        output << json.str();
        std::cout << "wrote " << output_path << "\n";
        std::cout << "device=" << properties.name << " cc=" << properties.major << '.' << properties.minor << "\n";
        std::cout << "verification max_abs=" << verification.max_abs << " mean_abs=" << verification.mean_abs << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ffn_transfer_probe: " << error.what() << '\n';
        return 1;
    }
}
