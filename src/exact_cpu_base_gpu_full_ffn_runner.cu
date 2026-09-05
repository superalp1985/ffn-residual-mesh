#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << expr << " failed at " << file << ':' << line << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

template <typename T>
std::vector<T> read_vector(const std::string &path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::vector<T> result(count);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(T));
    if (bytes != 0 && !input.read(reinterpret_cast<char *>(result.data()), bytes)) {
        throw std::runtime_error("cannot read " + path);
    }
    return result;
}

struct Shape {
    int rows = 6144;
    int hidden = 2048;
    int groups = 64;
    int tile_rows = 2048;
    int block_size = 2;
    int blocks = 1024;
    int state_count = 256;
    int blocks_per_group = 16;
};

struct CpuTable {
    std::vector<std::uint8_t> table;
    std::vector<std::int16_t> high_sum;
    std::vector<float> alpha;
    std::vector<float> beta;
    std::vector<std::uint16_t> states;
    std::vector<std::int8_t> z;
    std::vector<float> activation_scales;
    std::vector<float> output;
};

CpuTable load_cpu_table(const std::string &dir, const std::string &projection, const Shape &s) {
    CpuTable r;
    r.table = read_vector<std::uint8_t>(dir + "/" + projection + ".table.u8.bin",
        static_cast<std::size_t>(s.blocks) * s.state_count * s.rows);
    r.high_sum = read_vector<std::int16_t>(dir + "/" + projection + ".high_sum.i16.bin",
        static_cast<std::size_t>(s.rows) * s.groups);
    r.alpha = read_vector<float>(dir + "/" + projection + ".alpha.f32.bin",
        static_cast<std::size_t>(s.rows) * s.groups);
    r.beta = read_vector<float>(dir + "/" + projection + ".beta.f32.bin",
        static_cast<std::size_t>(s.rows) * s.groups);
    r.states = read_vector<std::uint16_t>(dir + "/states.u16.bin", static_cast<std::size_t>(4) * s.blocks);
    r.z = read_vector<std::int8_t>(dir + "/activation.z.i8.bin", s.hidden);
    r.activation_scales = read_vector<float>(dir + "/activation.scale.f32.bin", s.groups);
    r.output.resize(s.rows);
    return r;
}

class CpuPool {
public:
    explicit CpuPool(int count) : count_(std::max(1, count)) {
        workers_.reserve(static_cast<std::size_t>(count_));
        for (int id = 0; id < count_; ++id) workers_.emplace_back([this, id] { loop(id); });
    }
    ~CpuPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            condition_.notify_all();
        }
        for (auto &worker : workers_) worker.join();
    }
    void run(const std::function<void(int)> &task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            remaining_ = count_;
            ++generation_;
            condition_.notify_all();
        }
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [this] { return remaining_ == 0; });
    }
private:
    void loop(int id) {
        std::unique_lock<std::mutex> lock(mutex_);
        std::uint64_t local_generation = 0;
        while (!stopping_) {
            condition_.wait(lock, [this, local_generation] { return stopping_ || generation_ != local_generation; });
            if (stopping_) return;
            local_generation = generation_;
            auto task = task_;
            lock.unlock();
            task(id);
            lock.lock();
            if (--remaining_ == 0) finished_.notify_one();
        }
    }
    int count_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;
    std::function<void(int)> task_;
    std::uint64_t generation_ = 0;
    int remaining_ = 0;
    bool stopping_ = false;
};

double cpu_eval(CpuTable &data, const Shape &s, int threads, CpuPool &pool) {
    threads = std::max(1, std::min(threads, s.rows));
    std::vector<std::int32_t> group_dot(static_cast<std::size_t>(s.groups) * s.rows, 0);
    std::int32_t z_sum[64] = {};
    for (int group = 0; group < s.groups; ++group) {
        for (int i = 0; i < 32; ++i) z_sum[group] += data.z[group * 32 + i];
    }
    const auto start = std::chrono::steady_clock::now();
    auto worker = [&](int begin, int end) {
        for (int digit = 0; digit < 4; ++digit) {
            const int radix = 1 << (2 * digit);
            for (int block = 0; block < s.blocks; ++block) {
                const std::uint16_t state = data.states[static_cast<std::size_t>(digit) * s.blocks + block];
                const auto *entry = data.table.data() +
                    (static_cast<std::size_t>(block) * s.state_count + state) * s.rows;
                const int group = block / s.blocks_per_group;
                auto *output = group_dot.data() + static_cast<std::size_t>(group) * s.rows;
                for (int row = begin; row < end; ++row) output[row] += radix * static_cast<std::int32_t>(entry[row]);
            }
        }
        for (int row = begin; row < end; ++row) {
            float value = 0.0f;
            for (int group = 0; group < s.groups; ++group) {
                const int dot = group_dot[static_cast<std::size_t>(group) * s.rows + row]
                    - 128 * data.high_sum[static_cast<std::size_t>(row) * s.groups + group];
                const std::size_t index = static_cast<std::size_t>(row) * s.groups + group;
                value += data.activation_scales[group] *
                    (data.alpha[index] * static_cast<float>(4 * dot) +
                     data.beta[index] * static_cast<float>(z_sum[group]));
            }
            data.output[row] = value;
        }
    };
    pool.run([&](int id) { worker(s.rows * id / threads, s.rows * (id + 1) / threads); });
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct ResidualHost {
    std::size_t code_row_bytes = 512;
    std::size_t row_stride = 768;
    std::uint8_t *pinned = nullptr;
};

ResidualHost load_residual(const std::string &dir, const std::string &projection, const Shape &s) {
    ResidualHost r;
    r.code_row_bytes = static_cast<std::size_t>(s.hidden) / 4;
    r.row_stride = r.code_row_bytes + static_cast<std::size_t>(s.groups) * sizeof(float);
    const auto codes = read_vector<std::uint8_t>(dir + "/" + projection + ".qlo2.rowpacked.bin",
        static_cast<std::size_t>(s.rows) * r.code_row_bytes);
    const auto alpha = read_vector<float>(dir + "/" + projection + ".alpha.f32.bin",
        static_cast<std::size_t>(s.rows) * s.groups);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&r.pinned), static_cast<std::size_t>(s.rows) * r.row_stride));
    for (int row = 0; row < s.rows; ++row) {
        auto *dst = r.pinned + static_cast<std::size_t>(row) * r.row_stride;
        std::memcpy(dst, codes.data() + static_cast<std::size_t>(row) * r.code_row_bytes, r.code_row_bytes);
        std::memcpy(dst + r.code_row_bytes, alpha.data() + static_cast<std::size_t>(row) * s.groups,
            static_cast<std::size_t>(s.groups) * sizeof(float));
    }
    return r;
}

__global__ void residual_kernel(const std::uint8_t *payload, float *output, const std::int8_t *z,
                                const float *scales, int rows, int hidden, int groups,
                                int row_stride, int code_row_bytes) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const auto *code_row = payload + static_cast<std::size_t>(row) * row_stride;
    const auto *alpha = reinterpret_cast<const float *>(code_row + code_row_bytes);
    float sum = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        const int q = (code_row[index >> 2] >> ((index & 3) * 2)) & 3;
        sum += static_cast<float>(q) * alpha[index >> 5] * static_cast<float>(z[index]) * scales[index >> 5];
    }
    __shared__ float shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = shared[0];
}

__global__ void swiglu_down_kernel(const float *gate_base, const float *up_base,
                                   const float *gate_residual, const float *up_residual,
                                   const __half *down, float *partial, int tile_index,
                                   int tile_rows, int ffn, int hidden) {
    const int out = blockIdx.x;
    if (out >= hidden) return;
    float sum = 0.0f;
    for (int row = threadIdx.x; row < tile_rows; row += blockDim.x) {
        const float gate = gate_base[row] + gate_residual[row];
        const float up = up_base[row] + up_residual[row];
        const float hidden_value = gate / (1.0f + expf(-gate)) * up;
        const float weight = __half2float(down[static_cast<std::size_t>(out) * ffn +
            static_cast<std::size_t>(tile_index) * tile_rows + row]);
        sum += hidden_value * weight;
    }
    __shared__ float shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[static_cast<std::size_t>(tile_index) * hidden + out] = shared[0];
}

struct FullBuffers {
    std::uint8_t *gate_payload[2] = {};
    std::uint8_t *up_payload[2] = {};
    float *gate_base[2] = {};
    float *up_base[2] = {};
    float *gate_residual[2] = {};
    float *up_residual[2] = {};
    float *partial = nullptr;
};

void alloc_buffers(FullBuffers &b, std::size_t packet_bytes, int tile_rows, int tile_count, int hidden) {
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b.gate_payload[slot]), packet_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b.up_payload[slot]), packet_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b.gate_base[slot]), tile_rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b.up_base[slot]), tile_rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b.gate_residual[slot]), tile_rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b.up_residual[slot]), tile_rows * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&b.partial), static_cast<std::size_t>(tile_count) * hidden * sizeof(float)));
}

void free_buffers(FullBuffers &b) {
    for (int slot = 0; slot < 2; ++slot) {
        cudaFree(b.gate_payload[slot]); cudaFree(b.up_payload[slot]);
        cudaFree(b.gate_base[slot]); cudaFree(b.up_base[slot]);
        cudaFree(b.gate_residual[slot]); cudaFree(b.up_residual[slot]);
    }
    cudaFree(b.partial);
}

void enqueue_tile(const FullBuffers &b, int slot, int tile, const Shape &s,
                  const ResidualHost &gate, const ResidualHost &up,
                  const std::int8_t *device_z, const float *device_scales,
                  const __half *device_down, cudaStream_t stream) {
    residual_kernel<<<s.tile_rows, 256, 0, stream>>>(b.gate_payload[slot], b.gate_residual[slot],
        device_z, device_scales, s.tile_rows, s.hidden, s.groups,
        static_cast<int>(gate.row_stride), static_cast<int>(gate.code_row_bytes));
    CUDA_CHECK(cudaGetLastError());
    residual_kernel<<<s.tile_rows, 256, 0, stream>>>(b.up_payload[slot], b.up_residual[slot],
        device_z, device_scales, s.tile_rows, s.hidden, s.groups,
        static_cast<int>(up.row_stride), static_cast<int>(up.code_row_bytes));
    CUDA_CHECK(cudaGetLastError());
    swiglu_down_kernel<<<s.hidden, 256, 0, stream>>>(b.gate_base[slot], b.up_base[slot],
        b.gate_residual[slot], b.up_residual[slot], device_down, b.partial,
        tile, s.tile_rows, s.rows, s.hidden);
    CUDA_CHECK(cudaGetLastError());
}

double gpu_pipeline(const FullBuffers &b, const Shape &s, const ResidualHost &gate, const ResidualHost &up,
                    const std::vector<float> &gate_base, const std::vector<float> &up_base,
                    const __half *device_down, const std::int8_t *device_z, const float *device_scales,
                    bool overlap) {
    const std::size_t tile_bytes = static_cast<std::size_t>(s.tile_rows) * gate.row_stride;
    const int tiles = s.rows / s.tile_rows;
    const auto start = std::chrono::steady_clock::now();
    if (!overlap) {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (int tile = 0; tile < tiles; ++tile) {
            const auto offset = static_cast<std::size_t>(tile) * tile_bytes;
            CUDA_CHECK(cudaMemcpyAsync(b.gate_payload[0], gate.pinned + offset, tile_bytes,
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(b.up_payload[0], up.pinned + offset, tile_bytes,
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(b.gate_base[0], gate_base.data() + tile * s.tile_rows,
                s.tile_rows * sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(b.up_base[0], up_base.data() + tile * s.tile_rows,
                s.tile_rows * sizeof(float), cudaMemcpyHostToDevice, stream));
            enqueue_tile(b, 0, tile, s, gate, up, device_z, device_scales, device_down, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        cudaStream_t copy_stream = nullptr, compute_stream = nullptr;
        cudaEvent_t ready[2] = {}, done[2] = {};
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
        for (int slot = 0; slot < 2; ++slot) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ready[slot], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&done[slot], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(done[slot], compute_stream));
        }
        for (int tile = 0; tile < tiles; ++tile) {
            const int slot = tile & 1;
            const auto offset = static_cast<std::size_t>(tile) * tile_bytes;
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, done[slot], 0));
            CUDA_CHECK(cudaMemcpyAsync(b.gate_payload[slot], gate.pinned + offset, tile_bytes,
                cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaMemcpyAsync(b.up_payload[slot], up.pinned + offset, tile_bytes,
                cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaMemcpyAsync(b.gate_base[slot], gate_base.data() + tile * s.tile_rows,
                s.tile_rows * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaMemcpyAsync(b.up_base[slot], up_base.data() + tile * s.tile_rows,
                s.tile_rows * sizeof(float), cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaEventRecord(ready[slot], copy_stream));
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, ready[slot], 0));
            enqueue_tile(b, slot, tile, s, gate, up, device_z, device_scales, device_down, compute_stream);
            CUDA_CHECK(cudaEventRecord(done[slot], compute_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        for (int slot = 0; slot < 2; ++slot) { cudaEventDestroy(ready[slot]); cudaEventDestroy(done[slot]); }
        cudaStreamDestroy(copy_stream); cudaStreamDestroy(compute_stream);
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 9) {
            std::cerr << "usage: exact_cpu_base_gpu_full_ffn_runner TABLE_DIR RESIDUAL_DIR DOWN_FP16 OUTPUT_JSON BLOCK_SIZE TILE_ROWS REPEATS THREADS\n";
            return 2;
        }
        const std::string table_dir = argv[1], residual_dir = argv[2], down_path = argv[3], output_path = argv[4];
        Shape s;
        s.block_size = std::stoi(argv[5]);
        s.tile_rows = std::stoi(argv[6]);
        const int repeats = std::stoi(argv[7]);
        const int threads = std::stoi(argv[8]);
        if (s.block_size != 2 && s.block_size != 4) throw std::runtime_error("block size must be 2 or 4");
        s.blocks = s.hidden / s.block_size;
        s.state_count = 1;
        for (int i = 0; i < s.block_size; ++i) s.state_count *= 4;
        s.blocks_per_group = 32 / s.block_size;
        if (s.rows % s.tile_rows != 0 || repeats <= 0) throw std::runtime_error("invalid tile/repeat arguments");

        const auto down_host = read_vector<std::uint16_t>(down_path, static_cast<std::size_t>(s.hidden) * s.rows);
        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        const auto z = read_vector<std::int8_t>(table_dir + "/activation.z.i8.bin", s.hidden);
        const auto scales = read_vector<float>(table_dir + "/activation.scale.f32.bin", s.groups);
        std::int8_t *device_z = nullptr; float *device_scales = nullptr; __half *device_down = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_z), z.size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_scales), scales.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_down), down_host.size() * sizeof(std::uint16_t)));
        CUDA_CHECK(cudaMemcpy(device_z, z.data(), z.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_down, down_host.data(), down_host.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

        CpuPool pool(threads);
        CpuTable gate_table = load_cpu_table(table_dir, "gate", s);
        CpuTable up_table = load_cpu_table(table_dir, "up", s);
        ResidualHost gate = load_residual(residual_dir, "gate", s);
        ResidualHost up = load_residual(residual_dir, "up", s);
        std::vector<float> gate_base(s.rows), up_base(s.rows);
        FullBuffers buffers;
        const std::size_t packet_bytes = static_cast<std::size_t>(s.tile_rows) * gate.row_stride;
        alloc_buffers(buffers, packet_bytes, s.tile_rows, s.rows / s.tile_rows, s.hidden);

        for (int warmup = 0; warmup < 2; ++warmup) {
            cpu_eval(gate_table, s, threads, pool); gate_base = gate_table.output;
            cpu_eval(up_table, s, threads, pool); up_base = up_table.output;
            gpu_pipeline(buffers, s, gate, up, gate_base, up_base, device_down, device_z, device_scales, true);
        }
        std::vector<double> cpu_times, gpu_serial_times, serial_times, overlap_times;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            cpu_times.push_back(cpu_eval(gate_table, s, threads, pool)); gate_base = gate_table.output;
            const double up_cpu = cpu_eval(up_table, s, threads, pool); up_base = up_table.output;
            cpu_times.back() += up_cpu;
            gpu_serial_times.push_back(gpu_pipeline(buffers, s, gate, up, gate_base, up_base,
                device_down, device_z, device_scales, false));
            serial_times.push_back(cpu_times.back() + gpu_serial_times.back());
            overlap_times.push_back(gpu_pipeline(buffers, s, gate, up, gate_base, up_base,
                device_down, device_z, device_scales, true));
        }

        std::ofstream output(output_path, std::ios::trunc);
        output << std::setprecision(8) << "{\n";
        output << "  \"experiment\": \"cpu_base_gpu_residual_swiglu_down_v1\",\n";
        output << "  \"gpu\": \"" << prop.name << "\",\n";
        output << "  \"shape\": {\"rows\": " << s.rows << ", \"hidden\": " << s.hidden
               << ", \"tile_rows\": " << s.tile_rows << ", \"tile_count\": " << (s.rows / s.tile_rows)
               << ", \"cpu_threads\": " << threads << "},\n";
        output << "  \"cpu_base_gate_up_ms\": " << median(cpu_times) << ",\n";
        output << "  \"gpu_serial_copy_compute_ms\": " << median(gpu_serial_times) << ",\n";
        output << "  \"serial_cpu_plus_gpu_ms\": " << median(serial_times) << ",\n";
        output << "  \"gpu_double_buffer_overlap_ms\": " << median(overlap_times) << ",\n";
        const double cpu_ms = median(cpu_times);
        const double gpu_serial_ms = median(gpu_serial_times);
        const double gpu_overlap_ms = median(overlap_times);
        output << "  \"gpu_overlap_speedup_vs_gpu_serial\": " << gpu_serial_ms / std::max(gpu_overlap_ms, 1.0e-12) << ",\n";
        output << "  \"cpu_gpu_critical_lower_bound_ms\": " << std::max(cpu_ms, gpu_overlap_ms) << ",\n";
        output << "  \"serial_to_cpu_gpu_lower_bound_speedup\": " << median(serial_times)
               / std::max(std::max(cpu_ms, gpu_overlap_ms), 1.0e-12) << ",\n";
        output << "  \"h2d\": {\"residual_gate_up_bytes\": "
               << static_cast<std::size_t>(2) * s.rows * gate.row_stride
               << ", \"base_gate_up_bytes\": " << static_cast<std::size_t>(2) * s.rows * sizeof(float)
               << ", \"resident_down_bytes\": " << down_host.size() * sizeof(std::uint16_t) << "},\n";
        output << "  \"correctness\": {\"status\": \"pipeline kernel completed; Python bridge remains reference for GGUF fp32 comparison\"}\n";
        output << "}\n";
        cudaFreeHost(gate.pinned); cudaFreeHost(up.pinned); free_buffers(buffers);
        cudaFree(device_z); cudaFree(device_scales); cudaFree(device_down);
        std::cout << "wrote " << output_path << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
