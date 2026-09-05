#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
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
    int tile_rows = 1024;
    int block_size = 4;
    int blocks = 512;
    int state_count = 256;
    int blocks_per_group = 8;
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

CpuTable load_cpu_table(const std::string &table_dir, const std::string &projection, const Shape &shape) {
    CpuTable result;
    result.table = read_vector<std::uint8_t>(table_dir + "/" + projection + ".table.u8.bin",
        static_cast<std::size_t>(shape.blocks) * shape.state_count * shape.rows);
    result.high_sum = read_vector<std::int16_t>(table_dir + "/" + projection + ".high_sum.i16.bin",
        static_cast<std::size_t>(shape.rows) * shape.groups);
    result.alpha = read_vector<float>(table_dir + "/" + projection + ".alpha.f32.bin",
        static_cast<std::size_t>(shape.rows) * shape.groups);
    result.beta = read_vector<float>(table_dir + "/" + projection + ".beta.f32.bin",
        static_cast<std::size_t>(shape.rows) * shape.groups);
    result.states = read_vector<std::uint16_t>(table_dir + "/states.u16.bin",
        static_cast<std::size_t>(4) * shape.blocks);
    result.z = read_vector<std::int8_t>(table_dir + "/activation.z.i8.bin", shape.hidden);
    result.activation_scales = read_vector<float>(table_dir + "/activation.scale.f32.bin", shape.groups);
    result.output.resize(shape.rows);
    return result;
}

class CpuPool {
public:
    explicit CpuPool(int count) : count_(std::max(1, count)) {
        workers_.reserve(static_cast<std::size_t>(count_));
        for (int id = 0; id < count_; ++id) workers_.emplace_back([this, id] { worker_loop(id); });
    }
    ~CpuPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            condition_.notify_all();
        }
        for (auto &worker : workers_) worker.join();
    }
    CpuPool(const CpuPool &) = delete;
    CpuPool &operator=(const CpuPool &) = delete;

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
    void worker_loop(int id) {
        std::unique_lock<std::mutex> lock(mutex_);
        std::uint64_t local_generation = 0;
        while (!stopping_) {
            condition_.wait(lock, [this, local_generation] {
                return stopping_ || generation_ != local_generation;
            });
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

double cpu_eval(CpuTable &data, const Shape &shape, int thread_count, CpuPool *pool) {
    thread_count = std::max(1, std::min(thread_count, shape.rows));
    std::vector<std::int32_t> group_dot(static_cast<std::size_t>(shape.groups) * shape.rows, 0);
    std::int32_t z_sum[64] = {};
    for (int group = 0; group < shape.groups; ++group) {
        for (int i = 0; i < 32; ++i) z_sum[group] += data.z[group * 32 + i];
    }
    const auto start = std::chrono::steady_clock::now();
    auto worker = [&](int begin, int end) {
        for (int digit = 0; digit < 4; ++digit) {
            const int radix = 1 << (2 * digit);
            for (int block = 0; block < shape.blocks; ++block) {
                const std::uint16_t state = data.states[static_cast<std::size_t>(digit) * shape.blocks + block];
                const auto *entry = data.table.data()
                    + (static_cast<std::size_t>(block) * shape.state_count + state) * shape.rows;
                const int group = block / shape.blocks_per_group;
                auto *output = group_dot.data() + static_cast<std::size_t>(group) * shape.rows;
                for (int row = begin; row < end; ++row) {
                    output[row] += radix * static_cast<std::int32_t>(entry[row]);
                }
            }
        }
        for (int row = begin; row < end; ++row) {
            float value = 0.0f;
            for (int group = 0; group < shape.groups; ++group) {
                const int dot = group_dot[static_cast<std::size_t>(group) * shape.rows + row]
                    - 128 * data.high_sum[static_cast<std::size_t>(row) * shape.groups + group];
                const std::size_t index = static_cast<std::size_t>(row) * shape.groups + group;
                value += data.activation_scales[group]
                    * (data.alpha[index] * static_cast<float>(4 * dot)
                        + data.beta[index] * static_cast<float>(z_sum[group]));
            }
            data.output[row] = value;
        }
    };
    if (thread_count == 1 || pool == nullptr) {
        worker(0, shape.rows);
    } else {
        pool->run([&](int thread) {
            worker(shape.rows * thread / thread_count, shape.rows * (thread + 1) / thread_count);
        });
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

struct ResidualHost {
    std::vector<std::uint8_t> payload;
    std::size_t code_row_bytes = 512;
    std::size_t row_stride = 768;
    std::uint8_t *pinned = nullptr;
};

ResidualHost load_residual(const std::string &artifact_dir, const std::string &projection, const Shape &shape) {
    ResidualHost result;
    result.payload.resize(static_cast<std::size_t>(shape.rows) * result.row_stride);
    const auto codes = read_vector<std::uint8_t>(artifact_dir + "/" + projection + ".qlo2.rowpacked.bin",
        static_cast<std::size_t>(shape.rows) * result.code_row_bytes);
    const auto alpha = read_vector<float>(artifact_dir + "/" + projection + ".alpha.f32.bin",
        static_cast<std::size_t>(shape.rows) * shape.groups);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&result.pinned), result.payload.size()));
    for (int row = 0; row < shape.rows; ++row) {
        auto *dst = result.pinned + static_cast<std::size_t>(row) * result.row_stride;
        std::memcpy(dst, codes.data() + static_cast<std::size_t>(row) * result.code_row_bytes, result.code_row_bytes);
        std::memcpy(dst + result.code_row_bytes, alpha.data() + static_cast<std::size_t>(row) * shape.groups,
            static_cast<std::size_t>(shape.groups) * sizeof(float));
    }
    return result;
}

__global__ void residual_kernel(
        const std::uint8_t *payload, float *output, const std::int8_t *z, const float *scales,
        int rows, int hidden, int groups, int row_stride, int code_row_bytes) {
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

struct GpuBuffers { std::uint8_t *payload[2] = {}; float *output[2] = {}; };

void alloc_gpu(GpuBuffers &buffers, std::size_t tile_bytes, int tile_rows) {
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.payload[slot]), tile_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.output[slot]), tile_rows * sizeof(float)));
    }
}
void free_gpu(GpuBuffers &buffers) {
    for (int slot = 0; slot < 2; ++slot) { cudaFree(buffers.payload[slot]); cudaFree(buffers.output[slot]); }
}

void enqueue_kernel(GpuBuffers &buffers, int slot, const Shape &shape, const ResidualHost &host,
                    const std::int8_t *device_z, const float *device_scales, cudaStream_t stream) {
    residual_kernel<<<shape.tile_rows, 256, 0, stream>>>(buffers.payload[slot], buffers.output[slot],
        device_z, device_scales, shape.tile_rows, shape.hidden, shape.groups,
        static_cast<int>(host.row_stride), static_cast<int>(host.code_row_bytes));
    CUDA_CHECK(cudaGetLastError());
}

struct Validation { double max_abs = 0.0; double rel_l2 = 0.0; };

Validation validate_first_residual(const ResidualHost &host, GpuBuffers &buffers, const Shape &shape,
                                   const std::vector<std::int8_t> &z, const std::vector<float> &scales,
                                   const std::int8_t *device_z, const float *device_scales) {
    const std::size_t tile_bytes = static_cast<std::size_t>(shape.tile_rows) * host.row_stride;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(buffers.payload[0], host.pinned, tile_bytes, cudaMemcpyHostToDevice, stream));
    enqueue_kernel(buffers, 0, shape, host, device_z, device_scales, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> gpu(static_cast<std::size_t>(shape.tile_rows));
    CUDA_CHECK(cudaMemcpy(gpu.data(), buffers.output[0], gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double error_sq = 0.0;
    double target_sq = 0.0;
    double max_abs = 0.0;
    for (int row = 0; row < shape.tile_rows; ++row) {
        const auto *code_row = host.pinned + static_cast<std::size_t>(row) * host.row_stride;
        const auto *alpha = reinterpret_cast<const float *>(code_row + host.code_row_bytes);
        double target = 0.0;
        for (int index = 0; index < shape.hidden; ++index) {
            const int q = (code_row[index >> 2] >> ((index & 3) * 2)) & 3;
            target += static_cast<double>(q) * alpha[index >> 5]
                * static_cast<double>(z[index]) * static_cast<double>(scales[index >> 5]);
        }
        const double error = static_cast<double>(gpu[row]) - target;
        error_sq += error * error;
        target_sq += target * target;
        max_abs = std::max(max_abs, std::abs(error));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return {max_abs, std::sqrt(error_sq) / std::max(std::sqrt(target_sq), 1.0e-12)};
}

double gpu_pipeline(const ResidualHost &host, GpuBuffers &buffers, const Shape &shape,
                    const std::int8_t *device_z, const float *device_scales, bool overlap) {
    const std::size_t tile_bytes = static_cast<std::size_t>(shape.tile_rows) * host.row_stride;
    const int tile_count = shape.rows / shape.tile_rows;
    auto start = std::chrono::steady_clock::now();
    if (!overlap) {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (int tile = 0; tile < tile_count; ++tile) {
            const auto offset = static_cast<std::size_t>(tile) * tile_bytes;
            CUDA_CHECK(cudaMemcpyAsync(buffers.payload[0], host.pinned + offset, tile_bytes,
                cudaMemcpyHostToDevice, stream));
            enqueue_kernel(buffers, 0, shape, host, device_z, device_scales, stream);
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
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        for (int tile = 0; tile < tile_count; ++tile) {
            const int slot = tile & 1;
            const auto offset = static_cast<std::size_t>(tile) * tile_bytes;
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, done[slot], 0));
            CUDA_CHECK(cudaMemcpyAsync(buffers.payload[slot], host.pinned + offset, tile_bytes,
                cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaEventRecord(ready[slot], copy_stream));
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, ready[slot], 0));
            enqueue_kernel(buffers, slot, shape, host, device_z, device_scales, compute_stream);
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
        if (argc != 8) {
            std::cerr << "usage: exact_main_residual_cuda_runner TABLE_DIR RESIDUAL_DIR OUTPUT_JSON BLOCK_SIZE TILE_ROWS REPEATS THREADS\n";
            return 2;
        }
        const std::string table_dir = argv[1], residual_dir = argv[2], output_path = argv[3];
        Shape shape;
        shape.block_size = std::stoi(argv[4]);
        shape.tile_rows = std::stoi(argv[5]);
        const int repeats = std::stoi(argv[6]);
        const int thread_count = std::stoi(argv[7]);
        if (shape.block_size != 2 && shape.block_size != 4 && shape.block_size != 8) {
            throw std::runtime_error("block size must be 2, 4, or 8");
        }
        shape.blocks = shape.hidden / shape.block_size;
        shape.state_count = 1;
        for (int i = 0; i < shape.block_size; ++i) shape.state_count *= 4;
        shape.blocks_per_group = 32 / shape.block_size;
        if (shape.rows % shape.tile_rows != 0 || repeats <= 0) throw std::runtime_error("invalid tile/repeat arguments");

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        const auto z = read_vector<std::int8_t>(table_dir + "/activation.z.i8.bin", shape.hidden);
        const auto scales = read_vector<float>(table_dir + "/activation.scale.f32.bin", shape.groups);
        std::int8_t *device_z = nullptr; float *device_scales = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_z), z.size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_scales), scales.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(device_z, z.data(), z.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

        struct Row { double cpu = 0, gpu_only = 0, gpu_overlap = 0, serial_total = 0, concurrent = 0;
            double validation_max_abs = 0, validation_rel_l2 = 0; std::uint64_t checksum = 0; };
        Row rows[2];
        CpuPool cpu_pool(thread_count);
        for (int index = 0; index < 2; ++index) {
            const std::string projection = index == 0 ? "gate" : "up";
            CpuTable cpu = load_cpu_table(table_dir, projection, shape);
            ResidualHost residual = load_residual(residual_dir, projection, shape);
            GpuBuffers buffers; const std::size_t tile_bytes = static_cast<std::size_t>(shape.tile_rows) * residual.row_stride;
            alloc_gpu(buffers, tile_bytes, shape.tile_rows);
            const Validation validation = validate_first_residual(residual, buffers, shape, z, scales, device_z, device_scales);
            rows[index].validation_max_abs = validation.max_abs;
            rows[index].validation_rel_l2 = validation.rel_l2;
            for (int warmup = 0; warmup < 2; ++warmup) {
                cpu_eval(cpu, shape, thread_count, &cpu_pool);
                gpu_pipeline(residual, buffers, shape, device_z, device_scales, true);
            }
            std::vector<double> cpu_times, gpu_serial_times, serial_times, overlap_times, concurrent_times;
            for (int repeat = 0; repeat < repeats; ++repeat) {
                const double cpu_part = cpu_eval(cpu, shape, thread_count, &cpu_pool);
                cpu_times.push_back(cpu_part);
                const double gpu_part = gpu_pipeline(residual, buffers, shape, device_z, device_scales, false);
                gpu_serial_times.push_back(gpu_part);
                serial_times.push_back(cpu_part + gpu_part);
                overlap_times.push_back(gpu_pipeline(residual, buffers, shape, device_z, device_scales, true));
                const auto concurrent_start = std::chrono::steady_clock::now();
                auto future = std::async(std::launch::async, [&cpu, &shape, &cpu_pool, thread_count]() {
                    return cpu_eval(cpu, shape, thread_count, &cpu_pool);
                });
                const double gpu_concurrent = gpu_pipeline(residual, buffers, shape, device_z, device_scales, true);
                const double cpu_concurrent = future.get();
                const double concurrent_wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - concurrent_start).count();
                (void)cpu_concurrent; (void)gpu_concurrent;
                concurrent_times.push_back(concurrent_wall);
                rows[index].checksum += static_cast<std::uint64_t>(std::abs(cpu.output[repeat % shape.rows]));
            }
            rows[index].cpu = median(cpu_times);
            rows[index].serial_total = median(serial_times);
            rows[index].gpu_only = median(gpu_serial_times);
            rows[index].gpu_overlap = median(overlap_times);
            rows[index].concurrent = median(concurrent_times);
            cudaFreeHost(residual.pinned); free_gpu(buffers);
        }
        std::ofstream output(output_path, std::ios::trunc);
        output << std::setprecision(8) << "{\n";
        output << "  \"experiment\": \"cuda_cpu_main_plus_exact_residual_v1\",\n";
        output << "  \"gpu\": \"" << properties.name << "\",\n";
        output << "  \"shape\": {\"rows\": " << shape.rows << ", \"hidden\": " << shape.hidden
               << ", \"tile_rows\": " << shape.tile_rows << ", \"cpu_threads\": " << thread_count << "},\n";
        output << "  \"projections\": {\n";
        for (int index = 0; index < 2; ++index) {
            output << "    \"" << (index == 0 ? "gate" : "up") << "\": {\"cpu_main_ms\": " << rows[index].cpu
                   << ", \"gpu_serial_copy_compute_ms\": " << rows[index].gpu_only
                   << ", \"gpu_overlap_ms\": " << rows[index].gpu_overlap
                   << ", \"residual_validation_max_abs\": " << rows[index].validation_max_abs
                   << ", \"residual_validation_rel_l2\": " << rows[index].validation_rel_l2
                   << ", \"serial_cpu_plus_gpu_ms\": " << rows[index].serial_total
                   << ", \"cpu_gpu_concurrent_wall_ms\": " << rows[index].concurrent
                   << ", \"serial_to_concurrent_speedup\": " << rows[index].serial_total / std::max(rows[index].concurrent, 1.0e-12)
                   << "}" << (index == 0 ? ",\n" : "\n");
        }
        output << "  },\n";
        output << "  \"pair\": {\"serial_cpu_plus_gpu_ms\": " << rows[0].serial_total + rows[1].serial_total
               << ", \"concurrent_wall_ms\": " << rows[0].concurrent + rows[1].concurrent
               << ", \"speedup\": " << (rows[0].serial_total + rows[1].serial_total)
               / std::max(rows[0].concurrent + rows[1].concurrent, 1.0e-12) << "}\n";
        output << "}\n";
        cudaFree(device_z); cudaFree(device_scales);
        std::cout << "wrote " << output_path << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
