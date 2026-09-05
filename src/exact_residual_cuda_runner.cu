#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << expr << " failed at " << file << ':' << line << ": "
                << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

std::vector<std::uint8_t> read_bytes(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("cannot size " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (size != 0 && !input.read(reinterpret_cast<char *>(data.data()), size)) {
        throw std::runtime_error("cannot read " + path);
    }
    return data;
}

std::string read_text(const std::string &path) {
    const auto bytes = read_bytes(path);
    return std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

std::size_t json_integer_after(const std::string &text, const std::string &key) {
    const std::size_t key_pos = text.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("manifest missing " + key);
    }
    std::size_t pos = key_pos + key.size();
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == ':')) {
        ++pos;
    }
    std::size_t end = pos;
    while (end < text.size() && text[end] >= '0' && text[end] <= '9') {
        ++end;
    }
    if (end == pos) {
        throw std::runtime_error("manifest value is not an integer for " + key);
    }
    return static_cast<std::size_t>(std::stoull(text.substr(pos, end - pos)));
}

struct ProjectionHost {
    std::string name;
    std::vector<std::uint8_t> codes;
    std::vector<float> alpha;
    std::uint8_t *payload = nullptr;
    std::size_t row_stride = 0;
    std::size_t code_row_bytes = 0;
    std::size_t alpha_row_bytes = 0;
};

struct ProjectionShape {
    int rows = 0;
    int hidden = 0;
    int groups = 0;
    int packed_row_bytes = 0;
};

ProjectionHost load_projection(
        const std::string &artifact_dir,
        const std::string &name,
        const ProjectionShape &shape) {
    ProjectionHost result;
    result.name = name;
    const std::string prefix = artifact_dir + "/" + name;
    result.codes = read_bytes(prefix + ".qlo2.rowpacked.bin");
    const auto alpha_bytes = read_bytes(prefix + ".alpha.f32.bin");
    if (result.codes.size() != static_cast<std::size_t>(shape.rows) * shape.packed_row_bytes) {
        throw std::runtime_error(name + " packed code size does not match manifest");
    }
    if (alpha_bytes.size() != static_cast<std::size_t>(shape.rows) * shape.groups * sizeof(float)) {
        throw std::runtime_error(name + " alpha size does not match manifest");
    }
    result.alpha.resize(alpha_bytes.size() / sizeof(float));
    std::memcpy(result.alpha.data(), alpha_bytes.data(), alpha_bytes.size());
    result.code_row_bytes = static_cast<std::size_t>(shape.packed_row_bytes);
    result.alpha_row_bytes = static_cast<std::size_t>(shape.groups) * sizeof(float);
    result.row_stride = result.code_row_bytes + result.alpha_row_bytes;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&result.payload),
        static_cast<std::size_t>(shape.rows) * result.row_stride));
    for (int row = 0; row < shape.rows; ++row) {
        auto * destination = result.payload + static_cast<std::size_t>(row) * result.row_stride;
        std::memcpy(destination,
            result.codes.data() + static_cast<std::size_t>(row) * result.code_row_bytes,
            result.code_row_bytes);
        std::memcpy(destination + result.code_row_bytes,
            result.alpha.data() + static_cast<std::size_t>(row) * shape.groups,
            result.alpha_row_bytes);
    }
    return result;
}

__global__ void residual_kernel(
        const std::uint8_t *payload,
        float *output,
        const std::int8_t *z,
        const float *activation_scales,
        int rows,
        int hidden,
        int groups,
        int packed_row_bytes,
        int row_stride,
        int code_row_bytes) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const auto *code_row = payload + static_cast<std::size_t>(row) * row_stride;
    const auto *alpha_row = reinterpret_cast<const float *>(code_row + code_row_bytes);
    float sum = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        const std::uint8_t packed = code_row[index >> 2];
        const int q = (packed >> ((index & 3) * 2)) & 3;
        const int group = index >> 5;
        sum += static_cast<float>(q) * alpha_row[group]
            * static_cast<float>(z[index]) * activation_scales[group];
    }
    __shared__ float shared[256];
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = shared[0];
    }
}

struct Timing {
    double copy_ms = 0.0;
    double kernel_ms = 0.0;
    double total_ms = 0.0;
    double host_ms = 0.0;
};

double event_elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return static_cast<double>(milliseconds);
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

struct DeviceBuffers {
    std::uint8_t *payload[2] = {nullptr, nullptr};
    float *output[2] = {nullptr, nullptr};
};

void allocate_buffers(DeviceBuffers &buffers, std::size_t tile_bytes, int tile_rows) {
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.payload[slot]), tile_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffers.output[slot]),
            static_cast<std::size_t>(tile_rows) * sizeof(float)));
    }
}

void free_buffers(DeviceBuffers &buffers) {
    for (int slot = 0; slot < 2; ++slot) {
        if (buffers.payload[slot] != nullptr) cudaFree(buffers.payload[slot]);
        if (buffers.output[slot] != nullptr) cudaFree(buffers.output[slot]);
        buffers.payload[slot] = nullptr;
        buffers.output[slot] = nullptr;
    }
}

void launch(
        const ProjectionHost &projection,
        const ProjectionShape &shape,
        DeviceBuffers &buffers,
        int slot,
        int tile_rows,
        std::size_t tile_bytes,
        const std::int8_t *device_z,
        const float *device_scales,
        cudaStream_t stream) {
    const int blocks = tile_rows;
    residual_kernel<<<blocks, 256, 0, stream>>>(
        buffers.payload[slot], buffers.output[slot], device_z, device_scales,
        tile_rows, shape.hidden, shape.groups, shape.packed_row_bytes,
        static_cast<int>(projection.row_stride), static_cast<int>(projection.code_row_bytes));
    CUDA_CHECK(cudaGetLastError());
    (void)tile_bytes;
}

Timing measure_copy_only(
        const ProjectionHost &projection,
        DeviceBuffers &buffers,
        int rows,
        int tile_rows,
        std::size_t tile_bytes,
        int repeats) {
    const int tile_count = rows / tile_rows;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> gpu_times;
    std::vector<double> host_times;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));
        const auto host_start = std::chrono::steady_clock::now();
        for (int tile = 0; tile < tile_count; ++tile) {
            const std::size_t offset = static_cast<std::size_t>(tile) * tile_rows * projection.row_stride;
            CUDA_CHECK(cudaMemcpyAsync(buffers.payload[0], projection.payload + offset, tile_bytes,
                cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        const auto host_stop = std::chrono::steady_clock::now();
        gpu_times.push_back(event_elapsed(start, stop));
        host_times.push_back(std::chrono::duration<double, std::milli>(host_stop - host_start).count());
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return {median(gpu_times), 0.0, median(gpu_times), median(host_times)};
}

Timing measure_kernel_only(
        const ProjectionHost &projection,
        const ProjectionShape &shape,
        DeviceBuffers &buffers,
        const std::int8_t *device_z,
        const float *device_scales,
        int rows,
        int tile_rows,
        std::size_t tile_bytes,
        int repeats) {
    const int tile_count = rows / tile_rows;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMemcpy(buffers.payload[0], projection.payload, tile_bytes, cudaMemcpyHostToDevice));
    std::vector<double> gpu_times;
    std::vector<double> host_times;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));
        const auto host_start = std::chrono::steady_clock::now();
        for (int tile = 0; tile < tile_count; ++tile) {
            launch(projection, shape, buffers, 0, tile_rows, tile_bytes, device_z, device_scales, stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        const auto host_stop = std::chrono::steady_clock::now();
        gpu_times.push_back(event_elapsed(start, stop));
        host_times.push_back(std::chrono::duration<double, std::milli>(host_stop - host_start).count());
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return {0.0, median(gpu_times), median(gpu_times), median(host_times)};
}

Timing measure_serial(
        const ProjectionHost &projection,
        const ProjectionShape &shape,
        DeviceBuffers &buffers,
        const std::int8_t *device_z,
        const float *device_scales,
        int rows,
        int tile_rows,
        std::size_t tile_bytes,
        int repeats) {
    const int tile_count = rows / tile_rows;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> gpu_times;
    std::vector<double> host_times;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));
        const auto host_start = std::chrono::steady_clock::now();
        for (int tile = 0; tile < tile_count; ++tile) {
            const std::size_t offset = static_cast<std::size_t>(tile) * tile_rows * projection.row_stride;
            CUDA_CHECK(cudaMemcpyAsync(buffers.payload[0], projection.payload + offset, tile_bytes,
                cudaMemcpyHostToDevice, stream));
            launch(projection, shape, buffers, 0, tile_rows, tile_bytes, device_z, device_scales, stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        const auto host_stop = std::chrono::steady_clock::now();
        gpu_times.push_back(event_elapsed(start, stop));
        host_times.push_back(std::chrono::duration<double, std::milli>(host_stop - host_start).count());
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return {0.0, 0.0, median(gpu_times), median(host_times)};
}

Timing measure_overlap(
        const ProjectionHost &projection,
        const ProjectionShape &shape,
        DeviceBuffers &buffers,
        const std::int8_t *device_z,
        const float *device_scales,
        int rows,
        int tile_rows,
        std::size_t tile_bytes,
        int repeats) {
    const int tile_count = rows / tile_rows;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t ready[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ready[slot], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&done[slot], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(done[slot], compute_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    std::vector<double> gpu_times;
    std::vector<double> host_times;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start, compute_stream));
        const auto host_start = std::chrono::steady_clock::now();
        for (int tile = 0; tile < tile_count; ++tile) {
            const int slot = tile & 1;
            const std::size_t offset = static_cast<std::size_t>(tile) * tile_rows * projection.row_stride;
            CUDA_CHECK(cudaStreamWaitEvent(copy_stream, done[slot], 0));
            CUDA_CHECK(cudaMemcpyAsync(buffers.payload[slot], projection.payload + offset, tile_bytes,
                cudaMemcpyHostToDevice, copy_stream));
            CUDA_CHECK(cudaEventRecord(ready[slot], copy_stream));
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, ready[slot], 0));
            launch(projection, shape, buffers, slot, tile_rows, tile_bytes, device_z, device_scales, compute_stream);
            CUDA_CHECK(cudaEventRecord(done[slot], compute_stream));
        }
        CUDA_CHECK(cudaEventRecord(stop, compute_stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        const auto host_stop = std::chrono::steady_clock::now();
        gpu_times.push_back(event_elapsed(start, stop));
        host_times.push_back(std::chrono::duration<double, std::milli>(host_stop - host_start).count());
    }
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaEventDestroy(ready[slot]));
        CUDA_CHECK(cudaEventDestroy(done[slot]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(copy_stream));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    return {0.0, 0.0, median(gpu_times), median(host_times)};
}

struct Validation {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
};

Validation validate_first_tile(
        const ProjectionHost &projection,
        const ProjectionShape &shape,
        DeviceBuffers &buffers,
        const std::int8_t *z,
        const float *scales,
        int tile_rows,
        std::size_t tile_bytes,
        const std::int8_t *device_z,
        const float *device_scales) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(buffers.payload[0], projection.payload, tile_bytes,
        cudaMemcpyHostToDevice, stream));
    launch(projection, shape, buffers, 0, tile_rows, tile_bytes, device_z, device_scales, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> gpu(static_cast<std::size_t>(tile_rows));
    CUDA_CHECK(cudaMemcpy(gpu.data(), buffers.output[0], gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double error_sq = 0.0;
    double target_sq = 0.0;
    double max_abs = 0.0;
    for (int row = 0; row < tile_rows; ++row) {
        const auto *code_row = projection.payload + static_cast<std::size_t>(row) * projection.row_stride;
        const auto *alpha_row = reinterpret_cast<const float *>(code_row + projection.code_row_bytes);
        double target = 0.0;
        for (int index = 0; index < shape.hidden; ++index) {
            const int q = (code_row[index >> 2] >> ((index & 3) * 2)) & 3;
            target += static_cast<double>(q) * alpha_row[index >> 5]
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

void write_json_string(std::ostream &out, const std::string &value) {
    out << '"';
    for (const char character : value) {
        if (character == '\\' || character == '"') out << '\\';
        out << character;
    }
    out << '"';
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 5) {
            std::cerr << "usage: exact_residual_cuda_runner ARTIFACT_DIR OUTPUT_JSON [TILE_ROWS] [REPEATS]\n";
            return 2;
        }
        const std::string artifact_dir = argv[1];
        const std::string output_path = argv[2];
        const int tile_rows = argc >= 4 ? std::stoi(argv[3]) : 1024;
        const int repeats = argc >= 5 ? std::stoi(argv[4]) : 7;
        if (tile_rows <= 0 || repeats <= 0) throw std::runtime_error("tile_rows and repeats must be positive");

        const std::string manifest = read_text(artifact_dir + "/manifest.json");
        ProjectionShape shape;
        shape.rows = static_cast<int>(json_integer_after(manifest, "\"rows\""));
        shape.hidden = static_cast<int>(json_integer_after(manifest, "\"hidden\""));
        shape.groups = shape.hidden / 32;
        shape.packed_row_bytes = shape.hidden / 4;
        if (shape.rows % tile_rows != 0 || shape.hidden % 32 != 0 || shape.hidden % 4 != 0) {
            throw std::runtime_error("artifact shape is incompatible with tile or QLO2 layout");
        }

        const auto z_bytes = read_bytes(artifact_dir + "/activation.z.i8.bin");
        const auto scale_bytes = read_bytes(artifact_dir + "/activation.scale.f32.bin");
        if (z_bytes.size() != static_cast<std::size_t>(shape.hidden) ||
            scale_bytes.size() != static_cast<std::size_t>(shape.groups) * sizeof(float)) {
            throw std::runtime_error("activation artifact shape does not match manifest");
        }
        std::vector<std::int8_t> z(shape.hidden);
        std::vector<float> scales(shape.groups);
        std::memcpy(z.data(), z_bytes.data(), z_bytes.size());
        std::memcpy(scales.data(), scale_bytes.data(), scale_bytes.size());

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::int8_t *device_z = nullptr;
        float *device_scales = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_z), z.size() * sizeof(std::int8_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_scales), scales.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(device_z, z.data(), z.size() * sizeof(std::int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<ProjectionHost> projections;
        projections.push_back(load_projection(artifact_dir, "gate", shape));
        projections.push_back(load_projection(artifact_dir, "up", shape));
        const std::size_t tile_bytes = static_cast<std::size_t>(tile_rows)
            * projections.front().row_stride;
        std::vector<Timing> copy_results;
        std::vector<Timing> kernel_results;
        std::vector<Timing> serial_results;
        std::vector<Timing> overlap_results;
        std::vector<Validation> validations;
        for (auto &projection : projections) {
            DeviceBuffers buffers;
            allocate_buffers(buffers, tile_bytes, tile_rows);
            validations.push_back(validate_first_tile(projection, shape, buffers, z.data(), scales.data(),
                tile_rows, tile_bytes, device_z, device_scales));
            copy_results.push_back(measure_copy_only(projection, buffers, shape.rows, tile_rows, tile_bytes, repeats));
            kernel_results.push_back(measure_kernel_only(projection, shape, buffers, device_z, device_scales,
                shape.rows, tile_rows, tile_bytes, repeats));
            serial_results.push_back(measure_serial(projection, shape, buffers, device_z, device_scales,
                shape.rows, tile_rows, tile_bytes, repeats));
            overlap_results.push_back(measure_overlap(projection, shape, buffers, device_z, device_scales,
                shape.rows, tile_rows, tile_bytes, repeats));
            free_buffers(buffers);
            cudaFreeHost(projection.payload);
            projection.payload = nullptr;
        }

        std::ofstream output(output_path, std::ios::trunc);
        if (!output) throw std::runtime_error("cannot write " + output_path);
        output << std::setprecision(8);
        output << "{\n";
        output << "  \"experiment\": \"cuda_exact_qlo2_residual_runner_v1\",\n";
        output << "  \"layer\": 23,\n";
        output << "  \"gpu\": {\"name\": ";
        write_json_string(output, properties.name);
        output << ", \"global_memory_bytes\": " << properties.totalGlobalMem << "},\n";
        output << "  \"shape\": {\"rows\": " << shape.rows << ", \"hidden\": " << shape.hidden
                << ", \"groups\": " << shape.groups << ", \"tile_rows\": " << tile_rows
                << ", \"tile_count\": " << (shape.rows / tile_rows) << "},\n";
        output << "  \"projections\": {\n";
        for (std::size_t index = 0; index < projections.size(); ++index) {
            const auto &projection = projections[index];
            const auto &copy = copy_results[index];
            const auto &kernel = kernel_results[index];
            const auto &serial = serial_results[index];
            const auto &overlap = overlap_results[index];
            const auto &validation = validations[index];
            const double copy_ms = copy.copy_ms;
            const double kernel_ms = kernel.kernel_ms;
            const double serial_ms = serial.total_ms;
            const double overlap_ms = overlap.total_ms;
            const double hidden_copy = std::max(0.0, copy_ms + kernel_ms - overlap_ms);
            const double visible_copy = std::max(0.0, overlap_ms - kernel_ms);
            output << "    ";
            write_json_string(output, projection.name);
            output << ": {\n";
            output << "      \"tile_payload_bytes\": " << tile_bytes << ",\n";
            output << "      \"full_payload_bytes\": " << (static_cast<std::size_t>(shape.rows) * projection.row_stride) << ",\n";
            output << "      \"validation_max_abs\": " << validation.max_abs << ",\n";
            output << "      \"validation_rel_l2\": " << validation.rel_l2 << ",\n";
            output << "      \"copy_only_gpu_ms\": " << copy_ms << ",\n";
            output << "      \"copy_only_host_ms\": " << copy.host_ms << ",\n";
            output << "      \"kernel_only_gpu_ms\": " << kernel_ms << ",\n";
            output << "      \"serial_copy_compute_gpu_ms\": " << serial_ms << ",\n";
            output << "      \"serial_copy_compute_host_ms\": " << serial.host_ms << ",\n";
            output << "      \"double_buffer_overlap_gpu_ms\": " << overlap_ms << ",\n";
            output << "      \"double_buffer_overlap_host_ms\": " << overlap.host_ms << ",\n";
            output << "      \"overlap_speedup_vs_serial\": " << (serial_ms / std::max(overlap_ms, 1.0e-12)) << ",\n";
            output << "      \"hidden_copy_ms\": " << hidden_copy << ",\n";
            output << "      \"visible_copy_wait_ms\": " << visible_copy << "\n";
            output << "    }" << (index + 1 == projections.size() ? "\n" : ",\n");
        }
        const double pair_serial = serial_results[0].total_ms + serial_results[1].total_ms;
        const double pair_overlap = overlap_results[0].total_ms + overlap_results[1].total_ms;
        output << "  },\n";
        output << "  \"pair\": {\n";
        output << "    \"full_payload_bytes\": "
                << static_cast<std::size_t>(shape.rows) * projections[0].row_stride * projections.size() << ",\n";
        output << "    \"serial_gpu_ms\": " << pair_serial << ",\n";
        output << "    \"overlap_gpu_ms\": " << pair_overlap << ",\n";
        output << "    \"overlap_speedup_vs_serial\": " << pair_serial / std::max(pair_overlap, 1.0e-12) << "\n";
        output << "  }\n";
        output << "}\n";

        cudaFree(device_z);
        cudaFree(device_scales);
        std::cout << "wrote " << output_path << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
