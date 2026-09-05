#pragma once

#include <immintrin.h>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace residual_mesh {

// Persistent workers: allocation and thread startup are cold-start work.
class CpuPool {
public:
    explicit CpuPool(int count) : count_(count) {
        if (count < 1) throw std::invalid_argument("threads must be positive");
        for (int id = 1; id < count_; ++id) workers_.emplace_back([this, id] {
            std::uint64_t seen = 0;
            std::unique_lock<std::mutex> lock(mutex_);
            while (true) {
                ready_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                auto task = task_;
                lock.unlock();
                task(id);
                lock.lock();
                if (--remaining_ == 0) finished_.notify_one();
            }
        });
    }
    ~CpuPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        ready_.notify_all();
        for (auto &worker : workers_) worker.join();
    }
    int size() const { return count_; }
    void run(const std::function<void(int)> &task) {
        if (count_ == 1) { task(0); return; }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_ = task;
            remaining_ = count_ - 1;
            ++generation_;
        }
        ready_.notify_all();
        task(0);
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [&] { return remaining_ == 0; });
    }
private:
    int count_, remaining_ = 0;
    bool stop_ = false;
    std::uint64_t generation_ = 0;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_, finished_;
    std::function<void(int)> task_;
};

struct Shape {
    int rows, hidden, block;
    int groups() const { return hidden / 32; }
    int blocks() const { return hidden / block; }
    int states() const { return 1 << (2 * block); }
    int blocks_per_group() const { return 32 / block; }
    void validate() const {
        if (rows <= 0 || hidden <= 0 || hidden % 32 || (block != 2 && block != 4))
            throw std::invalid_argument("need positive rows, hidden divisible by 32, block 2 or 4");
    }
};

struct Activation {
    std::vector<std::size_t> offsets;
    std::vector<int> sums;
    void resize(const Shape &s) { offsets.resize(s.blocks() * 4); sums.resize(s.groups()); }
    void prepare(const Shape &s, const std::int8_t *z, bool encode_offsets = true) {
        for (int g = 0; g < s.groups(); ++g) {
            int sum = 0;
            for (int i = 0; i < 32; ++i) sum += z[g * 32 + i];
            sums[g] = sum;
        }
        if (!encode_offsets) return;
        for (int b = 0; b < s.blocks(); ++b) {
            for (int d = 0; d < 4; ++d) {
                int state = 0;
                for (int i = 0; i < s.block; ++i)
                    state |= (((int(z[b * s.block + i]) + 128) >> (2 * d)) & 3) << (2 * i);
                offsets[b * 4 + d] = (std::size_t(b) * s.states() + state) * s.rows;
            }
        }
    }
};

// Metadata is transposed once at cold start; runtime SIMD lanes are output rows.
struct BaseView {
    const std::uint8_t *table;
    const std::int16_t *high_sum_gr;
    const float *alpha_gr, *beta_gr;
};

inline void legacy_base(const Shape &s, const BaseView &v, const Activation &a,
                        const float *scales, float *output, std::int32_t *scratch,
                        int begin, int end, std::int32_t *verified_dots = nullptr) {
    for (int g = 0; g < s.groups(); ++g)
        std::fill(scratch + std::size_t(g) * s.rows + begin,
                  scratch + std::size_t(g) * s.rows + end, 0);
    for (int d = 0; d < 4; ++d) {
        for (int b = 0; b < s.blocks(); ++b) {
            const auto *entry = v.table + a.offsets[b * 4 + d];
            auto *dot = scratch + std::size_t(b / s.blocks_per_group()) * s.rows;
            for (int r = begin; r < end; ++r) dot[r] += int(entry[r]) << (2 * d);
        }
    }
    for (int r = begin; r < end; ++r) {
        float value = 0;
        for (int g = 0; g < s.groups(); ++g) {
            const auto ix = std::size_t(g) * s.rows + r;
            const int dot = scratch[ix] - 128 * v.high_sum_gr[ix];
            if (verified_dots) verified_dots[std::size_t(r) * s.groups() + g] = dot;
            value += scales[g] * (v.alpha_gr[ix] * float(4 * dot) + v.beta_gr[ix] * float(a.sums[g]));
        }
        output[r] = value;
    }
}

inline __m256i radix16(const std::uint8_t *p0, const std::uint8_t *p1,
                      const std::uint8_t *p2, const std::uint8_t *p3) {
    const auto a = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p0)));
    const auto b = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p1)));
    const auto c = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p2)));
    const auto d = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p3)));
    return _mm256_add_epi16(_mm256_add_epi16(a, _mm256_slli_epi16(b, 2)),
                            _mm256_add_epi16(_mm256_slli_epi16(c, 4), _mm256_slli_epi16(d, 6)));
}

inline void fused_base(const Shape &s, const BaseView &v, const Activation &a,
                       const float *scales, float *output, int begin, int end,
                       int prefetch_blocks, std::int32_t *verified_dots = nullptr) {
    int r = begin;
    for (; r + 32 <= end; r += 32) {
        __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        for (int g = 0; g < s.groups(); ++g) {
            const auto ix = std::size_t(g) * s.rows + r;
            __m256i dot0 = _mm256_setzero_si256(), dot1 = dot0;
            const int first = g * s.blocks_per_group(), last = first + s.blocks_per_group();
            for (int b = first; b < last; ++b) {
                if (prefetch_blocks > 0 && b + prefetch_blocks < s.blocks()) {
                    for (int d = 0; d < 4; ++d) {
                        const auto *p = v.table + a.offsets[(b + prefetch_blocks) * 4 + d] + r;
                        _mm_prefetch(reinterpret_cast<const char *>(p), _MM_HINT_T0);
                    }
                }
                const auto *p0 = v.table + a.offsets[b * 4] + r;
                const auto *p1 = v.table + a.offsets[b * 4 + 1] + r;
                const auto *p2 = v.table + a.offsets[b * 4 + 2] + r;
                const auto *p3 = v.table + a.offsets[b * 4 + 3] + r;
                dot0 = _mm256_add_epi16(dot0, radix16(p0, p1, p2, p3));
                dot1 = _mm256_add_epi16(dot1, radix16(p0 + 16, p1 + 16, p2 + 16, p3 + 16));
            }
            // max unsigned group dot = 32*3*255 = 24480, so int16 is exact.
            dot0 = _mm256_sub_epi16(dot0, _mm256_slli_epi16(
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v.high_sum_gr + ix)), 7));
            dot1 = _mm256_sub_epi16(dot1, _mm256_slli_epi16(
                _mm256_loadu_si256(reinterpret_cast<const __m256i *>(v.high_sum_gr + ix + 16)), 7));
            __m256i dots[4] = {
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dot0)),
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dot0, 1)),
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dot1)),
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dot1, 1))
            };
            if (verified_dots) {
                alignas(32) int lanes[32];
                for (int j = 0; j < 4; ++j)
                    _mm256_store_si256(reinterpret_cast<__m256i *>(lanes + j * 8), dots[j]);
                for (int j = 0; j < 32; ++j)
                    verified_dots[std::size_t(r + j) * s.groups() + g] = lanes[j];
            }
            const auto scale = _mm256_set1_ps(scales[g]);
            const auto sum = _mm256_set1_ps(float(a.sums[g]));
            __m256 contributions[4];
            for (int j = 0; j < 4; ++j) {
                auto high = _mm256_cvtepi32_ps(_mm256_slli_epi32(dots[j], 2));
                auto value = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(v.alpha_gr + ix + 8 * j), high),
                                           _mm256_mul_ps(_mm256_loadu_ps(v.beta_gr + ix + 8 * j), sum));
                contributions[j] = _mm256_mul_ps(scale, value);
            }
            acc0 = _mm256_add_ps(acc0, contributions[0]);
            acc1 = _mm256_add_ps(acc1, contributions[1]);
            acc2 = _mm256_add_ps(acc2, contributions[2]);
            acc3 = _mm256_add_ps(acc3, contributions[3]);
        }
        _mm256_storeu_ps(output + r, acc0);
        _mm256_storeu_ps(output + r + 8, acc1);
        _mm256_storeu_ps(output + r + 16, acc2);
        _mm256_storeu_ps(output + r + 24, acc3);
    }
    for (; r < end; ++r) {
        float value = 0;
        for (int g = 0; g < s.groups(); ++g) {
            int dot = 0;
            for (int b = g * s.blocks_per_group(); b < (g + 1) * s.blocks_per_group(); ++b)
                for (int d = 0; d < 4; ++d) dot += int(v.table[a.offsets[b * 4 + d] + r]) << (2 * d);
            const auto ix = std::size_t(g) * s.rows + r;
            dot -= 128 * v.high_sum_gr[ix];
            if (verified_dots) verified_dots[std::size_t(r) * s.groups() + g] = dot;
            value += scales[g] * (v.alpha_gr[ix] * float(4 * dot) + v.beta_gr[ix] * float(a.sums[g]));
        }
        output[r] = value;
    }
}

inline int sum8(__m256i v) {
    auto sum = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

// Q4 is repacked at cold start: 16 low nibbles followed by 16 high nibbles.
// This is an optimized local baseline, not the ggml Q4_K kernel.
inline void direct_q4(const Shape &s, const std::uint8_t *packed,
                      const float *alpha_rg, const float *beta_rg,
                      const std::int8_t *z, const float *scales, const Activation &a,
                      float *output, int begin, int end, int *verified_dots = nullptr) {
    const auto mask = _mm_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    for (int r = begin; r < end; ++r) {
        float result = 0;
        for (int g = 0; g < s.groups(); ++g) {
            const auto ix = std::size_t(r) * s.groups() + g;
            const auto bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(packed + ix * 16));
            auto q = _mm256_castsi128_si256(_mm_and_si128(bytes, mask));
            q = _mm256_inserti128_si256(q, _mm_and_si128(_mm_srli_epi16(bytes, 4), mask), 1);
            auto vz = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(z + g * 32));
            const int dot = sum8(_mm256_madd_epi16(_mm256_maddubs_epi16(q, vz), ones));
            if (verified_dots) verified_dots[ix] = dot;
            result += scales[g] * (alpha_rg[ix] * float(dot) + beta_rg[ix] * float(a.sums[g]));
        }
        output[r] = result;
    }
}

inline void row_range(const Shape &s, int id, int count, int &begin, int &end) {
    const int chunks = (s.rows + 31) / 32;
    begin = std::min(s.rows, 32 * (chunks * id / count));
    end = std::min(s.rows, 32 * (chunks * (id + 1) / count));
}

} // namespace residual_mesh
