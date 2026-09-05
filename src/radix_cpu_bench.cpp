#include "radix_cpu_kernels.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace residual_mesh;

template <typename T>
std::vector<T> read_file(const std::string &path, std::size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<T> out(count);
    if (count && !f.read(reinterpret_cast<char *>(out.data()), std::streamsize(count * sizeof(T))))
        throw std::runtime_error("cannot read " + path);
    return out;
}

template <typename T>
std::vector<T> transpose(const std::vector<T> &v, const Shape &s) {
    std::vector<T> out(v.size());
    for (int r = 0; r < s.rows; ++r)
        for (int g = 0; g < s.groups(); ++g) out[std::size_t(g) * s.rows + r] = v[std::size_t(r) * s.groups() + g];
    return out;
}

struct Method {
    std::string name;
    int kind, prefetch;
    std::vector<double> samples;
    double checksum = 0;
    double max_error = 0;
};

int main(int argc, char **argv) {
    try {
        if (argc != 9) throw std::invalid_argument("usage: bench DIR ROWS HIDDEN BLOCK TOKENS REPEATS THREADS PREFETCH_LIST");
        const std::string dir = argv[1];
        Shape s{std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4])};
        s.validate();
        const int tokens = std::stoi(argv[5]), repeats = std::stoi(argv[6]), threads = std::stoi(argv[7]);
        if (tokens < 1 || repeats < 1 || threads < 1 || threads > 256)
            throw std::invalid_argument("positive tokens/repeats and threads in [1,256] required");
        std::vector<Method> methods{{"direct_q4_avx2", 0, 0, {}}, {"legacy_table", 1, 0, {}}};
        std::stringstream list(argv[8]);
        std::string item;
        while (std::getline(list, item, ',')) {
            const int distance = std::stoi(item);
            if (distance < 0 || distance > 64) throw std::invalid_argument("prefetch distance out of range");
            methods.push_back({"fused_avx2_prefetch_" + item, 2, distance, {}});
        }
        auto table = read_file<std::uint8_t>(dir + "/table.bin", std::size_t(s.blocks()) * s.states() * s.rows);
        auto q = read_file<std::uint8_t>(dir + "/q.bin", std::size_t(s.rows) * s.hidden);
        if (*std::max_element(q.begin(), q.end()) > 15) throw std::invalid_argument("Q4 codes exceed 15");
        auto high_sum = read_file<std::int16_t>(dir + "/high_sum.bin", std::size_t(s.rows) * s.groups());
        auto alpha = read_file<float>(dir + "/alpha.bin", high_sum.size());
        auto beta = read_file<float>(dir + "/beta.bin", high_sum.size());
        auto z = read_file<std::int8_t>(dir + "/z.bin", std::size_t(tokens) * s.hidden);
        auto scales = read_file<float>(dir + "/scale.bin", std::size_t(tokens) * s.groups());
        auto high_gr = transpose(high_sum, s);
        auto alpha_gr = transpose(alpha, s), beta_gr = transpose(beta, s);
        std::vector<std::uint8_t> packed(q.size() / 2);
        for (std::size_t g = 0; g < high_sum.size(); ++g)
            for (int i = 0; i < 16; ++i) packed[g * 16 + i] = q[g * 32 + i] | (q[g * 32 + i + 16] << 4);
        BaseView view{table.data(), high_gr.data(), alpha_gr.data(), beta_gr.data()};
        CpuPool pool(threads);
        Activation a;
        a.resize(s);
        std::vector<float> output(s.rows), high_ref(s.rows), full_ref(s.rows), low_ref(s.rows);
        std::vector<int> scratch(high_sum.size()), check_dots(high_sum.size()), high_dots(high_sum.size()), full_dots(high_sum.size());
        auto evaluate = [&](const Method &m, int token, bool verify) {
            a.prepare(s, z.data() + std::size_t(token) * s.hidden, m.kind != 0);
            pool.run([&](int id) {
                int begin, end;
                row_range(s, id, threads, begin, end);
                int *dots = verify ? check_dots.data() : nullptr;
                if (m.kind == 0)
                    direct_q4(s, packed.data(), alpha.data(), beta.data(), z.data() + std::size_t(token) * s.hidden,
                              scales.data() + std::size_t(token) * s.groups(), a, output.data(), begin, end, dots);
                else if (m.kind == 1)
                    legacy_base(s, view, a, scales.data() + std::size_t(token) * s.groups(),
                                output.data(), scratch.data(), begin, end, dots);
                else
                    fused_base(s, view, a, scales.data() + std::size_t(token) * s.groups(),
                               output.data(), begin, end, m.prefetch, dots);
            });
        };
        std::size_t integer_mismatches = 0;
        double max_scaled_error = 0, max_merge_error = 0;
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < s.rows; ++r) {
                float vh = 0, vf = 0, vl = 0;
                for (int g = 0; g < s.groups(); ++g) {
                    const auto ix = std::size_t(r) * s.groups() + g;
                    int dh = 0, df = 0, dl = 0, zs = 0;
                    for (int i = 0; i < 32; ++i) {
                        const int code = q[ix * 32 + i], activation = z[std::size_t(t) * s.hidden + g * 32 + i];
                        dh += (code >> 2) * activation;
                        dl += (code & 3) * activation;
                        df += code * activation;
                        zs += activation;
                    }
                    high_dots[ix] = dh; full_dots[ix] = df;
                    const float scale = scales[std::size_t(t) * s.groups() + g];
                    vh += scale * (alpha[ix] * float(4 * dh) + beta[ix] * float(zs));
                    vf += scale * (alpha[ix] * float(df) + beta[ix] * float(zs));
                    vl += scale * (alpha[ix] * float(dl));
                }
                high_ref[r] = vh; full_ref[r] = vf; low_ref[r] = vl;
            }
            for (auto &m : methods) {
                evaluate(m, t, true);
                const auto &ref_dots = m.kind == 0 ? full_dots : high_dots;
                const auto &ref = m.kind == 0 ? full_ref : high_ref;
                for (std::size_t i = 0; i < check_dots.size(); ++i)
                    if (check_dots[i] != ref_dots[i]) ++integer_mismatches;
                for (int r = 0; r < s.rows; ++r) {
                    m.max_error = std::max(m.max_error, double(std::abs(output[r] - ref[r])));
                    if (m.kind != 0) max_merge_error = std::max(max_merge_error, double(std::abs(output[r] + low_ref[r] - full_ref[r])));
                }
                max_scaled_error = std::max(max_scaled_error, m.max_error);
            }
        }
        // Interleave methods to reduce order/temperature bias, consume every output.
        for (int r = -3; r < repeats; ++r) {
            const int token = (r + 3) % tokens;
            for (std::size_t k = 0; k < methods.size(); ++k) {
                auto &m = methods[(k + r + 3) % methods.size()];
                const auto start = std::chrono::steady_clock::now();
                evaluate(m, token, false);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                if (r >= 0) {
                    m.samples.push_back(ms);
                    m.checksum += std::accumulate(output.begin(), output.end(), 0.0);
                }
            }
        }
        std::cout << std::setprecision(10) << "{\"integer_mismatches\":" << integer_mismatches
                  << ",\"max_scaled_abs_error\":" << max_scaled_error
                  << ",\"max_merge_abs_error\":" << max_merge_error
                  << ",\"tokens\":" << tokens << ",\"threads\":" << threads
                  << ",\"table_bytes\":" << table.size()
                  << ",\"selected_table_bytes\":" << std::size_t(4) * s.blocks() * s.rows
                  << ",\"packed_q4_bytes\":" << packed.size()
                  << ",\"methods\":[";
        for (std::size_t k = 0; k < methods.size(); ++k) {
            const auto &m = methods[k];
            auto sorted = m.samples;
            std::sort(sorted.begin(), sorted.end());
            const double median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
            if (k) std::cout << ',';
            std::cout << "{\"name\":\"" << m.name << "\",\"scope\":\"" << (m.kind == 0 ? "full_projection" : "base_only")
                      << "\",\"median_ms\":" << median << ",\"min_ms\":" << sorted.front()
                      << ",\"p95_ms\":" << sorted[std::size_t(std::ceil(0.95 * sorted.size())) - 1]
                      << ",\"max_abs_error\":" << m.max_error << ",\"checksum\":" << m.checksum << ",\"samples_ms\":[";
            for (std::size_t i = 0; i < m.samples.size(); ++i) { if (i) std::cout << ','; std::cout << m.samples[i]; }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return integer_mismatches == 0 && std::isfinite(max_scaled_error) ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
