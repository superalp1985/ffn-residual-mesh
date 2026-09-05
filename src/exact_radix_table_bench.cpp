#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: bench table.bin states.u16 rows blocks groups blocks_per_group state_count repeats\n";
        return 2;
    }
    const std::string table_path = argv[1];
    const std::string states_path = argv[2];
    const int rows = std::stoi(argv[3]);
    const int blocks = std::stoi(argv[4]);
    const int groups = std::stoi(argv[5]);
    const int blocks_per_group = std::stoi(argv[6]);
    const int state_count = std::stoi(argv[7]);
    const int repeats = std::stoi(argv[8]);
    const std::size_t table_bytes = static_cast<std::size_t>(rows) * blocks * state_count;
    std::vector<std::uint8_t> table(table_bytes);
    std::ifstream table_file(table_path, std::ios::binary);
    table_file.read(reinterpret_cast<char*>(table.data()), static_cast<std::streamsize>(table.size()));
    if (table_file.gcount() != static_cast<std::streamsize>(table.size())) {
        std::cerr << "table read failed\n";
        return 3;
    }
    std::vector<std::uint16_t> states(static_cast<std::size_t>(4) * blocks);
    std::ifstream states_file(states_path, std::ios::binary);
    states_file.read(reinterpret_cast<char*>(states.data()), static_cast<std::streamsize>(states.size() * sizeof(std::uint16_t)));
    if (states_file.gcount() != static_cast<std::streamsize>(states.size() * sizeof(std::uint16_t))) {
        std::cerr << "states read failed\n";
        return 4;
    }

    std::vector<std::int32_t> group_dot(static_cast<std::size_t>(groups) * rows);
    std::uint64_t checksum = 0;
    double best_ms = 1e30;
    double total_ms = 0.0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        std::fill(group_dot.begin(), group_dot.end(), 0);
        const auto start = std::chrono::steady_clock::now();
        for (int digit = 0; digit < 4; ++digit) {
            const int radix = 1 << (2 * digit);
            for (int block = 0; block < blocks; ++block) {
                const std::uint16_t state = states[static_cast<std::size_t>(digit) * blocks + block];
                const std::uint8_t* entry = table.data() + (static_cast<std::size_t>(block) * state_count + state) * rows;
                const int group = block / blocks_per_group;
                std::int32_t* output = group_dot.data() + static_cast<std::size_t>(group) * rows;
                for (int row = 0; row < rows; ++row) {
                    output[row] += radix * static_cast<std::int32_t>(entry[row]);
                }
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        best_ms = std::min(best_ms, elapsed_ms);
        total_ms += elapsed_ms;
        checksum += static_cast<std::uint64_t>(group_dot[repeat % group_dot.size()]);
    }
    std::cout << "{\"median_like_ms\":" << (total_ms / repeats)
              << ",\"best_ms\":" << best_ms
              << ",\"checksum\":" << checksum
              << ",\"logical_table_read_bytes\":" << (static_cast<std::size_t>(4) * blocks * rows)
              << "}\n";
    return 0;
}
