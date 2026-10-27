#include "isaaclab/algorithms/algorithms.h"
#include "isaaclab/devices/height_map/height_map.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <limits>

int main(int argc, char** argv) {
    assert(argc == 2);
    isaaclab::OrtRunner runner(argv[1]);
    std::cout << std::setprecision(9);
    for (int step = 0; step < 10; ++step) {
        if (step == 5) runner.reset();
        std::vector<float> obs(153);
        for (size_t j = 0; j < obs.size(); ++j) obs[j] = std::sin((step + j) * 0.1f);
        auto actions = runner.act({{"obs", obs}});
        assert(actions.size() == 12);
        std::cout << "ACTION";
        for (float value : actions) std::cout << " " << value;
        std::cout << "\n";
    }
    bool rejected = false;
    try { runner.act({{"obs", std::vector<float>(152)}}); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { runner.act({{"obs", std::vector<float>(153, std::numeric_limits<float>::quiet_NaN())}}); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    auto& map = isaaclab::HeightMap::instance();
    map.set({0.1f, -0.1f});
    assert(map.get(2, 1.0f)[0] == 0.1f);
    assert(map.get(3, 1.0f) == std::vector<float>(3));
    map.set({std::numeric_limits<float>::quiet_NaN()});
    assert(map.get(1, 1.0f)[0] == 0.0f);
    map.set({2.0f});
    assert(map.get(1, 1.0f)[0] == 0.0f);
}
