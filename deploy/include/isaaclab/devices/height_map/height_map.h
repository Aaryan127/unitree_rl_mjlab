// Robot-centric height map for the PGTT `height_scan` observation.
//
// Layout (must match training, mjlab/tasks/velocity/mdp/observations.py::height_scan):
//   - grid of size_x by size_y meters (deploy.yaml params), `resolution` spacing,
//     centered on the base, rotated with the base yaw (x forward, y left)
//   - order: y outer loop (from -size_y/2 to +size_y/2), x inner loop (-size_x/2 to +size_x/2)
//   - value = (base_z - terrain_z) - base_height, clipped to [-1, 1]
//     (0 = ground where it is when standing on flat ground)
//
// Producer: any process that builds the map (e.g. elevation mapping from the Go2 LiDAR)
// sends it as one UDP datagram of num_points float32 values (little endian) to
// udp_port on the machine running the controller, at >= 10 Hz.
// See scripts/send_height_map.py for a test sender.
//
// Consumer: the `height_scan` observation. If no map was received, the map has the wrong
// size, or it is older than `timeout` seconds, all zeros are returned. The student was
// trained with missing/blind height maps reported as zeros, so it keeps walking
// (as a blind policy) instead of reacting to stale terrain.

#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace isaaclab
{

class HeightMap
{
public:
    static HeightMap& instance()
    {
        static HeightMap map;
        return map;
    }

    void set(const std::vector<float>& values)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        values_ = values;
        stamp_ = std::chrono::steady_clock::now();
        received_ = true;
    }

    // Latest map if it is valid and fresh, otherwise n zeros.
    std::vector<float> get(size_t n, float timeout_s)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        const float age = std::chrono::duration<float>(std::chrono::steady_clock::now() - stamp_).count();
        const bool ok = received_ && values_.size() == n && age <= timeout_s;
        if (ok != last_ok_) {
            if (ok) std::cout << "[HeightMap] receiving height map (" << n << " points)." << std::endl;
            else    std::cout << "[HeightMap] no valid height map (received=" << received_ << ", size="
                              << values_.size() << "/" << n << ", age=" << age << "s): using zeros." << std::endl;
            last_ok_ = ok;
        }
        return ok ? values_ : std::vector<float>(n, 0.0f);
    }

    // Start the UDP receiver once (port <= 0 disables it).
    void start_udp(int port)
    {
        if (port <= 0 || running_.exchange(true)) return;
        thread_ = std::thread([this, port] { receive_loop(port); });
        std::cout << "[HeightMap] listening for height maps on UDP port " << port << std::endl;
    }

    ~HeightMap()
    {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

private:
    HeightMap() = default;

    void receive_loop(int port)
    {
        int sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) { std::cerr << "[HeightMap] socket() failed" << std::endl; return; }
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::cerr << "[HeightMap] bind() to port " << port << " failed" << std::endl;
            close(sock);
            return;
        }
        timeval tv{0, 100000};  // 100 ms, so the thread notices shutdown
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        std::vector<char> buffer(65536);
        while (running_) {
            const ssize_t bytes = recv(sock, buffer.data(), buffer.size(), 0);
            if (bytes <= 0 || bytes % sizeof(float) != 0) continue;
            std::vector<float> values(bytes / sizeof(float));
            std::memcpy(values.data(), buffer.data(), bytes);
            set(values);
        }
        close(sock);
    }

    std::mutex mtx_;
    std::vector<float> values_;
    std::chrono::steady_clock::time_point stamp_{};
    bool received_ = false;
    bool last_ok_ = true;  // so the first failure is reported
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}  // namespace isaaclab
