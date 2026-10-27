// Exercise the actual C++ observation/action/ONNX path without connecting to DDS.
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <filesystem>

class MockRobot : public isaaclab::Articulation {
public:
    void update() override {
        data.root_ang_vel_b = Eigen::Vector3f(1, 2, 3);
        data.projected_gravity_b = Eigen::Vector3f(0, 0, -1);
        data.root_quat_w = Eigen::Quaternionf::Identity();
        data.velocity_axes = {0.3f, -0.2f, 0.1f};
        for (int i = 0; i < data.joint_pos.size(); ++i) {
            data.joint_pos[i] = data.default_joint_pos[i] + 0.01f * i;
            data.joint_vel[i] = 0.1f * i;
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::filesystem::path bundle(argv[1]);
    auto cfg = YAML::LoadFile((bundle / "params/deploy.yaml").string());
    cfg["observations"]["height_scan"]["params"]["udp_port"] = 0;
    auto robot = std::make_shared<MockRobot>();
    isaaclab::ManagerBasedRLEnv env(cfg, robot);
    env.alg = std::make_unique<isaaclab::OrtRunner>((bundle / "exported/policy.onnx").string());
    env.reset();
    const auto reset_targets = env.action_manager->processed_actions();
    for (int i = 0; i < 12; ++i)
        assert(std::abs(reset_targets[i] - cfg["actions"]["JointPositionAction"]["offset"][i].as<float>()) < 1e-6f);
    auto obs = env.observation_manager->compute().at("obs");
    assert(obs.size() == 153);
    assert(obs[0] == 1 && obs[1] == 2 && obs[2] == 3 && obs[5] == -1);
    assert(std::abs(obs[6] - 0.3f) < 1e-6f && std::abs(obs[7] + 0.2f) < 1e-6f);
    for (int i = 0; i < 12; ++i) {
        assert(std::abs(obs[9 + i] - 0.01f * i) < 1e-6f);
        assert(std::abs(obs[21 + i] - 0.1f * i) < 1e-6f);
        assert(obs[33 + i] == 0);
    }
    // Re-reading observations must not advance the gait phase.
    const auto again = env.observation_manager->compute().at("obs");
    for (int i = 45; i < 54; ++i) assert(obs[i] == again[i]);
    env.step();
    obs = env.observation_manager->compute().at("obs");
    const float phase = 2 * static_cast<float>(M_PI) * 2.0f * env.step_dt;
    assert(std::abs(obs[45] - std::cos(phase)) < 1e-6f);
    assert(std::abs(obs[49] - std::sin(phase)) < 1e-6f);
    assert(!isaaclab::mdp::bad_orientation(&env, 1.0f));
    robot->data.projected_gravity_b[2] = 1;
    assert(isaaclab::mdp::bad_orientation(&env, 1.0f));
    robot->data.projected_gravity_b[2] = std::numeric_limits<float>::quiet_NaN();
    assert(isaaclab::mdp::bad_orientation(&env, 1.0f));
    const auto clip = cfg["actions"]["JointPositionAction"]["clip"];
    assert(!clip.IsNull());  // audit bundle must use matching sim/deploy joint limits
    env.action_manager->process_action(std::vector<float>(12, 1e6f));
    const auto targets = env.action_manager->processed_actions();
    for (int i = 0; i < 12; ++i) assert(std::abs(targets[i] - clip[i][1].as<float>()) < 1e-5f);
    std::cout << "C++ observation order, gait clock, target reset/limits and fall checks passed\n";
}
