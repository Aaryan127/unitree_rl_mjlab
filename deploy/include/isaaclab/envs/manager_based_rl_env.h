// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include <cmath>
#include <set>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        const auto n = robot->data.joint_ids_map.size();
        if (!std::isfinite(step_dt) || step_dt <= 0 || n == 0 ||
            robot->data.default_joint_pos.size() != n ||
            robot->data.joint_stiffness.size() < n ||
            robot->data.joint_damping.size() != robot->data.joint_stiffness.size())
            throw std::runtime_error("Invalid deployment timing or joint-vector dimensions");
        std::set<int> ids;
        for (size_t i = 0; i < n; ++i) {
            const float id = robot->data.joint_ids_map[i];
            if (!std::isfinite(id) || id < 0 || id >= robot->data.joint_stiffness.size() || std::floor(id) != id ||
                !ids.insert(static_cast<int>(id)).second)
                throw std::runtime_error("Invalid or duplicate SDK joint index");
            if (!std::isfinite(robot->data.default_joint_pos[i]) ||
                !std::isfinite(robot->data.joint_stiffness[i]) || robot->data.joint_stiffness[i] < 0 ||
                !std::isfinite(robot->data.joint_damping[i]) || robot->data.joint_damping[i] < 0)
                throw std::runtime_error("Invalid deployment pose or PD gain");
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        if (alg) alg->reset();  // e.g. clear the recurrent state of the PGTT student
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        robot->update();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);
        episode_length += 1;
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
};

};
