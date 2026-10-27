// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <atomic>
#include <chrono>
#include <mutex>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    ~State_RLBase() {
        policy_thread_running = false;
        if (policy_thread.joinable()) policy_thread.join();
    }
    
    void enter()
    {
        // A timed-out worker must never block the 1 kHz FSM's move to Passive.
        // State objects stay alive in CtrlFSM while a retiring worker finishes.
        if (policy_thread.joinable()) {
            if (!worker_stopped) {
                policy_failed = true;
                spdlog::error("Previous policy worker has not stopped; refusing re-entry");
                return;
            }
            policy_thread.join();
        }
        policy_failed = false;
        unsafe_orientation = false;
        try {
            env->reset();
            std::lock_guard<std::mutex> lock(target_mutex);
            targets = env->action_manager->processed_actions();
        } catch (const std::exception& error) {
            policy_failed = true;
            spdlog::error("Policy entry failed: {}", error.what());
            return;
        }
        // Set gains only after a valid initial state/target has been read.
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }
        last_policy_tick = now_ns();
        // Start policy thread
        policy_thread_running = true;
        worker_stopped = false;
        policy_thread = std::thread([this]{
            using clock = std::chrono::steady_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            while (policy_thread_running)
            {
                try {
                    env->step();
                    unsafe_orientation = isaaclab::mdp::bad_orientation(env.get(), 1.0f);
                    {
                        std::lock_guard<std::mutex> lock(target_mutex);
                        targets = env->action_manager->processed_actions();
                    }
                    last_policy_tick = now_ns();
                } catch (const std::exception& error) {
                    spdlog::error("Policy inference failed: {}", error.what());
                    policy_failed = true;
                    break;
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
            worker_stopped = true;
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable() && worker_stopped) {
            policy_thread.join();
        }
    }

private:
    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    bool policy_fault() const {
        return policy_failed || unsafe_orientation ||
            (policy_thread_running && now_ns() - last_policy_tick > 100000000);
    }
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
    std::atomic<bool> worker_stopped{true};
    std::atomic<bool> policy_failed{false};
    std::atomic<bool> unsafe_orientation{false};
    std::atomic<int64_t> last_policy_tick{0};
    std::mutex target_mutex;
    std::vector<float> targets;
};

REGISTER_FSM(State_RLBase)
