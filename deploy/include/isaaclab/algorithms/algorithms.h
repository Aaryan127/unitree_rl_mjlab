// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaaclab
{

class Algorithms
{
public:
    virtual ~Algorithms() = default;
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    // Clear internal state (e.g. the hidden state of a recurrent policy).
    virtual void reset() {}

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

// Runs an ONNX policy.
//
// Feed-forward policies: inputs are taken from the observation map by name
// (a single "obs" input for a flat observation list in deploy.yaml).
//
// Recurrent policies (e.g. the PGTT student exported by scripts/export_deploy.py):
// every input "<name>_in" that has a matching output "<name>_out" is internal state.
// It starts at zero, is fed back from the previous step's output, and is cleared by
// reset(). The action is the output named "actions" (or the first output).
class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            auto shape = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            for (auto& dim : shape) if (dim < 0) dim = 1;  // dynamic batch -> 1
            input_shapes.push_back(shape);
            input_sizes.push_back(numel(shape));
            input_names_str.push_back(session->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            auto shape = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            for (auto& dim : shape) if (dim < 0) dim = 1;
            output_shapes.push_back(shape);
            output_names_str.push_back(session->GetOutputNameAllocated(i, allocator).get());
        }
        for (auto& s : input_names_str) input_names.push_back(s.c_str());
        for (auto& s : output_names_str) output_names.push_back(s.c_str());

        // Action output.
        action_output = 0;
        for (size_t i = 0; i < output_names_str.size(); ++i) {
            if (output_names_str[i] == "actions") action_output = i;
        }
        action.resize(numel(output_shapes[action_output]));

        // Recurrent state pairs: "<name>_in" <-> "<name>_out".
        for (size_t i = 0; i < input_names_str.size(); ++i) {
            const auto& in = input_names_str[i];
            if (in.size() < 3 || in.compare(in.size() - 3, 3, "_in") != 0) continue;
            const auto out = in.substr(0, in.size() - 3) + "_out";
            for (size_t j = 0; j < output_names_str.size(); ++j) {
                if (output_names_str[j] == out) {
                    if (numel(output_shapes[j]) != input_sizes[i])
                        throw std::runtime_error("Recurrent ONNX state sizes do not match: " + in);
                    state_pairs.push_back({i, j});
                    states[i] = std::vector<float>(input_sizes[i], 0.0f);
                }
            }
        }
        std::cout << "[OrtRunner] " << model_path << ": " << input_names.size() << " inputs, "
                  << output_names.size() << " outputs, " << state_pairs.size()
                  << " recurrent state(s), " << action.size() << " actions" << std::endl;
    }

    void reset() override
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        for (auto& [idx, state] : states) std::fill(state.begin(), state.end(), 0.0f);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::vector<Ort::Value> input_tensors;
        for (size_t i = 0; i < input_names.size(); ++i)
        {
            std::vector<float>* data = nullptr;
            if (states.count(i)) {
                data = &states.at(i);
            } else {
                auto it = obs.find(input_names_str[i]);
                if (it == obs.end()) {
                    throw std::runtime_error("Input name " + input_names_str[i] + " not found in observations.");
                }
                data = &it->second;
            }
            if (static_cast<int64_t>(data->size()) != input_sizes[i]) {
                throw std::runtime_error("Input '" + input_names_str[i] + "' has " + std::to_string(data->size())
                    + " values, the model expects " + std::to_string(input_sizes[i])
                    + ". Check the observation list in deploy.yaml.");
            }
            for (float value : *data)
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite ONNX input");
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, data->data(), data->size(), input_shapes[i].data(), input_shapes[i].size()));
        }

        auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
                                    input_tensors.size(), output_names.data(), output_names.size());

        const float* act_data = outputs[action_output].GetTensorMutableData<float>();
        for (size_t i = 0; i < action.size(); ++i)
            if (!std::isfinite(act_data[i])) throw std::runtime_error("Non-finite ONNX action");
        std::memcpy(action.data(), act_data, action.size() * sizeof(float));
        for (const auto& [in_idx, out_idx] : state_pairs) {
            const float* h = outputs[out_idx].GetTensorMutableData<float>();
            std::memcpy(states.at(in_idx).data(), h, states.at(in_idx).size() * sizeof(float));
        }
        return action;
    }

private:
    static int64_t numel(const std::vector<int64_t>& shape)
    {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<std::vector<int64_t>> output_shapes;

    size_t action_output = 0;
    std::vector<std::pair<size_t, size_t>> state_pairs;   // (input index, output index)
    std::unordered_map<size_t, std::vector<float>> states; // input index -> state buffer
};
};
