// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>

#include <gflags/gflags.h>

#include <openvino/core/partial_shape.hpp>
#include <openvino/openvino.hpp>

#include "openvino/op/subtract.hpp"
#include "openvino/op/constant.hpp"

#include "tools_helpers.hpp"

ov::AnyMap global_plugin_config = {};
std::unordered_set<std::string> available_devices = {};
std::string target_device = "";
std::string target_plugin_name = "";

template <typename T>
void inline fill_data_ptr_normal_random_float(T* data,
                                              size_t size,
                                              const float mean,
                                              const float stddev,
                                              const int seed = 1) {
    std::default_random_engine random(seed);
    std::normal_distribution<> normal_d{mean, stddev};
    for (size_t i = 0; i < size; i++) {
        auto value = static_cast<float>(normal_d(random));
        if (typeid(T) == typeid(typename ov::fundamental_type_for<ov::element::f16>)) {
            data[i] = static_cast<T>(ov::float16(value).to_bits());
        } else {
            data[i] = static_cast<T>(value);
        }
    }
}

ov::Tensor create_and_fill_tensor_normal_distribution(const ov::element::Type element_type,
                                                      const ov::Shape& shape,
                                                      const float mean,
                                                      const float stddev,
                                                      const int seed) {
    auto tensor = ov::Tensor{element_type, shape};
#define CASE(X)                                                                              \
    case X:                                                                                  \
        fill_data_ptr_normal_random_float(tensor.data<ov::element_type_traits<X>::value_type>(), \
                                          shape_size(shape),                                 \
                                          mean,                                              \
                                          stddev,                                            \
                                          seed);                                             \
        break;
    switch (element_type) {
        CASE(ov::element::Type_t::boolean)
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
    case ov::element::Type_t::u1:
    case ov::element::Type_t::i4:
    case ov::element::Type_t::u4:
        fill_data_ptr_normal_random_float(static_cast<uint8_t*>(tensor.data()),
                                          tensor.get_byte_size(),
                                          mean,
                                          stddev,
                                          seed);
        break;
    default:
        OPENVINO_THROW("Unsupported element type: ", element_type);
    }
#undef CASE
    return tensor;
}

void safePluginUnload(ov::Core& core, const std::string& deviceName) {
    try {
        core.unload_plugin(deviceName);
    } catch (const ov::Exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
}

///////////////////////////////////////
//
// Building models section
//
///////////////////////////////////////

std::shared_ptr<ov::Model> make_2_input_subtract(ov::Shape input_shape = {1, 3, 24, 24},
                                                 ov::element::Type type = ov::element::f32);

std::shared_ptr<ov::Model> make_2_input_subtract(ov::Shape input_shape, ov::element::Type type) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    auto param1 = std::make_shared<ov::op::v0::Parameter>(type, input_shape);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(param0, param1);
    auto result = std::make_shared<ov::op::v0::Result>(subtract);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0, param1});
    model->set_friendly_name("TwoInputSubtract");
    return model;
}

std::vector<std::shared_ptr<ov::Model>> models;
void SetupNetworks() {
    models.emplace_back(make_2_input_subtract());
    models.emplace_back(make_2_input_subtract());
    models.emplace_back(make_2_input_subtract());
    models.emplace_back(make_2_input_subtract());
    models.emplace_back(make_2_input_subtract());
}

///////////////////////////////////////
//
/// @brief Test 1 CoreThreadingTestsWithCacheEnabled smoke_compiled_model_cache_enabled
//
// const Params paramsStreamsDRIVER[] = {
//     std::tuple<Device, Config>{
//         ov::test::utils::DEVICE_NPU,
//         {{ov::num_streams(ov::streams::AUTO), 
//         ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::DRIVER)}}},
//   };

void CoreThreadingTestsWithCacheEnabled () {
    ov::Core core;
    std::cout << "core created!" << std::endl;

    unsigned int threadsNum = 20;
    unsigned int iterations = 10;

    try {
        core.set_property(ov::enable_profiling(true));
        std::cout << "set_property ov::enable_profiling" << std::endl;

        // still fails silently when NUM_STREAMS=AUTO
        // core.set_property(ov::num_streams(ov::streams::AUTO));
        // std::cout << "set_property ov::num_streams AUTO" << std::endl;

        core.set_property(ov::cache_dir("C:\\Users\\sbutnari\\ov_cache"));
        std::cout << "set_property ov::cache_dir" << std::endl;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
    }

    std::map<std::string, std::string> config;
    std::string key("NPU_COMPILER_TYPE");
    std::string config_value("DRIVER");
    config[key] = std::move(config_value);


    std::atomic<unsigned int> counter{0u};
    std::vector<std::thread> threads(threadsNum);

    // Iterates through each model
    for (auto& model : models) {

        for (auto& thread : threads) {
            thread = std::thread([&]() {
                for (unsigned int i = 0; i < iterations; ++i) {
                    auto value = counter++;
                    std::cout << "value " << value <<  std::endl;
                    try {
                        (void)core.compile_model(model, "NPU", {config.begin(), config.end()});
                    }
                    catch (const std::exception& error) {
                        std::cerr << error.what() << std::endl;
                    }
                }});
        }
        for (auto& thread : threads) {
            if (thread.joinable())
                thread.join();
        }
    }
    core.set_property(ov::cache_dir(""));
}

///////////////////////////////////////
//
/// @brief Test 2 CoreThreadingTestsWithIter smoke_CompileModel
// 
// const Params params[] = {
//     std::tuple<Device, Config>{ov::test::utils::DEVICE_NPU, {{ov::enable_profiling(false)}}},
//     std::tuple<Device, Config>{ov::test::utils::DEVICE_HETERO,
//                                {{ov::device::priorities(ov::test::utils::DEVICE_NPU, ov::test::utils::DEVICE_CPU)}}},
// };

void CoreThreadingTestsWithIter () {
    ov::Core core;
    std::cout << "core created!" << std::endl;

    unsigned int threadsNum = 4;
    unsigned int iterations = 50;

    std::atomic<unsigned int> counter{0u};

    core.set_property(ov::enable_profiling(false));
    std::cout << "set_property ov::enable_profiling" << std::endl;

    std::vector<std::thread> threads(threadsNum);
    for (auto& thread : threads) {
        thread = std::thread([&]() {
            for (unsigned int i = 0; i < iterations; ++i) {
                auto value = counter++;
                std::cout << "value " << value <<  std::endl;
                try {
                    (void)core.compile_model(models[value % models.size()], "NPU");
                }
                catch (const std::exception& error) {
                    std::cerr << error.what() << std::endl;
                }
            }});
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}

///////////////////////////////////////
//
/// @brief Test 3 CoreThreadingTestsWithIter smoke_CompileModel_MultipleCores
// 
// const Params params[] = {
//     std::tuple<Device, Config>{ov::test::utils::DEVICE_NPU, {{ov::enable_profiling(false)}}},
//     std::tuple<Device, Config>{ov::test::utils::DEVICE_HETERO,
//                                {{ov::device::priorities(ov::test::utils::DEVICE_NPU, ov::test::utils::DEVICE_CPU)}}},
// };

void CoreThreadingTestsWithIter_MultipleCores () {
    unsigned int threadsNum = 4;
    unsigned int iterations = 50;

    std::atomic<unsigned int> counter{0u};

    std::vector<std::thread> threads(threadsNum);
    for (auto& thread : threads) {
        thread = std::thread([&]() {
            for (unsigned int i = 0; i < iterations; ++i) {

                //creating new core at each iteration!
                ov::Core core;
                core.set_property(ov::enable_profiling(false));

                auto value = counter++;
                std::cout << "new core, value " << value <<  std::endl;
                try {
                    (void)core.compile_model(models[value % models.size()], "NPU");
                }
                catch (const std::exception& error) {
                    std::cerr << error.what() << std::endl;
                }
            }});
    }

    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}

///////////////////////////////////////
//
/// @brief Test 4 CoreThreadingTestsWithIter nightly_AsyncInfer_ShareInput
// 
// const Params params[] = {
//     std::tuple<Device, Config>{ov::test::utils::DEVICE_NPU, {{ov::enable_profiling(false)}}},
//     std::tuple<Device, Config>{ov::test::utils::DEVICE_HETERO,
//                                {{ov::device::priorities(ov::test::utils::DEVICE_NPU, ov::test::utils::DEVICE_CPU)}}},
// };

void CoreThreadingTestsWithIter_AsyncInfer_ShareInput () {
    unsigned int threadsNum = 4;
    unsigned int iterations = 50;

    auto model = models[0];
    std::map<ov::Output<ov::Node>, ov::Tensor> inputs;
    for (const auto& input : model->inputs()) {
        auto tensor = create_and_fill_tensor_normal_distribution(input.get_element_type(),
                                                                                  input.get_shape(),
                                                                                  0.0f,
                                                                                  0.2f,
                                                                                  7235346);
        std::cout << "Create input, type: " << input.get_element_type() << std::endl;
        inputs.insert({input, tensor});
    }
    std::cout << "Inputs created" << std::endl;

    std::atomic<unsigned int> counter{0u};
    std::vector<std::thread> threads(threadsNum);

    for (auto& thread : threads) {
        thread = std::thread([&]() {
            //creating new core at each iteration!
            ov::Core core;
            core.set_property(ov::enable_profiling(false));

            auto compiled_model = core.compile_model(model, "NPU");
            auto nireq = compiled_model.get_property(ov::optimal_number_of_infer_requests);

            std::cout << "InferRequest nireq: " << nireq << std::endl;
            int count = nireq;

            std::vector<ov::InferRequest> inferReqsQueue;
            while (count--) {
                ov::InferRequest req = compiled_model.create_infer_request();
                for (const auto& input : inputs) {
                    req.set_tensor(input.first, input.second);
                }
                inferReqsQueue.push_back(req);
            }
            for (auto& req : inferReqsQueue) {
                auto value = counter++;
                std::cout << "inferReq start_async value " << value <<  std::endl;
                req.start_async();
            }
            for (auto& req : inferReqsQueue) {
                std::cout << "inferReq waiting" << std::endl;
                req.wait();
                std::cout << "inferReq done" << std::endl;
            }});
    }
    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
    }
}

int main(int argc, char* argv[]) {
    try {
        const auto& version = ov::get_openvino_version();
        std::cout << version.description << " version ......... ";
        std::cout << OPENVINO_VERSION_MAJOR << "." << OPENVINO_VERSION_MINOR << "." << OPENVINO_VERSION_PATCH
                    << std::endl;
        std::cout << "Build ........... ";
        std::cout << version.buildNumber << std::endl;

        SetupNetworks();
        std::cout << "Networks created!" << std::endl;

        // CoreThreadingTestsWithCacheEnabled ();
        // CoreThreadingTestsWithIter ();
        // CoreThreadingTestsWithIter_MultipleCores ();
        CoreThreadingTestsWithIter_AsyncInfer_ShareInput ();

    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
