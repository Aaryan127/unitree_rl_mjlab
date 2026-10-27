cmake_minimum_required(VERSION 3.12)
project(go2_controller)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(unitree_sdk2 CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0")
else()
    message(FATAL_ERROR "No bundled ONNX Runtime for ${CMAKE_SYSTEM_PROCESSOR}")
endif()

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  include ${ONNXRUNTIME_DIR}/include ${PROJECT_SOURCE_DIR}/../../include)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC
  unitree_sdk2 Boost::program_options yaml-cpp Eigen3::Eigen
  spdlog::spdlog fmt::fmt Threads::Threads rt
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0)

add_executable(go2_ctrl main.cpp)
target_link_libraries(go2_ctrl PRIVATE ${PROJECT_NAME}_lib)

option(PGTT_BUILD_TESTS "Build offline deployment contract tests" OFF)
set(PGTT_TEST_POLICY_DIR "" CACHE PATH "Exported PGTT bundle for offline tests")
if(PGTT_BUILD_TESTS)
    enable_testing()
    add_executable(pgtt_contract_test ../../../tests/cpp/test_deploy_contract.cpp)
    target_link_libraries(pgtt_contract_test PRIVATE ${PROJECT_NAME}_lib)
    target_compile_options(pgtt_contract_test PRIVATE -UNDEBUG)
    if(PGTT_TEST_POLICY_DIR)
        add_test(NAME pgtt_deploy_contract COMMAND pgtt_contract_test ${PGTT_TEST_POLICY_DIR})
    endif()
endif()
