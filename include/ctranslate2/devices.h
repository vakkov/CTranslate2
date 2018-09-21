#pragma once

#include <stdexcept>
#include <string>

namespace ctranslate2 {

  enum class Device {
    CPU,
    CUDA
  };

  Device str_to_device(const std::string& device);
  std::string device_to_str(Device device);

#define UNSUPPORTED_DEVICE_CASE(DEVICE)                       \
  case DEVICE: {                                              \
    throw std::runtime_error("unsupported device " #DEVICE);  \
    break;                                                    \
  }

#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    const Device D = DEVICE;                    \
    STMT;                                       \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__
#ifndef WITH_CUDA
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#else
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#endif

}
