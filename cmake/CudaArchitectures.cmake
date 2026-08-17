include_guard(GLOBAL)

# Extends CMake's legacy FindCUDA architecture selector with support for
# multi-digit compute capabilities such as Blackwell's 12.0.
function(ct2_select_nvcc_arch_flags out_variable)
  set(arch_list ${ARGN})

  if(NOT arch_list)
    set(arch_list "Auto")
  endif()

  if("${arch_list}" STREQUAL "Auto")
    if(CUDA_VERSION VERSION_LESS "12.8")
      cuda_select_nvcc_arch_flags(legacy_flags Auto)
      set(${out_variable} ${legacy_flags} PARENT_SCOPE)
      set(${out_variable}_readable ${legacy_flags_readable} PARENT_SCOPE)
      return()
    endif()

    # CUDA_DETECT_INSTALLED_GPUS stores the unfiltered result in
    # CUDA_GPU_DETECT_OUTPUT. CMake 3.22 otherwise replaces 12.0 with its
    # highest known architecture, 8.6+PTX.
    CUDA_DETECT_INSTALLED_GPUS(unused_detected_archs)
    if(CUDA_GPU_DETECT_OUTPUT)
      set(detected_archs "${CUDA_GPU_DETECT_OUTPUT}")
      separate_arguments(detected_archs)
      set(arch_list ${detected_archs})
      message(STATUS "Autodetected CUDA architecture(s): ${arch_list}")
    else()
      set(arch_list ${CUDA_COMMON_GPU_ARCHITECTURES} "12.0+PTX")
      message(STATUS "Automatic GPU detection failed. Building for common architectures including 12.0.")
    endif()
  endif()

  list(REMOVE_DUPLICATES arch_list)

  set(nvcc_flags)
  set(nvcc_archs_readable)
  set(legacy_archs)

  foreach(arch_spec IN LISTS arch_list)
    set(arch_name "${arch_spec}")
    set(add_ptx FALSE)
    if(arch_name MATCHES "^(.*)\\+PTX$")
      set(add_ptx TRUE)
      set(arch_name "${CMAKE_MATCH_1}")
    endif()

    if(arch_name MATCHES "^([0-9]+)\\.([0-9]+)(\\(([0-9]+)\\.([0-9]+)\\))?$")
      set(binary_arch "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
      if(NOT "${CMAKE_MATCH_4}" STREQUAL "")
        set(virtual_arch "${CMAKE_MATCH_4}${CMAKE_MATCH_5}")
      else()
        set(virtual_arch "${binary_arch}")
      endif()

      if(binary_arch STREQUAL "120" AND CUDA_VERSION VERSION_LESS "12.8")
        message(FATAL_ERROR "CUDA architecture 12.0 requires CUDA Toolkit 12.8 or newer (found ${CUDA_VERSION})")
      endif()

      list(APPEND nvcc_flags
        -gencode arch=compute_${virtual_arch},code=sm_${binary_arch})
      list(APPEND nvcc_archs_readable sm_${binary_arch})
      if(add_ptx)
        list(APPEND nvcc_flags
          -gencode arch=compute_${virtual_arch},code=compute_${virtual_arch})
        list(APPEND nvcc_archs_readable compute_${virtual_arch})
      endif()
    else()
      list(APPEND legacy_archs "${arch_spec}")
    endif()
  endforeach()

  if(legacy_archs)
    cuda_select_nvcc_arch_flags(legacy_flags ${legacy_archs})
    list(APPEND nvcc_flags ${legacy_flags})
    if(legacy_flags_readable)
      separate_arguments(legacy_archs_readable UNIX_COMMAND "${legacy_flags_readable}")
      list(APPEND nvcc_archs_readable ${legacy_archs_readable})
    endif()
  endif()

  list(REMOVE_DUPLICATES nvcc_archs_readable)
  string(REPLACE ";" " " nvcc_archs_readable "${nvcc_archs_readable}")

  set(${out_variable} ${nvcc_flags} PARENT_SCOPE)
  set(${out_variable}_readable ${nvcc_archs_readable} PARENT_SCOPE)
endfunction()
