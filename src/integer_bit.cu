/* -*- C++ -*- */
/*    This file is part of CUDA_tests
      Copyright (C) 2016  Julien Thevenon ( julien_thevenon at yahoo.fr )

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with this program.  If not, see <http://www.gnu.org/licenses/>
*/
#include <iostream>
#include <iomanip>
#include <cinttypes>
#include <cstring>

#include "my_cuda.h"
#ifdef __NVCC__
#include "thrust/version.h"
#endif // __NVCC__

CUDA_KERNEL(cuda_kernel, uint32_t * p_int_ptr, uint32_t * p_nipples_ptr)
{
  int l_shift = threadIdx.x << 2;
  uint32_t l_mask = ((uint32_t)0xF) << l_shift;
  p_nipples_ptr[threadIdx.x] = (*p_int_ptr & l_mask) >> l_shift;
}

#ifdef __NVCC__
template <typename TYPE>
TYPE byte_to_kilo_byte(TYPE p_value_in_byte)
{
    return p_value_in_byte / 1024;
}

template <typename TYPE>
TYPE byte_to_mega_byte(TYPE p_value_in_byte)
{
    return byte_to_kilo_byte<TYPE>(p_value_in_byte) / 1024;
}

void display_GPU_info()
{
    std::cout << "CUDA version  : " << CUDART_VERSION << std::endl;
    std::cout << "THRUST version: " << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << "." << THRUST_SUBMINOR_VERSION << std::endl;

    int l_cuda_device_nb = 0;
    cudaError_t l_error = cudaGetDeviceCount(&l_cuda_device_nb);
    if(cudaSuccess != l_error)
    {
	    std::cout << "ERROR : " << cudaGetErrorName(l_error) << ":" << cudaGetErrorString(l_error) << std::endl;
	    exit(-1);
    }
    std::cout << "Number of CUDA devices: " << l_cuda_device_nb << std::endl;

    for(int l_device_index = 0; l_device_index < l_cuda_device_nb; ++l_device_index)
    {
        std::cout << "Cuda device[" << l_device_index << "]" << std::endl;
        cudaDeviceProp l_properties;
        cudaGetDeviceProperties(&l_properties, l_device_index);
        std::cout << R"(\tName                      : ")" << l_properties.name << R"(")" << std::endl;
        std::cout <<   "\tDevice compute capability : " << l_properties.major << "." << l_properties.minor << std::endl;
        std::cout <<    "\tGlobal memory             : " << byte_to_mega_byte(l_properties.totalGlobalMem) << "mb" << std::endl;
        std::cout <<    "\tShared memory             : " << byte_to_kilo_byte(l_properties.sharedMemPerBlock) << "kb" << std::endl;
        std::cout <<    "\tConstant memory           : " << byte_to_kilo_byte(l_properties.totalConstMem) << "kb" << std::endl;
        std::cout <<    "\tBlock registers           : " << l_properties.regsPerBlock << std::endl << std::endl;

        std::cout <<    "\tWarp size                 : " << l_properties.warpSize << std::endl;
        std::cout <<    "\tThreads per block         : " << l_properties.maxThreadsPerBlock << std::endl;
        std::cout <<    "\tMax block dimensions      : [ " << l_properties.maxThreadsDim[0] << ", " << l_properties.maxThreadsDim[1]  << ", " << l_properties.maxThreadsDim[2] << " ]" << std::endl;
        std::cout <<    "\tMax grid dimensions       : [ " << l_properties.maxGridSize[0] << ", " << l_properties.maxGridSize[1]  << ", " << l_properties.maxGridSize[2] << " ]" << std::endl;
        std::cout <<    "\tMultiprocessor count      : " << l_properties.multiProcessorCount << std::endl;
        std::cout <<    "\tConcurrent kernels        : " << l_properties.concurrentKernels << std::endl;
        std::cout << std::endl;

    }
}

#endif // __NVCC__

int launch_cuda_code(void)
{
#ifdef __NVCC__
  display_GPU_info();
#endif // __NVCC__

  uint32_t l_int = 0x87654321;
  uint32_t l_nipples[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  uint32_t * l_int_ptr;
  uint32_t * l_nipples_ptr;

  gpuErrChk(cudaMalloc(&l_int_ptr, sizeof(uint32_t)));
  gpuErrChk(cudaMalloc(&l_nipples_ptr, sizeof(uint32_t) * 8));
  gpuErrChk(cudaMemcpy(l_int_ptr, &l_int, sizeof(uint32_t), cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(l_nipples_ptr, &l_nipples[0], sizeof(uint32_t) * 8, cudaMemcpyHostToDevice));

  dim3 dimBlock(8, 1);
  dim3 dimGrid( 1, 1);
  launch_kernels(cuda_kernel,dimGrid, dimBlock,l_int_ptr, l_nipples_ptr);

  gpuErrChk(cudaMemcpy(l_nipples, l_nipples_ptr, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost));
  gpuErrChk(cudaFree(l_nipples_ptr));
  gpuErrChk(cudaFree(l_int_ptr));

  std::cout << std::hex << "0x" << l_int << std::dec << std::endl ;
  for(unsigned int l_index = 0; l_index < 8; ++l_index)
    {
      std::cout << "Nipple[" << l_index << "] = 0x" << std::hex <<  l_nipples[l_index] << std::dec << std::endl ;
    } 
  return EXIT_SUCCESS;
}
// EOF
