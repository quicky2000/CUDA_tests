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

COMMON_KERNEL_ATTRIBUTES
void common_kernel(MY_CUDA_PARAMS_DECL uint32_t * p_int_ptr, uint32_t * p_nipples_ptr)
{
  int l_shift = threadIdx.x << 2;
  uint32_t l_mask = ((uint32_t)0xF) << l_shift;
  p_nipples_ptr[threadIdx.x] = (*p_int_ptr & l_mask) >> l_shift;
}
__global__
void cuda_kernel(MY_CUDA_PARAMS_DECL uint32_t * p_int_ptr, uint32_t * p_nipples_ptr)
{
  common_kernel(MY_CUDA_PARAMS_INST p_int_ptr, p_nipples_ptr);
}


int main(void)
{
  uint32_t l_int = 0x87654321;
  uint32_t l_nipples[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  uint32_t * l_int_ptr;
  uint32_t * l_nipples_ptr;

  cudaMalloc(&l_int_ptr, sizeof(uint32_t));
  cudaMalloc(&l_nipples_ptr, sizeof(uint32_t) * 8);
  cudaMemcpy(l_int_ptr, &l_int, sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(l_nipples_ptr, &l_nipples[0], sizeof(uint32_t) * 8, cudaMemcpyHostToDevice);

  dim3 dimBlock(8, 1);
  dim3 dimGrid( 1, 1);
  launch_kernels(cuda_kernel,dimGrid, dimBlock,l_int_ptr, l_nipples_ptr);

  cudaMemcpy(l_nipples, l_nipples_ptr, sizeof(uint32_t) * 8, cudaMemcpyDeviceToHost);
  cudaFree(l_nipples_ptr);
  cudaFree(l_int_ptr);

  std::cout << std::hex << "0x" << l_int << std::dec << std::endl ;
  for(unsigned int l_index = 0; l_index < 8; ++l_index)
    {
      std::cout << "Nipple[" << l_index << "] = 0x" << std::hex <<  l_nipples[l_index] << std::dec << std::endl ;
    } 
  return EXIT_SUCCESS;
}
// EOF
