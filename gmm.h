// Include this file in user program to access GMM-specific features.
#ifndef _GMM_H_
#define _GMM_H_

#include "hint.h"

#ifdef __cplusplus
extern "C" {
#endif

// The GMM extensions to CUDA runtime interfaces.
// Interface implementations reside in interfaces.c.
//cudaError_t cudaMallocEx(void **devPtr, size_t size, int flags);
//cudaError_t cudaSetKernelPrio(int prio);

int clReference(int which_arg, int flags);

#ifdef __cplusplus
}
#endif

#endif
