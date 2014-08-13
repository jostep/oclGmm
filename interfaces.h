#ifndef _GMM_INTERFACES_H_
#define _GMM_INTERFACES_H_


#define __USE_GNU
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#if defined __APPLE__ || defined (MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#define OPENCL_PATH (CLPATH "/usr/lib/libOpenCL") 

#define TREAT_ERROR()                           \
do{                                             \
    char * __error;                             \
    if ((__error = dlerror()) != NULL) {        \
    fputs(__error, stderr);                     \
    abort();                                    \
    }                                           \
                    }while(0)






/* Intercept function func and store its previous value into var */
#define INTERCEPT_CL(func, var)       \
    do {                                    \
                if(var) break;                  \
                var = (typeof(var))dlsym(RTLD_NEXT, func);  \
                TREAT_ERROR();                  \
    } while(0)


/*
 *
#define INTERCEPT_CUDA2(func, var)  \
    do {                                    \
                if(var) break;                  \
                void *__handle = dlopen(CUDA_CURT_PATH, RTLD_LOCAL | RTLD_LAZY);    \
                var = (typeof(var))dlsym(__handle, func);  \
                TREAT_ERROR();                  \
    } while(0)
*
*
*/



#endif
