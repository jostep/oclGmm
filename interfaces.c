// The CUDA runtime interfaces that intercept/enhance the
// default CUDA runtime with DB resource management
// functionalities. All functions/symbols exported by the
// library reside in this file.

#include <stdint.h>
#include "common.h"

#include "core.h"
#include "client.h"
#include "protocol.h"
#include "interfaces.h"
#include "hint.h"
#include "CL/opencl.h"

// Original OpenCL handlers
//
/*
cudaError_t (*nv_cudaMalloc)(void **, size_t) = NULL;
cudaError_t (*nv_cudaFree)(void *) = NULL;
*/

cl_mem (*ocl_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int)= NULL;
cl_int (*ocl_clReleaseMemObject)(cl_mem)= NULL;
cl_context (*ocl_clCreateContext)(cl_context_properties *,cl_uint ,const cl_device_id *,void*, void *,cl_int *)=NULL;
static int initialized = 0;

// The library constructor.
// The order of initialization matters. First, link to the default
// CUDA interface implementations, because CUDA interfaces have
// been intercepted by our library and we should be able to redirect
// CUDA calls to their default implementations if GMM environment
// fails to initialize successfully. Then, initialize GMM local
// context. Finally, after the local environment has been initialized
// successfully, we connect our local context to the global GMM arena
// and let the party begin.
__attribute__((constructor))
void gmm_init(void)
{
	INTERCEPT_CL("clCreateBuffer", ocl_clCreateBuffer);
	INTERCEPT_CL("clReleaseMemObject",ocl_clReleaseMemObject);
    INTERCEPT_CL("clCreateContext",ocl_clCreateContext);	


    gprint_init();
    

	if (gmm_context_init() == -1) {
		gprint(FATAL, "failed to initialize GMM local context\n");
		return;
	}

	if (client_attach() == -1) {
		gprint(FATAL, "failed to attach to the GMM global arena\n");
		gmm_context_fini();
		return;
	}

	// Before marking GMM context initialized, invoke an NV function
	// to initialize CUDA runtime and let whatever memory regions
	// implicitly required by CUDA runtime be allocated now. Those
	// regions should be always attached and not managed by GMM runtime.
	/*do {
        cl_platform_id platform=NULL;
        cl_device_id device=NULL;
        cl_uint *num_platform;
        cl_uint *num_device;
        cl_ulong mem;
       
        clGetPlatformIDs(1,platform,num_platform);
        clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,sizeof(device),device,num_device);
        clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem),&mem,NULL);
    
        //ERCI: Currently, unable to find out api to acquire the size of the used mem.
	} while (0);
    */
	initialized = 1;
	gprint(DEBUG, "gmm initialized\n");
}

// The library destructor.
__attribute__((destructor))
void gmm_fini(void)
{
	if (initialized) {
		// NOTE: gmm_context_fini has to happen before client_detach
		// because garbage collections will need to update global info.
		// XXX: Possible bugs if client thread is busy when context is
		// being freed.
		gmm_context_fini();
		client_detach();
		initialized = 0;
	}

	gprint_fini();
}

GMM_EXPORT
cl_context clCreateContext(const cl_context_properties *properties,cl_uint num_devices,const cl_device_id *devices,void (CL_CALLBACK* pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data), void *user_data,cl_int *errcode_ret){
    cl_context ret;
    if (initialized){
        return gmm_clCreateContext(properties,num_devices,devices,pfn_notify, user_data,errcode_ret);
    }
    else{
        gprint(WARN,"clCreateContext outside the gmm\n");
        return ocl_clCreateContext(properties,num_devices,devices,pfn_notify,user_data,errcode_ret);
    } 
    return ret;
}

GMM_EXPORT
cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode){
    cl_mem ret=NULL;
    if(initialized){
       return gmm_clCreateBuffer(context,flags,size,host_ptr,errcode,0);
    }
    else {
       gprint(WARN,"clCreateBuffer called outside the GMM\n");
       return ocl_clCreateBuffer(context,flags,size,host_ptr,errcode); 
    }
    return  ret;
}
/*
// GMM-specific: allowing passing dptr array hints.
GMM_EXPORT
cudaError_t cudaMallocEx(void **devPtr, size_t size, int flags)
{
	if (initialized)
		return gmm_cudaMalloc(devPtr, size, flags);
	else {
		gprint(WARN, "cudaMallocEx called outside GMM\n");
		return nv_cudaMalloc(devPtr, size);
	}
}
*/
GMM_EXPORT
cl_int clReleaseMemObject(cl_mem memObj){
    cl_int ret;

    if(initialized){
        return gmm_clReleaseMemObject(&memObj);
    }
    else{
        gprint(WARN,"clReleaseMemObj Call");
        return ocl_clReleaseMemObject(memObj);
    }

    return ret;
}



/*
 *
 *
 *
 *
 *
 *
 *
 *
 *

GMM_EXPORT
cudaError_t cudaMemcpy(
		void *dst,
		const void *src,
		size_t count,
		enum cudaMemcpyKind kind)
{
	cudaError_t ret;
	int cow = 0;

#ifdef GMM_CONFIG_COW
	if (kind & cudaMemcpyHostToDeviceCow)
		cow = 1;
#endif
	kind = (enum cudaMemcpyKind)(kind & (~cudaMemcpyHostToDeviceCow));

	if (initialized) {
		if (kind == cudaMemcpyHostToDevice)
			ret = gmm_cudaMemcpyHtoD(dst, src, count, cow);
		else if (kind == cudaMemcpyDeviceToHost)
			ret = gmm_cudaMemcpyDtoH(dst, src, count);
		else if (kind == cudaMemcpyDeviceToDevice)
			ret = gmm_cudaMemcpyDtoD(dst, src, count);
		else {
			gprint(WARN, "HtoH memory copy not supported by GMM\n");
			ret = nv_cudaMemcpy(dst, src, count, kind);
		}
	}
	else {
		gprint(WARN, "cudaMemcpy called outside GMM\n");
		ret = nv_cudaMemcpy(dst, src, count, kind);
	}

	return ret;
}

GMM_EXPORT
cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaMemGetInfo(free, total);
	else {
		gprint(WARN, "cudaMemGetInfo called outside GMM\n");
		ret = nv_cudaMemGetInfo(free, total);
	}

	return ret;
}

GMM_EXPORT
cudaError_t cudaConfigureCall(
		dim3 gridDim,
		dim3 blockDim,
		size_t sharedMem,
		cudaStream_t stream)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	else {
		gprint(WARN, "cudaConfigureCall called outside GMM\n");
		ret = nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	}

	return ret;
}

GMM_EXPORT
cudaError_t cudaSetupArgument(
		const void *arg,
		size_t size,
		size_t offset)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaSetupArgument(arg, size, offset);
	else {
		gprint(WARN, "cudaSetupArgument called outside GMM\n");
		ret = nv_cudaSetupArgument(arg, size, offset);
	}

	return ret;
}

GMM_EXPORT
cudaError_t cudaMemset(void * devPtr, int value, size_t count)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaMemset(devPtr, value, count);
	else {
		gprint(WARN, "cudaMemset called outside GMM\n");
		ret = nv_cudaMemset(devPtr, value, count);
	}

	return ret;
}

//GMM_EXPORT
//cudaError_t cudaDeviceSynchronize()
//{
//	return nv_cudaDeviceSynchronize();
//}

GMM_EXPORT
cudaError_t cudaLaunch(const void *entry)
{
	cudaError_t ret;

	if (initialized)
		ret = gmm_cudaLaunch(entry);
	else {
		gprint(WARN, "cudaLaunch called outside GMM\n");
		ret = nv_cudaLaunch(entry);
	}

	return ret;
}

// The priority of the next kernel launch.
// Value ranges from 0 (highest) to PRIO_MAX (lowest).
int prio_kernel = PRIO_DEFAULT;

// GMM-specific: specify kernel launch priority.
GMM_EXPORT
cudaError_t cudaSetKernelPrio(int prio)
{
	if (!initialized)
		return cudaErrorInitializationError;
	if (prio < 0 || prio > PRIO_LOWEST)
		return cudaErrorInvalidValue;

	prio_kernel = prio;
	return cudaSuccess;
}

// For passing reference hints before each kernel launch.
// TODO: should prepare the following structures for each stream.
int refs[NREFS];
int rwflags[NREFS];
int nrefs = 0;

// GMM-specific: pass reference hints.
// %which_arg tells which argument (starting with 0) in the following
// cudaSetupArgument calls is a device memory pointer. %flags is the
// read-write flag.
// The GMM runtime should expect to see call sequence similar to below:
// cudaReference, ..., cudaReference, cudaConfigureCall,
// cudaSetupArgument, ..., cudaSetupArgument, cudaLaunch
//
GMM_EXPORT
cudaError_t cudaReference(int which_arg, int flags)
{
	int i;

	gprint(DEBUG, "cudaReference: %d %x\n", which_arg, flags);

	if (!initialized)
		return cudaErrorInitializationError;

	if (which_arg < NREFS) {
		for (i = 0; i < nrefs; i++) {
			if (refs[i] == which_arg)
				break;
		}
		if (i == nrefs) {
			refs[nrefs] = which_arg;
#ifdef GMM_CONFIG_RW
			rwflags[nrefs++] = flags | HINT_READ;	// let's be conservative with HINT_WRITE now
#else
			rwflags[nrefs++] = HINT_DEFAULT |
					(flags & HINT_PTARRAY) | HINT_PTADEFAULT;
#endif
		}
		else {
#ifdef GMM_CONFIG_RW
			rwflags[i] |= flags;
#endif
		}
	}
	else {
		gprint(ERROR, "bad cudaReference argument %d (max %d)\n", \
				which_arg, NREFS-1);
		return cudaErrorInvalidValue;
	}

	return cudaSuccess;
}

*
*
*
*
*
*
*
*
*
*
*
*
*/
