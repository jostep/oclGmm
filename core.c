#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <sched.h>
#include <sys/time.h>
#include <signal.h>
#include <CL/opencl.h>



#include "common.h"
#include "client.h"
#include "core.h"
#include "hint.h"
#include "replacement.h"
#include "msq.h"
#include "debug.h"




static int gmm_free(struct region *m);
extern cl_mem (*ocl_clCreateBuffer)(cl_context , cl_mem_flags, size_t, void *, cl_int*);
extern cl_int (*ocl_clReleaseMemObject)(cl_mem);
//extern cl_int (*ocl_clReleaseCommandQueue)(cl_command_queue);
extern cl_context(*ocl_clCreateContext)(cl_context_properties * ,cl_uint ,const cl_device_id *,void*, void *,cl_int*);
//extern cl_command_queue (*ocl_clCreateCommandQueue)(cl_context, cl_device_id,cl_command_queue_properties,cl_int *);

struct region * region_lookup(struct gmm_context *ctx, const cl_mem *ptr);

static int dma_channel_init(struct gmm_context *ctx,struct dma_channel *chan, int htod);
static void dma_channel_fini(struct dma_channel *chan);
struct gmm_context *pcontext=NULL;
static void list_alloced_add(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_alloced);
	list_add(&r->entry_alloced, &ctx->list_alloced);
	release(&ctx->lock_alloced);
}

static void list_alloced_del(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_alloced);
	list_del(&r->entry_alloced);
	release(&ctx->lock_alloced);
}

static void list_attached_add(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_add(&r->entry_attached, &ctx->list_attached);
	release(&ctx->lock_attached);
}

static void list_attached_del(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_del(&r->entry_attached);
	release(&ctx->lock_attached);
}

static void list_attached_mov(struct gmm_context *ctx, struct region *r)
{
	acquire(&ctx->lock_attached);
	list_move(&r->entry_attached, &ctx->list_attached);
	release(&ctx->lock_attached);
}


static inline void region_pin(struct region *r)
{
	int pinned = atomic_inc(&(r)->pinned);
	if (pinned == 0)
		update_detachable(-r->size);
}

static inline void region_unpin(struct region *r)
{
	int pinned = atomic_dec(&(r)->pinned);
	if (pinned == 1)
		update_detachable(r->size);
}

long LMAX(long a, long b)
{
	return a < b ? b : a;
}

int gmm_context_init(){

    if(pcontext!=NULL){ 
        gprint(FATAL,"pcontext already exists!\n");
        return -1;
    }
   
    pcontext = (struct gmm_context *)malloc(sizeof (*pcontext));
    if(!pcontext){
        gprint(FATAL,"malloc failed for pcontext: %s\n",strerror(errno));
        return -1;
    }
    /*
    //get platform and id for the pcontext,then create corresponding context and CQ for
    //the pcontext.
    if(clGetPlatformIDs(numPlatform,pcontext->platform,NULL)!=CL_SUCCESS){
        gprint(FATAL,"Cannot get platform ID!\n",strerror(errno));
    }

    if(clGetDeviceIDs(pcontext->platform[0],CL_DEVICE_TYPE_GPU,1,pcontext->device,numDevices)!=CL_SUCCESS){
        gprint(FATAL,"failed to get device ID!\n",strerror(errno));
        }

    pcontext->context_kernel=clCreateContext(NULL,1,pcontext->device,NULL,NULL,errcode_CON);
    if(errcode_CON!=CL_SUCCESS){

        gprint(FATAL,"failed to create context\n");
    }
    */
    initlock(&pcontext->lock);
    latomic_set(&pcontext->size_attached,0L);
    INIT_LIST_HEAD(&pcontext->list_alloced);
    INIT_LIST_HEAD(&pcontext->list_attached);
    initlock(&pcontext->lock_alloced);
    initlock(&pcontext->lock_attached);


    stats_init(&pcontext->stats);
    return 0;
}

void gmm_context_initEX(){
   
    cl_int * errcode_CQ=NULL; 
    //failed to show the diff between htod and dtoh 
    printf("we've finished the first dma_init and before it.\n");
    if(dma_channel_init(pcontext,&pcontext->dma_htod,1)!=0){
        gprint(FATAL,"failed to create HtoD DMA channel\n");
        free(pcontext);
        pcontext=NULL;
        return ;
    }    
    printf("we've finished the first dma_init\n");
    if(dma_channel_init(pcontext,&pcontext->dma_dtoh,0)!=0){
        gprint(FATAL,"failed to create DtoH DMA channel\n");
        dma_channel_fini(&pcontext->dma_htod);
        free(pcontext);
        pcontext=NULL;
        return ;
    }
    printf("we've finished the second dma_init\n");
    pcontext->commandQueue_kernel=clCreateCommandQueue(pcontext->context_kernel,pcontext->device[0],CL_QUEUE_PROFILING_ENABLE,errcode_CQ);// add error handler;ERCI
    if (errcode_CQ!=CL_SUCCESS){
        
        gprint(FATAL,"failed to create command queue\n");
    } 
    printf("we've finished the command queue creation\n");
}


void gmm_context_fini(){

    struct list_head *p;
    struct region *r;

    while(!list_empty(&pcontext->list_alloced)){
        p=pcontext->list_alloced.next;
        r=list_entry(p, struct region, entry_alloced);
        if(gmm_free(r)){
            list_move_tail(p,&pcontext->list_alloced);
        }
    }

    dma_channel_fini(&pcontext->dma_dtoh);
    dma_channel_fini(&pcontext->dma_htod);

    clReleaseCommandQueue(pcontext->commandQueue_kernel);

    stats_print(&pcontext->stats);
    free(pcontext);
    pcontext=NULL;
}

cl_context gmm_clCreateContext(cl_context_properties *properties,cl_uint num_devices,const cl_device_id *devices,void *pfn_notify (const char *errinfo, const void *private_info, size_t cb, void *user_data), void *user_data,cl_int *errcode_ret){
    pcontext->device=devices;
    pcontext->context_kernel=ocl_clCreateContext(properties,num_devices,devices,pfn_notify,user_data,errcode_ret);
    printf("we are before the INIIEX\n");
    gmm_context_initEX();//finishing the init process of gmm_context
    printf("initEX finished\n");
    return pcontext->context_kernel;
}

cl_mem gmm_clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int* errcode_CB, int gmm_flags){
    struct region *r;
    int nblocks;

    gprint(DEBUG,"clCreateBuffer begins, size(%lu), flags(%x)\n",size,flags);

    if(size>memsize_total()){
        gprint(ERROR,"clCreateBuffer size(%lu) too large(max %ld)\n", size, memsize_total());
        errcode_CB=CL_INVALID_VALUE;
        return NULL;
    }
    else if(size<=0){
        gprint(ERROR,"clCreateBuffer size (%lu) too small\n", size);
        errcode_CB=CL_INVALID_VALUE;
        return NULL;
    }

    r = (struct region*) calloc(1,sizeof(*r));
    if(!r){
        gprint(FATAL,"malloc for a new region: %s\n",strerror(errno));
        errcode_CB=CL_MEM_OBJECT_ALLOCATION_FAILURE;
        return NULL;
    }

    r->swp_addr=malloc(size);
    if (!r->swp_addr){

        gprint(FATAL,"malloc for a swap buffer: %s\n",strerror(errno));
        free(r);
        errcode_CB=CL_MEM_OBJECT_ALLOCATION_FAILURE;
        return NULL;
    }

    if (gmm_flags & HINT_PTARRAY){
        if(size%sizeof(void *)){
            gprint (ERROR, "dptr array size (%lu) not aligned \n",size);
            free(r->swp_addr);
            free(r);
            errcode_CB=CL_INVALID_VALUE;
            return NULL;
        }
    
        r->pta_addr=calloc(1,size);
        if (!r->pta_addr){

            gprint(FATAL,"malloc for a dptr array: %s\n",strerror(errno));
            free(r);
            errcode_CB=CL_MEM_OBJECT_ALLOCATION_FAILURE;
            return NULL;
        }
    }
    r->size= (long) size;
    r->state= STATE_DETACHED;
    r->flags=flags;
    r->gmm_flags=gmm_flags;
    
    list_alloced_add(pcontext, r);
    stats_inc_alloc(&pcontext->stats, size);

    gprint(DEBUG, "clCreateBuffer ends : r(%p) swp(%p) pta(%p)\n",r,r->swp_addr,r->pta_addr);

    errcode_CB=CL_SUCCESS;

    return *r->swp_addr; 
}

static int dma_channel_init(struct gmm_context *ctx,struct dma_channel *chan, int htod){
   int ret =0;
   cl_int * errcode_DMA=NULL;
   cl_int * errcode_INIT=NULL;
   printf("inside the first the commandQueue Creation and before it.");
#ifdef GMM_CONFIG_DMA_ASYNC
   int i;
#endif 

   chan->commandQueue_chan=/*ocl_*/clCreateCommandQueue(ctx->context_kernel,ctx->device[0],CL_QUEUE_PROFILING_ENABLE,errcode_DMA);//
   if(errcode_INIT!=CL_SUCCESS){
        gprint(FATAL,"failed to create channel\n");
   }
   printf("inside the first the commandQueue Creation");

#ifdef GMM_CONFIG_DMA_ASYNC
   initlock(&chan->lock);
   chan->ibuf=0;
   //
   for (i=0;i<NBUFS;i++){
        ocl_clCreateBuffer(pcontext->context_kernel,CL_MEM_ALLOC_HOST_PTR,BUFSIZE,chan->stage_bufs[i],errcode_DMA);
        if(errcode_DMA!=CL_SUCCESS){
            gprint(FATAL,"failed for staging buffer\n");
            break;
        }
        errcode_DMA=NULL;
        chan->stage_bufs[i]=clCreateUserEvent(pcontext->context_kernel,errcode_DMA);
        if(errcode_DMA!=CL_SUCCESS){
            gprint(FATAL,"failed to create for staging buffer\n");
            ocl_clReleaseMemObject(chan->stage_bufs[i]);
            break;
        }
        if (i<NBUFS){
            while(--i>0){
                clReleaseEvent(chan->events[i]);
                ocl_clReleaseMemObject(chan->stage_bufs[i]);
            }
        }

        ret=-1;

   }
#endif
    return ret;
}

static void dma_channel_fini(struct dma_channel *chan){
#ifdef GMM_CONFIG_DMA_ASYNC
    int i;

    for (i=0;i<NBUFS;i++){
        clReleaseEvent(chan->events[i]);
        ocl_clReleaseMemObject(chan->stage_bufs[i]);
    }
#endif 

    clReleaseCommandQueue(chan->commandQueue_chan);
}

cl_int gmm_clReleaseMemObject(cl_mem *memObjPtr){
    
    struct region *r;
    struct list_head *pos;
    int found=0;
    
    if (!(r= region_lookup(pcontext, memObjPtr))){
        free(r->blocks);
        gprint(ERROR,"cannot find region containg %p in clReleaseMemObject",memObjPtr);
    }
    stats_inc_freed(&pcontext->stats,r->size);
    if(gmm_free(r)<0)
        return CL_INVALID_MEM_OBJECT;
    return CL_SUCCESS;
}

struct region * region_lookup(struct gmm_context *ctx, const cl_mem *ptr){
        
    struct region *r = NULL;
    struct list_head *pos;
    int found =0;

    acquire (&ctx->lock_alloced);
    list_for_each(pos, &ctx->list_alloced){
        r= list_entry(pos, struct region, entry_alloced);
        if (r->state==STATE_FREEING||r->state==STATE_ZOMBIE){
            continue;
        }
        if(((unsigned long)ptr >=(unsigned long)(r->swp_addr))&&
           ((unsigned long)ptr<
           ((unsigned long)(r->swp_addr)+(unsigned long)(r->size)))){
            found =1;
            break;
        }
    }
    
    release(&ctx->lock_alloced);
    if(!found){
        r=NULL;
    }
    return r;
}

static int gmm_free(struct region *r){
    gprint(DEBUG, "freeing r (%p %p %ld %d %d)\n",r,r->swp_addr,r->size,r->flags,r->state);
re_acquire:
    acquire(&r->lock);
    switch (r->state){
    case STATE_ATTACHED:
        if(!region_pinned(r))
            list_attached_del(pcontext,r);
        else{
            release(&r->lock);
            sched_yield();
            goto re_acquire;
        }
        break;
    case STATE_EVICTING:
        r->state=STATE_FREEING;
        release(&r->lock);
        gprint(DEBUG,"region set freeing");
        return 0;
    case STATE_FREEING:
        release(&r->lock);
        sched_yield();
        goto re_acquire;
    default:
        break;
    }
    release(&r->lock);
    list_attached_del(pcontext,r);
    if(r->blocks)
        free(r->blocks);
    if(r->pta_addr)
        free(r->blocks);
    if(r->swp_addr)
        free(r->blocks);
    if(r->dev_addr){
        ocl_clReleaseMemObject(*r->dev_addr);
        latomic_sub(&pcontext->size_attached,r->size);
        update_attached(-r->size);
        update_detachable(-r->size);
    }
    free(r);
    gprint(DEBUG,"region freed\n");
    return 1;

}

/*
long region_evict(struct region *r)
{
	int nblocks = NRBLOCKS(r->size);
	long size_spared = 0;
	char *skipped;
	int i, ret = 0;

	gprint(INFO, "evicting region %p\n", r);
	//gmm_print_region(r);

	if (!r->dev_addr)
		panic("dev_addr is null");
	if (region_pinned(r))
		panic("evicting a pinned region");

	skipped = (char *)calloc(nblocks, sizeof(char));
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}

	// First round
	for (i = 0; i < nblocks; i++) {
		if (r->state == STATE_FREEING)
			goto success;
		if (try_acquire(&r->blocks[i].lock)) {
			if (!r->blocks[i].swp_valid) {
				ret = block_sync(r, i);
			}
			release(&r->blocks[i].lock);
			if (ret != 0)
				goto finish;	// this is problematic if r is freeing
			skipped[i] = 0;
		}
		else
			skipped[i] = 1;
	}

	// Second round
	for (i = 0; i < nblocks; i++) {
		if (r->state == STATE_FREEING)
			goto success;
		if (skipped[i]) {
			acquire(&r->blocks[i].lock);
			if (!r->blocks[i].swp_valid) {
				ret = block_sync(r, i);
			}
			release(&r->blocks[i].lock);
			if (ret != 0)
				goto finish;	// this is problematic if r is freeing
		}
	}

success:
	list_attached_del(pcontext, r);
	if (r->dev_addr) {
		nv_cudaFree(r->dev_addr);
		r->dev_addr = NULL;
		size_spared = r->size;
	}
	latomic_sub(&pcontext->size_attached, r->size);
	update_attached(-r->size);
	update_detachable(-r->size);
	region_inval(r, 0);
	acquire(&r->lock);
	if (r->state == STATE_FREEING) {
		if (r->swp_addr) {
			free(r->swp_addr);
			// region_lookup will not look up a region marked freed,
			// so we can safely set swp_addr to null without worrying
			// about wrong region_lookup results.
			r->swp_addr = NULL;
		}
		r->state = STATE_ZOMBIE;
	}
	else
		r->state = STATE_DETACHED;
	release(&r->lock);

	gprint(INFO, "region evicted\n");
	//gmm_print_region(r);

finish:
	free(skipped);
	return (ret == 0) ? size_spared : (-1);
}


long local_victim_evict(long size_needed)
{
	struct list_head victims;
	struct victim *v;
	struct region *r;

	gprint(DEBUG, "local eviction: %ld bytes\n", size_needed);
	INIT_LIST_HEAD(&victims);

	if (victim_select(size_needed, NULL, 0, 1, &victims) != 0)
		return -1;

	if (list_empty(&victims))
		return 0;

	v = list_entry(victims.next, struct victim, entry);
	r = v->r;
	free(v);
	return region_evict(r);
}
*/
