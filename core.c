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
extern cl_command_queue (*ocl_clCreateCommandQueue)(cl_context, cl_device_id,cl_command_queue_properties,cl_int *);
extern cl_int (*ocl_clEnqueueFillBuffer)(cl_command_queue, cl_mem,const void * , size_t, size_t,size_t, cl_uint, const cl_event, cl_event);
void gmm_context_initEX();
static int gmm_memset(struct region *r, cl_mem *dst, int value, size_t count);
extern cl_int (*ocl_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void* , cl_uint, const cl_event *, cl_event *);


struct region * region_lookup(struct gmm_context *ctx, const cl_mem ptr);
static int dma_channel_init(struct gmm_context *,struct dma_channel *, int);
static void dma_channel_fini(struct dma_channel *chan);
struct gmm_context *pcontext=NULL;
static int gmm_memset(struct region *r, cl_mem buffer,int value, size_t count);
static int block_sync(struct region *r, int block);
static int gmm_htod_block(struct region *r,unsigned long offset,const void *src,
                          \unsigned long size,int block,int skip,char *skipped);


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

static void begin_dma(struct dma_channel *chan){
#ifdef GMM_CONFIG_DMA_ASYNC
    acquire(&chan->lock);
#endif
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
    gprint(DEBUG,"check the ptrs for pcontext %p, and dma: %p ",pcontext,&pcontext->dma_htod);  
    cl_int * errcode_CQ=NULL; 
    /*
     * //failed to show the diff between htod and dtoh 
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
    */
    pcontext->commandQueue_kernel=ocl_clCreateCommandQueue(pcontext->context_kernel,pcontext->device[0],CL_QUEUE_PROFILING_ENABLE,errcode_CQ);// add error handler;ERCI
    if (errcode_CQ!=CL_SUCCESS){
        
        gprint(FATAL,"failed to create command queue\n");
    } 
}


void gmm_context_fini(){

    struct list_head *p;
    struct region *r;
    while(!list_empty(&pcontext->list_alloced)){
        p=pcontext->list_alloced.next;
        r=list_entry(p, struct region, entry_alloced);
        if(!gmm_free(r)){
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
    gprint(DEBUG,"ocl has not finished?");
    if(errcode_ret!=CL_SUCCESS){
        printf("Cannot create the Context for the PC\n");
    }
    gmm_context_initEX();//finishing the init process of gmm_context
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
    return pcontext->context_kernel;
}

cl_mem gmm_clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int* errcode_CB, int gmm_flags){
    struct region *r;
    int nblocks;
    gprint(DEBUG,"clCreateBuffer begins, size(%lu), flags(%x)\n",size,flags);
    if(context!=pcontext->context_kernel){
        gprint(FATAL,"context unmatched\n");
    }
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
    
    return r->swp_addr; 
}

static int dma_channel_init(struct gmm_context *ctx,struct dma_channel *chan, int htod){
   int ret =0;
   cl_int * errcode_DMA=NULL;
   cl_int * errcode_INIT=NULL;
#ifdef GMM_CONFIG_DMA_ASYNC
   int i;
#endif 

   chan->commandQueue_chan=ocl_clCreateCommandQueue(ctx->context_kernel,ctx->device[0],CL_QUEUE_PROFILING_ENABLE,errcode_DMA);//
   if(errcode_INIT!=CL_SUCCESS){
        gprint(FATAL,"failed to create channel\n");
   }

#ifdef GMM_CONFIG_DMA_ASYNC
   initlock(&chan->lock);
   chan->ibuf=0;
   
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
   }
        if (i<NBUFS){
            while(--i>0){
                clReleaseEvent(chan->events[i]);
                ocl_clReleaseMemObject(chan->stage_bufs[i]);
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

cl_int gmm_clReleaseMemObject(cl_mem memObjPtr){
    
    struct list_head *pos;
    int found=0;
    struct region *r;
    if (!(r= region_lookup(pcontext, memObjPtr))){
        free(r->blocks);
        gprint(ERROR,"cannot find region containg %p in clReleaseMemObject",memObjPtr);
    }
    gprint(DEBUG,"after freeing the blocks and region lookup");
    stats_inc_freed(&pcontext->stats,r->size);
    if(gmm_free(r)<0)
        return CL_INVALID_MEM_OBJECT;
    return CL_SUCCESS;
}

struct region * region_lookup(struct gmm_context *ctx, const cl_mem ptr){
        
    struct region *r = NULL;
    struct list_head *pos;
    int found =0;

    acquire (&ctx->lock_alloced);
    list_for_each(pos, &ctx->list_alloced){
        r= list_entry(pos, struct region, entry_alloced);
        if (r->state==STATE_FREEING||r->state==STATE_ZOMBIE){
            continue;
        }
        if(((unsigned long)(ptr) >=(unsigned long)(r->swp_addr))&&
           ((unsigned long)(ptr)<
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
    gprint(DEBUG,"freeing r (r %p|| swp_addr%p|| size %ld||flags %d||r-state %d)\n",r,r->swp_addr,r->size,r->flags,r->state);
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
    list_alloced_del(pcontext,r);
    gprint(DEBUG,"Before releasing content in r, the ptr are here %p,%p,%p\n",r->pta_addr,r->swp_addr,r->dev_addr);
    if(r->blocks)
        free(r->blocks);
    if(r->pta_addr)
        free(r->blocks);
    if(r->swp_addr)
        free(r->blocks);
    if(r->dev_addr){
        ocl_clReleaseMemObject(r->dev_addr);
        latomic_sub(&pcontext->size_attached,r->size);
        update_attached(-r->size);
        update_detachable(-r->size);
    }
    free(r);
    gprint(DEBUG,"region freed\n");
    return 1;

}

cl_int gmm_clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem  buffer, int value, size_t pattern_size, size_t offset,size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event* event){

    struct region *r;
    //ALSO have to check CQ,context,here.
    if(size<=0){
        return CL_INVALID_VALUE;
    }
    r=region_lookup(pcontext,buffer);
    if(!r){
        gprint(FATAL,"cannot find the region\n");
        return CL_INVALID_MEM_OBJECT;
    }
    if(r->state==STATE_FREEING||r->state==STATE_ZOMBIE){
        gprint(ERROR,"region already freedi\n");
        return CL_INVALID_MEM_OBJECT;
    }
    if(r->flags & FLAG_COW){
        gprint(ERROR,"region tagged CoW\n");
        return CL_INVALID_MEM_OBJECT;
    }
    if(buffer+size>&(r->swp_addr)+r->size){
        gprint(ERROR,"size overloaded\n");
        return CL_INVALID_VALUE;
    }
    if(r->size!=size){
        gprint(ERROR,"should set the whole region");
        return -1;
    }
    stats_time_begin();
    if(gmm_memset(r,buffer,value,size)<0){
        gprint(ERROR,"memsetting error unknown\n");
    }
    stats_time_end(&pcontext->stats,time_memset);
    stats_inc(&pcontext->stats,bytes_memset,size);

    return CL_SUCCESS; 
}



static int gmm_memset(struct region *r, cl_mem buffer,int value, size_t count){
    
    unsigned long off,end,count;
    int ifirst, ilast, iblock;
    char *skipped;
    int ret=0;
    void *s;

    gprint(DEBUG,"memset: r(%p,%p,%ld)dst(%p)value(%d)count(%lu)",\
           r,r->swp_addr,r->size,buffer,value,count);
    if(r->gmm_flags & HINT_PTARRAY){
        return gmm_memset_pta(r,buffer,value,count);
    }
    if(r->state==STATE_DETACHED){
        gprint(DEBUG,"memset: setting a detached region\n");
#ifdef GMM_CONFIG_COW
        r->gmm_flags |= FLAG_MEMSET;
        r->value_memset=value;
#else 
        memset(buffer,value,count);
        region_valid(r,1);
#endif
        region_inval(r,0);
        return 0;
    }
    gprint(DEBUG,"memset: setting a non-detached region\n");
    
    s=malloc(BLOCKSIZE);
    if(!s){
        gprint(FATAL,"malloc failed for skipped[]: %s\n",strerror(errno));
        free(s);
        return -1;
    }

    for(iblock=ifirst;iblock<=ilast;iblock++){
        count=MIN(BLOCKUP(off),end)-off;
        ret=gmm_htod_block(r,off,s,size,iblock,1,skipped+(iblock-ifirst));
        if(ret!=0)
            goto finish;
        off +=size;
    }

    off=(unsigned long)(buffer - r->swp_addr);

    for(iblock=ifirst;iblock<=ilast;iblock++){
        count=MIN(BLOCKUP(off),end)-off;
        if(skipped[iblock-ifirst]){
            ret=gmm_htod_block(r,off,s,size,iblock,0,NULL);
            if(ret!=0)
                goto finish;
        }
        off +=size;
    }

finish:
    free(skipped);
    free(s);
    return ret;
} 

static int gmm_memcpy_dtoh(void *dst, const void *src, unsigned long size)
{
	struct dma_channel *chan = &pcontext->dma_dtoh;
	unsigned long	off_dtos,	// Device to Stage buffer
    off_stoh,	// Stage buffer to Host buffer
    delta;
	int ret = 0, ibuf_old;
    
	begin_dma(chan);
    
	// First issue DtoH commands for all staging buffers
	ibuf_old = chan->ibuf;
	off_dtos = 0;
	while (off_dtos < size && off_dtos < NBUFS * BUFSIZE) {
		delta = MIN(off_dtos + BUFSIZE, size) - off_dtos;
		if (cudaMemcpyAsync(chan->stage_bufs[chan->ibuf], src + off_dtos,
                            delta, cudaMemcpyDeviceToHost, chan->stream) != cudaSuccess) {
			gprint(FATAL, "cudaMemcpyAsync failed in dtoh\n");
			ret = -1;
			goto finish;
		}
		if (cudaEventRecord(chan->events[chan->ibuf], chan->stream)
            != cudaSuccess) {
			gprint(FATAL, "cudaEventRecord failed in dtoh\n");
			ret = -1;
			goto finish;
		}
        
		chan->ibuf = (chan->ibuf + 1) % NBUFS;
		off_dtos += delta;
	}
    
	// Now copy data to user buffer, meanwhile issuing the
	// rest DtoH commands if any.
	chan->ibuf = ibuf_old;
	off_stoh = 0;
	while (off_stoh < size) {
		delta = MIN(off_stoh + BUFSIZE, size) - off_stoh;
        
		if (cudaEventSynchronize(chan->events[chan->ibuf]) != cudaSuccess) {
			gprint(FATAL, "cudaEventSynchronize failed in dtoh\n");
			ret = -1;
			goto finish;
		}
		memcpy(dst + off_stoh, chan->stage_bufs[chan->ibuf], delta);
		off_stoh += delta;
        
		if (off_dtos < size) {
			delta = MIN(off_dtos + BUFSIZE, size) - off_dtos;
			if (cudaMemcpyAsync(chan->stage_bufs[chan->ibuf], src + off_dtos,
                                delta, cudaMemcpyDeviceToHost, chan->stream)
                != cudaSuccess) {
				gprint(FATAL, "cudaMemcpyAsync failed in dtoh\n");
				ret = -1;
				goto finish;
			}
			if (cudaEventRecord(chan->events[chan->ibuf], chan->stream)
                != cudaSuccess) {
				gprint(FATAL, "cudaEventRecord failed in dtoh\n");
				ret = -1;
				goto finish;
			}
			off_dtos += delta;
		}
        
		chan->ibuf = (chan->ibuf + 1) % NBUFS;
	}
    
finish:
	end_dma(chan);
	return ret;
}


static int gmm_htod_pta(
        struct region *r,
        cl_mem dst,
        const cl_mem src,
        size_t count){

    unsigned long = (unsigned long)(dst- r->swp_addr);

    gprint(DEBUG,"htod_pta: r(%p %lu %lu) nost aligned for htod_pta\n",r->swp_addr,r->size,off);

    if(off%sizeof(void *)){
        gprint(ERROR,"offset(%lu) not aligned for htod_pta\n",off);
        return -1;
    }
    if(count%sizeof(void *)){
        gprint(ERROR,"count (%lu)not aligned for htod_pta",count);
        return -1;
    }
    stats_time_begin();
    memcpy(r->pta_addr+off, src, count);
    stats_time_end(&pcontext->stats,time_u2s);
    stats_inc(&pcontext->stats,bytes_u2s,count);

    return 0;
}

static int gmm_memcpy_htod(cl_mem dst, const cl_mem src, unsigned long size){

    struct dma_channel *chan = &pcontext->dma_htod;
    unsigned long off,delta;
    int ret=0, ilast;
    
    begin_dma(chan);
    off=0;
    while(off<size){
        detla=MIN(off+BUFSIZE, size)-off;
        if()
    }



}
static int gmm_htod_block(
                          struct region *r,
                          unsigned long offset,
                          const void *src,
                          unsigned long size,
                          int block,
                          int skip,
                          char *skipped)
{
	struct block *b = r->blocks + block;
	int partial = (offset % BLOCKSIZE) ||
    (size < BLOCKSIZE && (offset + size) < r->size);
	int ret = 0;
    
	if (BLOCKIDX(offset) != block)
		panic("htod_block");
    
    /*	GMM_DPRINT("gmm_htod_block: r(%p) offset(%lu) src(%p)" \
     " size(%lu) block(%d) partial(%d)\n", \
     r, offset, src, size, block, partial);
     */
	// This `no-locking' case will cause a subtle block sync problem:
	// Suppose this block is invalid in both swp and dev, then the
	// following actions will set its swp to valid. Now if the evictor
	// sees invalid swp before it being set to valid and decides to do
	// a block sync, it may accidentally sync the data from host to
	// device, which should never happen during a block eviction. So
	// the safe action is to only test/change the state of a block while
	// holding its lock.
    //	if (b->swp_valid || !b->dev_valid) {
    //		// no locking needed
    //		memcpy(r->swp_addr + offset, src, size);
    //		if (!b->swp_valid)
    //			b->swp_valid = 1;
    //		if (b->dev_valid)
    //			b->dev_valid = 0;
    //	}
    //	else {
    
	stats_time_begin();
	while (!try_acquire(&b->lock)) {
		if (skip) {
			if (skipped)
				*skipped = 1;
			return 0;
		}
	}
	stats_time_end(&pcontext->stats, time_sync);
    
	if (b->swp_valid || !b->dev_valid) {
		if (!b->swp_valid)
			b->swp_valid = 1;
		if (b->dev_valid)
			b->dev_valid = 0;
		release(&b->lock);
        
		stats_time_begin();
		// this is not thread-safe; otherwise, move memcpy before release
		memcpy(r->swp_addr + offset, src, size);
		stats_time_end(&pcontext->stats, time_u2s);
		stats_inc(&pcontext->stats, bytes_u2s, size);
	}
	else { // dev_valid == 1 && swp_valid == 0
		if (partial) {
			// XXX: We don't need to pin the device memory because we are
			// holding the lock of a swp_valid=0,dev_valid=1 block, which
			// will prevent the evictor, if any, from freeing the device
			// memory under us.
			ret = block_sync(r, block);
			if (ret == 0)
				b->dev_valid = 0;
			release(&b->lock);
			if (ret != 0)
				goto finish;
		}
		else {
			b->swp_valid = 1;
			b->dev_valid = 0;
			release(&b->lock);
		}
        
		stats_time_begin();
		memcpy(r->swp_addr + offset, src, size);
		stats_time_end(&pcontext->stats, time_u2s);
		stats_inc(&pcontext->stats, bytes_u2s, size);
	}
    
    //	}
    
finish:
	if (skipped)
		*skipped = 0;
	return ret;
}



static int block_sync(struct region *r, int block)
{
	int dvalid = r->blocks[block].dev_valid;
	int svalid = r->blocks[block].swp_valid;
	unsigned long off, size;
	int ret = 0;
    
	// Nothing to sync if both are valid or both are invalid
	if ((dvalid ^ svalid) == 0) {
		/*gprint(WARN, "nothing to do for block_sync r(%p %p %lu %d %d) " \
         "block(%d)\n", r, r->swp_addr, r->size, r->flags, r->state, \
         block);*/
		return 0;
	}
	if (!r->dev_addr || !r->swp_addr)
		panic("block_sync");
    
	gprint(DEBUG, \
           "block sync begins: r(%p) block(%d) svalid(%d) dvalid(%d)\n", \
           r, block, svalid, dvalid);
    
	// Have to wait until the kernel modifying the region finishes,
	// otherwise it is possible that the data we read are inconsistent
	// with what's being written by the kernel.
	// TODO: will make the modifying flags more fine-grained (block level).
	stats_time_begin();
	while (atomic_read(&r->writing) > 0) ;
	stats_time_end(&pcontext->stats, time_sync);
    
	off = block * BLOCKSIZE;
	size = MIN(off + BLOCKSIZE, r->size) - off;
	if (dvalid && !svalid) {
		stats_time_begin();
		ret = gmm_memcpy_dtoh(r->swp_addr + off, r->dev_addr + off, size);
		if (ret == 0)
			r->blocks[block].swp_valid = 1;
		stats_time_end(&pcontext->stats, time_d2s);
		stats_inc(&pcontext->stats, bytes_d2s, size);
	}
	else if (!dvalid && svalid) {
		stats_time_begin();
		while (atomic_read(&r->reading) > 0) ;
		stats_time_end(&pcontext->stats, time_sync);
        
		stats_time_begin();
		ret = gmm_memcpy_htod(r->dev_addr + off, r->swp_addr + off, size);
		if (ret == 0)
			r->blocks[block].dev_valid = 1;
		stats_time_end(&pcontext->stats, time_s2d);
		stats_inc(&pcontext->stats, bytes_s2d, size);
	}
	else
		panic("block_sync");
    
	gprint(DEBUG, "block sync ends\n");
	return ret;
}

/*
static int gmm_memset(struct region *r, cl_mem *dst, int value, size_t count)
{
	unsigned long off, end, size;
	int ifirst, ilast, iblock;
	char *skipped;
	int ret = 0;
	void *s;

	gprint(DEBUG, "memset: r(%p %p %ld) dst(%p) value(%d) count(%lu)\n", \
			r, r->swp_addr, r->size, dst, value, count);

	if (r->flags & HINT_PTARRAY)
		return gmm_memset_pta(r, dst, value, count);

    if (r->state == STATE_DETACHED) {
        gprint(DEBUG, "memset: setting a detached region\n");
#ifdef GMM_CONFIG_COW
        // Note: the data movement overhead cannot be removed if the
        // region is read-only but later evicted from device memory.
        r->flags |= FLAG_MEMSET;
        r->value_memset = value;
#else
        memset(dst, value, count);
        region_valid(r, 1);
#endif
        region_inval(r, 0);
        return 0;
    }
    gprint(DEBUG, "memset: setting a non-detached region\n");

	// The temporary source buffer holding %value's
	s = malloc(BLOCKSIZE);
	if (!s) {
		gprint(FATAL, "malloc failed for memset temp buffer: %s\n", \
				strerror(errno));
		return -1;
	}
	memset(s, value, BLOCKSIZE);

	off = (unsigned long)(dst - &r->swp_addr);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)malloc(ilast - ifirst + 1);
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		free(s);
		return -1;
	}

	// Copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely because it's being evicted).
	// skipped[] records whether each block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		ret = gmm_htod_block(r, off, s, size, iblock, 1,
				skipped + (iblock - ifirst));
		if (ret != 0)
			goto finish;
		off += size;
	}

	// Then, copy the rest blocks, no skipping.
	off = (unsigned long)(dst -(&r->swp_addr));
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
			ret = gmm_htod_block(r, off, s, size, iblock, 0, NULL);
			if (ret != 0)
				goto finish;
		}
		off += size;
	}

finish:
	free(skipped);
	free(s);
	return ret;
}

*/

#if defined(GMM_CONFIG_HTOD_RADICAL)
// The radical version
static int gmm_htod(
                    struct region *r,
                    void *dst,
                    const void *src,
                    size_t count)
{
	int iblock, ifirst, ilast;
	unsigned long off, end;
	void *s = (void *)src;
	char *skipped;
	int ret = 0;
    
	if (r->flags & HINT_PTARRAY)
		return gmm_htod_pta(r, dst, src, count);
    
	off = (unsigned long)(dst - r->swp_addr);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)calloc(ilast - ifirst + 1, sizeof(char));
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}
    
	// For each full-block over-writing, set dev_valid=0 and swp_valid=1.
	// Since we know the memory range being over-written, setting flags ahead
	// help prevent the evictor, if there is one, from wasting time evicting
	// those blocks. This is one unique advantage of us compared with CPU
	// memory management, where the OS usually does not have such interfaces
	// or knowledge.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		if ((offset % BLOCKSIZE) == 0 &&
			(size == BLOCKSIZE || (offset + size) == r->size)) {
			if (try_acquire(&r->blocks[iblock].lock)) {
				r->blocks[iblock].dev_valid = 0;
				r->blocks[iblock].swp_valid = 1;
				release(&r->blocks[iblock].lock);
			}
		}
		off += size;
	}
    
	// Then, copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely due to being evicted). skipped[]
	// records whether a block was skipped.
	off = (unsigned long)(dst - r->swp_addr);
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		ret = gmm_htod_block(r, off, s, size, iblock, 1,
                             skipped + (iblock - ifirst));
		if (ret != 0)
			goto finish;
		s += size;
		off += size;
	}
    
	// Finally, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->swp_addr);
	s = (void *)src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		unsigned long size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
			ret = gmm_htod_block(r, off, s, size, iblock, 0, NULL);
			if (ret != 0)
				goto finish;
		}
		s += size;
		off += size;
	}
    
finish:
	free(skipped);
	return ret;
}
#else
// The conservative version
static int gmm_htod(
                    struct region *r,
                    void *dst,
                    const void *src,
                    size_t count)
{
	unsigned long off, end, size;
	int ifirst, ilast, iblock;
	void *s = (void *)src;
	char *skipped;
	int ret = 0;
    
	gprint(DEBUG, "htod: r(%p %p %ld %d %d) dst(%p) src(%p) count(%lu)\n", \
           r, r->swp_addr, r->size, r->flags, r->state, dst, src, count);
    
	if (r->flags & HINT_PTARRAY)
		return gmm_htod_pta(r, dst, src, count);
    
	off = (unsigned long)(dst - r->swp_addr);
	end = off + count;
	ifirst = BLOCKIDX(off);
	ilast = BLOCKIDX(end - 1);
	skipped = (char *)calloc(ilast - ifirst + 1, sizeof(char));
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}
    
	// Copy data block by block, skipping blocks that are not available
	// for immediate operation (very likely due to being evicted).
	// skipped[] records whether each block was skipped.
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		ret = gmm_htod_block(r, off, s, size, iblock, 1,
                             skipped + (iblock - ifirst));
		if (ret != 0)
			goto finish;
		s += size;
		off += size;
	}
    
	// Then, copy the rest blocks, no skipping.
	off = (unsigned long)(dst - r->swp_addr);
	s = (void *)src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
			ret = gmm_htod_block(r, off, s, size, iblock, 0, NULL);
			if (ret != 0)
				goto finish;
		}
		s += size;
		off += size;
	}
    
finish:
	free(skipped);
	return ret;
}
#endif




static int gmm_clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t count, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event* event){
    
    struct dma_channel *chan = &pcontext->dma_htod;
    unsigned long off, delta;
    int ret=0, ilast;

    struct region * r;
    if (count<=0){
        return CL_INVALID_VALUE;
    }

    r = region_lookup(pcontext, dst);
    if(!r){
        gprint(ERROR,"cannot find the region containing %p in htod\n",dst);
        return CL_INVALID_MEM_OBJECT;
    }

    if(r->state== STATE_FREEING||r->STATE==ZOMBIE){
        gprint(WARN,"region already freed\n");
        return CL_INVALID_VALUE;
    }
    if (r->flags&& FLAG_COW){
        gprint(ERROR,"HtoD,the region is tagged on COW\n");
        return CL_INVALID_MEM_OBJECT;
    }
    if(dst+count > r->swp_addr + r->size){
        gprint(ERROR,"htod out of region boundary\n");
        return CL_INVALID_VALUE;
    }

    stats_time_begin();
    /*if(cow){

        stats_time_begin();
        if(gmm_)
        
    }
    else{*/
        if(gmm_htod(r,dst,src,count)<0)
            return CL_INVALID_MEM_OBJECT;
    //}
    stats_time_end(&pcontext->stats,time_htod);
    stats_inc(&pcontext->stats,bytes_htod, count);
    
    return CL_SUCCESS;
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
