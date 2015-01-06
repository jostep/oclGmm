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
static int gmm_launch(struct region **rgns, int nrgns,cl_kernel);
static int gmm_launch2(struct region **rgns, int nrgns,cl_kernel,cl_uint,const size_t*,const size_t*,const size_t*,cl_event*);
static int gmm_attach(struct region **rgns, int n);
static int gmm_load(struct region **rgns, int nrgns);
static int gmm_evict(long size_needed, struct region **excls, int nexcl);
extern cl_mem (*ocl_clCreateBuffer)(cl_context , cl_mem_flags, size_t, void *, cl_int*);
extern cl_int (*ocl_clReleaseMemObject)(cl_mem);
extern cl_context(*ocl_clCreateContext)(cl_context_properties * ,cl_uint ,const cl_device_id *,void*, void *,cl_int*);
//extern cl_command_queue (*ocl_clCreateCommandQueue)(cl_context, cl_device_id,cl_command_queue_properties,cl_int *);
extern cl_int (*ocl_clEnqueueFillBuffer)(cl_command_queue, cl_mem,const void * , size_t, size_t,size_t, cl_uint, const cl_event, cl_event);
void gmm_context_initEX();
static int gmm_memset(struct region *r, cl_mem buffer, int value, size_t count);
extern cl_int (*ocl_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void* , cl_uint, const cl_event *, cl_event *);
extern cl_int (*ocl_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void* , cl_uint, const cl_event *, cl_event *);
extern cl_int (*ocl_clEnqueueCopyBuffer)(cl_command_queue, cl_mem,cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
extern cl_int (*ocl_clBuildProgram)(cl_program program,cl_uint num_devices, const cl_device_id *devices_list,const char *options,void(*pfn_notify)(cl_program, void* user_data),void * user_data);
extern cl_program (*ocl_clCreateProgramWithSource)(cl_context context, cl_uint count, const char**strings, const size_t * lengths, cl_int *errcode_ret);
extern cl_int (*ocl_clEnqueueTask)(cl_command_queue, cl_kernel, cl_uint, const cl_event *,cl_event*);
extern cl_int (*ocl_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
extern cl_int (*ocl_clEnqueueNDRangeKernel)(cl_command_queue,cl_kernel,cl_uint,const size_t *, const size_t *, const size_t *,cl_uint,const cl_event*,cl_event*);



static int region_load_memset(struct region *r);
static int region_load(struct region *r);
static int region_load_cow(struct region *r);
static int region_load_pta(struct region *r);
static int region_attach(struct region *r,int pin,struct region **excls,int nexcl);



struct region * region_lookup(struct gmm_context *ctx, const cl_mem ptr);
static int dma_channel_init(struct gmm_context *,struct dma_channel *, int);
static void dma_channel_fini(struct dma_channel *chan);
struct gmm_context *pcontext=NULL;
static int block_sync(struct region *r, int block);
static int gmm_htod_block(struct region *r,unsigned long offset,const void *src,unsigned long size,int block,int skip,char *skipped);



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

static void end_dma(struct dma_channel *chan){
#ifdef GMM_CONFIG_DMA_ASYNC
    release(&chan->lock);
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
    cl_int * errcode_CQ=NULL; 
    pcontext->commandQueue_kernel=clCreateCommandQueue(pcontext->context_kernel,pcontext->device[0],CL_QUEUE_PROFILING_ENABLE,errcode_CQ);// add error handler;ERCI
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
    cl_context temp=ocl_clCreateContext(properties,num_devices,devices,pfn_notify,user_data,errcode_ret);
    pcontext->context_kernel=temp;
    if(*errcode_ret!=CL_SUCCESS){
        printf("Cannot create the Context %lu for the PC\n",*errcode_ret);
        
        if(errcode_ret==CL_INVALID_DEVICE)
            printf("Cannot create the Context due to %d : invalid device\n",errcode_ret);
    }

    gmm_context_initEX();//finishing the init process of gmm_context
    if(dma_channel_init(pcontext,&pcontext->dma_htod,1)!=0){
        gprint(FATAL,"failed to create HtoD DMA channel\n");
        free(pcontext);
        pcontext=NULL;
        return ;
    }    
    if(dma_channel_init(pcontext,&pcontext->dma_dtoh,0)!=0){
        gprint(FATAL,"failed to create DtoH DMA channel\n");
        dma_channel_fini(&pcontext->dma_htod);
        free(pcontext);
        pcontext=NULL;
        return ;
    }
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
    else if(size<0){
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
        gprint(FATAL,"malloc for a swap buffer: %s\n");
        free(r);
        return errcode_CB;
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
        gprint(DEBUG,"pta_addr (%p) alloced for %p\n",r->pta_addr,r);
    }
    
    nblocks= NRBLOCKS(size);
    r->blocks=(struct block*)calloc(nblocks,sizeof(struct block));
    if(!r->blocks){
        gprint(FATAL,"malloc failed for blocks array:%s\n",strerror(errno));
        if(r->pta_addr){
            free(r->pta_addr);
        }
        free(r->swp_addr);
        free(r);
        return NULL;
    }



    r->size= (long) size;
    r->state= STATE_DETACHED;
    r->flags=flags;
    r->gmm_flags=gmm_flags;
    r->dev_addr=NULL;
    list_alloced_add(pcontext, r);
    stats_inc_alloc(&pcontext->stats, size);
    
    gprint(DEBUG, "clCreateBuffer ends : r(%p) swp(%p) pta(%p)\n",r,r->swp_addr,r->pta_addr);
    
    if(errcode_CB)
        *errcode_CB=CL_SUCCESS;
    
    return (cl_mem)r->swp_addr; 
}

static int dma_channel_init(struct gmm_context *ctx,struct dma_channel *chan, int htod){
   int ret =0;
   cl_int * errcode_DMA=NULL;
   cl_int * errcode_INIT=NULL;
#ifdef GMM_CONFIG_DMA_ASYNC
   int i;
#endif 

   chan->commandQueue_chan=clCreateCommandQueue(ctx->context_kernel,ctx->device[0],CL_QUEUE_PROFILING_ENABLE,errcode_INIT);//
   if(errcode_INIT!=CL_SUCCESS){
        gprint(FATAL,"failed to create channel\n");
   }

#ifdef GMM_CONFIG_DMA_ASYNC
   initlock(&chan->lock);
   chan->ibuf=0;
   
   for (i=0;i<NBUFS;i++){
        chan->stage_bufs[i]=ocl_clCreateBuffer(ctx->context_kernel,CL_MEM_READ_WRITE,BUFSIZE,NULL,errcode_DMA);
        
        if(errcode_DMA!=CL_SUCCESS){
            gprint(FATAL,"failed for staging buffer\n");
            break;
        }
        errcode_DMA=NULL;
        chan->events[i]=clCreateUserEvent(pcontext->context_kernel,errcode_DMA);
        if(errcode_DMA!=CL_SUCCESS){
            gprint(FATAL,"failed to create for staging buffer\n");
            ocl_clReleaseMemObject(chan->stage_bufs[i]);
            break;
        }
        if(CL_SUCCESS!=clSetUserEventStatus(chan->events[i],CL_COMPLETE)){
            gprint(FATAL,"failed to set DMA event status\n");
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
        
        if(CL_SUCCESS!=clReleaseEvent(chan->events[i])){
            gprint(FATAL,"unable to release event of DMA Channal\n");
        }
       if(CL_SUCCESS!=ocl_clReleaseMemObject(chan->stage_bufs[i])){
            gprint(FATAL,"unable to release memObj of DMA channal\n");
       }
    }
#endif 
    clReleaseCommandQueue(chan->commandQueue_chan);

}

cl_int gmm_clReleaseMemObject(cl_mem memObjPtr){
    
    struct region *r;
    if (!(r= region_lookup(pcontext, memObjPtr))){
        //free(r->blocks);
        gprint(ERROR,"cannot find region containg %p in clReleaseMemObject\n",memObjPtr);
        return CL_INVALID_MEM_OBJECT;
    }
    stats_inc_freed(&pcontext->stats,r->size);
    if(gmm_free(r)<0)
        return CL_INVALID_MEM_OBJECT;
    return CL_SUCCESS;
}

struct region * region_lookup(struct gmm_context *ctx, const cl_mem ptr){
        
    struct region *r = NULL;
    struct list_head *pos;
    int found =0;
    unsigned long wa=0x0000ffff;
    unsigned long start,end;
    acquire (&ctx->lock_alloced);
    list_for_each(pos, &ctx->list_alloced){
        r= list_entry(pos, struct region, entry_alloced);
        start=(unsigned long)r->swp_addr;
        end=(unsigned long)start+r->size;
        printf("the start addr will be(%p) (%lu) , end(%p)(%lu) ",start,start,end,end);
        if (r->state==STATE_FREEING||r->state==STATE_ZOMBIE){
            continue;
        }
        if(start<=(unsigned long)(ptr)&&end>(unsigned long)ptr){
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
    gprint(DEBUG,"freeing r (r %p|| swp_addr%p|| size %ld||flags %d||r-state %d)\n",r,r->swp_addr,r->size,r->gmm_flags,r->state);
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
    //gprint(DEBUG,"Before releasing content in r, the ptr are here %p,%p,%p\n",r->pta_addr,r->swp_addr,r->dev_addr);
    if(r->blocks)
        free(r->blocks);
    if(r->pta_addr)
        free(r->pta_addr);
    if(r->swp_addr)
        free(r->swp_addr);
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
    if(r->gmm_flags &&FLAG_COW){
        gprint(ERROR,"region tagged CoW\n");
        return CL_INVALID_MEM_OBJECT;
    }
    if((unsigned long)buffer+(unsigned long)size >(unsigned long)r->swp_addr+r->size){
        gprint(ERROR,"size overloaded\n");
        return CL_INVALID_VALUE;
    }
    if(r->size!=size){
        gprint(ERROR,"should set the whole region\n");
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

static int gmm_memset_pta(struct region * r, cl_mem dst, int value, size_t count){
    unsigned long off=(unsigned long)dst - (unsigned long)r->swp_addr;
    memset((void*)((unsigned long)r->pta_addr+off),value,count);
    return 0;
}

static int gmm_memset(struct region *r, cl_mem buffer,int value, size_t count){
    
    unsigned long off,end,size;
    int ifirst, ilast, iblock;
    char *skipped;
    int ret=0;
    void *s;

    gprint(DEBUG,"memset: r(%p,%p,%ld)dst(%p)value(%d)count(%lu)\n",\
           r,r->swp_addr,r->size,buffer,value,count);
    if(r->gmm_flags && HINT_PTARRAY){
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

    memset(s,value,BLOCKSIZE);
    off=(unsigned long)buffer-(unsigned long)r->swp_addr;
    end = off+count;
    ifirst=BLOCKIDX(off);
    ilast=BLOCKIDX(end-1);
    skipped=(char *)malloc(ilast-ifirst+1);
    if(!skipped){
        gprint(FATAL,"malloc failed for skipped %s\n",strerror(errno));
        free(s);
        return -1;
    }

    for(iblock=ifirst;iblock<=ilast;iblock++){
        size=MIN(BLOCKUP(off),end)-off;
        ret=gmm_htod_block(r,off,s,size,iblock,1,skipped+(iblock-ifirst));
        if(ret!=0)
            goto finish;
        off +=size;
    }

    off=(unsigned long)buffer -(unsigned long)r->swp_addr;

    for(iblock=ifirst;iblock<=ilast;iblock++){
        size=MIN(BLOCKUP(off),end)-off;
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

static int gmm_memcpy_dtoh(void* dst, cl_mem src, unsigned long size,int prev_off)
{
	struct dma_channel *chan = &pcontext->dma_dtoh;
	unsigned long	off_dtos,	// Device to Stage buffer
    off_stoh,	// Stage buffer to Host buffer
    delta;
	int ret = 0, ibuf_old;
    cl_int  err;
    cl_int errcode_temp;
	begin_dma(chan);
    cl_event temp[NBUFS];
    int i=0;
    while(i<NBUFS){
        temp[i]=clCreateUserEvent(pcontext->context_kernel,&errcode_temp);
        if(errcode_temp!=CL_SUCCESS){
            gprint(FATAL,"Failed to Create user event in HTOD\n");
        }
        if(CL_SUCCESS!=clSetUserEventStatus(temp[i],CL_COMPLETE)){
            gprint(FATAL,"Failed to Set user event in HTOD\n");
        }
        i++;
    }
    
	// First issue DtoH commands for all staging buffers
	ibuf_old = chan->ibuf;
	off_dtos = 0;
	while (off_dtos < size && off_dtos < NBUFS * BUFSIZE) {
		delta = MIN(off_dtos + BUFSIZE, size) - off_dtos;
        if(ocl_clEnqueueCopyBuffer(chan->commandQueue_chan,src,chan->stage_bufs[chan->ibuf],off_dtos+prev_off,0,delta,0,NULL,&temp[chan->ibuf])!=CL_SUCCESS){
        
			gprint(FATAL,"Copy Buffer failed in dtoh\n");
			ret = -1;
			goto finish;
		}
		/*if (clEnqueueMarker(chan->commandQueue_chan,&chan->events[chan->ibuf])!= CL_SUCCESS) {
			gprint(FATAL,"opencl marker failed in dtoh\n");
			ret = -1;
			goto finish;
		}*/
        
		chan->ibuf = (chan->ibuf + 1) % NBUFS;
		off_dtos += delta;
	}
    
    /*if(CL_SUCCESS!=clFinish(chan->commandQueue_chan)){
        gprint(FATAL,"FIRST dtoh waiting failed\n");
    }*/
	// Now copy data to user buffer, meanwhile issuing the
	// rest DtoH commands if any.
	chan->ibuf = ibuf_old;
	off_stoh = 0;
    /*cl_event temp[NBUFS];
    while(i<NBUFS){
        temp[i]=clCreateUserEvent(pcontext->context_kernel,&errcode_temp);
        if(errcode_temp!=CL_SUCCESS){
            gprint(FATAL,"Failed to Create user event in HTOD\n");
        }
        if(CL_SUCCESS!=clSetUserEventStatus(temp[i],CL_COMPLETE)){
            gprint(FATAL,"Failed to Set user event in HTOD\n");
        }
        i++;
    }*/
	while (off_stoh < size) {
		delta = MIN(off_stoh + BUFSIZE, size) - off_stoh;
        
		/*err=clWaitForEvents(1,&chan->events[chan->ibuf]);
        if(err!=CL_SUCCESS){
            
			gprint(FATAL,"openCL Event Synchronize failed %d in dtoh\n",err);
            if(err==CL_INVALID_CONTEXT){
                gprint(DEBUG,"reason is invalid context\n");
            }
            else if(err==CL_INVALID_VALUE){
                gprint(DEBUG,"reason is invalid value\n");
            }
            else if(err==CL_INVALID_EVENT)
                gprint(DEBUG,"reason is invalid event\n");
            else{
                gprint(DEBUG,"I have no fucking clue\n");
            }
			ret = -1;
			goto finish;
		}*/
        cl_int err_RB;
        err_RB=ocl_clEnqueueReadBuffer(chan->commandQueue_chan,chan->stage_bufs[chan->ibuf],CL_TRUE,0,delta,(void*)(unsigned long)dst+off_stoh+prev_off,1,&temp[chan->ibuf],NULL);
        if(err_RB!=CL_SUCCESS){
            gprint(FATAL,"cannot copy to dst with err %d\n",err_RB);
            ret=-1;
            goto finish;
        }
		off_stoh += delta;
        
		if (off_dtos < size) {
			delta = MIN(off_dtos + BUFSIZE, size) - off_dtos;
            if(ocl_clEnqueueCopyBuffer(chan->commandQueue_chan,src,chan->stage_bufs[chan->ibuf],off_dtos+prev_off,0,delta,0,NULL,NULL)){
				gprint(FATAL, "openCL memcpy Async failed in dtoh\n");
				ret = -1;
				goto finish;
			}
		    /*if (clEnqueueMarker(chan->commandQueue_chan,&chan->events[chan->ibuf])!= CL_SUCCESS){
				gprint(FATAL,"opencl marker failed in dtoh\n");
				ret = -1;
				goto finish;
			}*/
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
        const void * src,
        size_t count){

    unsigned long off= (unsigned long)dst- (unsigned long)r->swp_addr;

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
    memcpy((void*)((unsigned long)r->pta_addr+off), src, count);
    stats_time_end(&pcontext->stats,time_u2s);
    stats_inc(&pcontext->stats,bytes_u2s,count);

    return 0;
}

static int gmm_memcpy_htod(cl_mem dst,const void * src, unsigned long size,int prev_off){

    struct dma_channel *chan = &pcontext->dma_htod;
    unsigned long off,delta;
    int ret=0, ilast;
    cl_int *debug=NULL;
    cl_int status;
    cl_int*errcode_CB=NULL;
    cl_int errcode_EX;
    cl_int errcode_temp;
    cl_event temp=clCreateUserEvent(pcontext->context_kernel,&errcode_temp);
    if(errcode_temp!=CL_SUCCESS){
        gprint(FATAL,"Failed to Create user event in HTOD\n");
    }
    if(CL_SUCCESS!=clSetUserEventStatus(temp,CL_COMPLETE)){
        gprint(FATAL,"Failed to Set user event in HTOD\n");
    }
    begin_dma(chan);
    off=0;
    while(off<size){
        delta=MIN(off+BUFSIZE, size)-off;
        /*errcode_EX=clFinish(chan->commandQueue_chan);
        if(errcode_EX!=CL_SUCCESS){
            gprint(FATAL,"unsuccessfully waiting with err: %d \n",errcode_EX);
        }
        if(CL_SUCCESS!=clGetEventInfo(chan->events[chan->ibuf],CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(cl_int),&status, NULL)){
            gprint(DEBUG,"unable to get the event status for htod\n");
        }
         if(status!=CL_SUBMITTED&&status!=CL_COMPLETE){
            debug=clWaitForEvents(1,&chan->events[chan->ibuf]);
            if(debug!=CL_SUCCESS){
                gprint(FATAL,"sync failed in htod and the err is%d\n",debug);
                ret=-1;
                goto finish;
            }
            gprint(DEBUG,"not bypassed for %d",chan->ibuf);
        }*/
        //clWaitForEvents(1,&chan->events[chan->ibuf]);
        errcode_CB=ocl_clEnqueueWriteBuffer(chan->commandQueue_chan,chan->stage_bufs[chan->ibuf],CL_FALSE,0,delta,(src+off+prev_off),1,&chan->events[chan->ibuf],&temp);
        if(errcode_CB!=CL_SUCCESS){
            gprint(DEBUG,"copy buffer failure\n");
            switch((int)errcode_CB){
                case CL_INVALID_MEM_OBJECT:
                    gprint(DEBUG,"invalid mem object\n");
                    break;
                case CL_INVALID_VALUE:
                    gprint(DEBUG,"invalid mem object\n");
                    break;
                default:
                    gprint(DEBUG,"else\n");
                    break; 
            }
        }
        errcode_CB=ocl_clEnqueueCopyBuffer(chan->commandQueue_chan,chan->stage_bufs[chan->ibuf],dst,0,off+prev_off,delta,1,&temp,&chan->events[chan->ibuf]);
        if(errcode_CB!=CL_SUCCESS){
            if(errcode_CB==CL_INVALID_COMMAND_QUEUE){
                gprint(FATAL,"invalid command queue\n");
            }
            else if(errcode_CB==CL_INVALID_MEM_OBJECT){
                gprint(FATAL,"invalid mem obect\n");
            }
            else if(errcode_CB==CL_MEM_OBJECT_ALLOCATION_FAILURE){
                gprint(FATAL,"mem alloc is incorrect,with mem_used (%ld) and mem_total (%ld)\n",memsize_getUsed(),memsize_total());
            
            }
            else if(errcode_CB==CL_INVALID_VALUE){
                gprint(FATAL,"invalid value\n");
            
            }
            else
                gprint(DEBUG,"reason unknown\n");
            gprint(FATAL,"cl write buffer failed in htod\n");
            ret=-1;
            goto finish;
        }

        /*if(clEnqueueMarker(chan->commandQueue_chan,&chan->events[chan->ibuf])!=CL_SUCCESS){
            gprint(FATAL,"cl marker failed in htod\n");
            ret=-1;
            goto finish;
        }*/
        ilast=chan->ibuf;
        chan->ibuf=(chan->ibuf+1)%NBUFS;
        off+=delta;
        
    }
    if(clEnqueueMarker(chan->commandQueue_chan,&chan->events[ilast])!=CL_SUCCESS){
            gprint(FATAL,"cl marker failed in htod\n");
            ret=-1;
            goto finish;
    }
    /*cl_int errcode_WFE;
    errcode_WFE=clWaitForEvents(1,&chan->events[ilast]);
    if(errcode_WFE!=CL_SUCCESS){
        gprint(FATAL,"cl event sync failed in htod with err %d\n",errcode_WFE);
        ret=-1;
    }*/
    errcode_EX=clFinish(chan->commandQueue_chan);
    if(errcode_EX!=CL_SUCCESS){
        gprint(FATAL,"unsuccessfully waiting with err: %d \n",errcode_EX);
        ret=-1;
    }


finish:
    end_dma(chan);
    return ret;
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
		memcpy((void *)((unsigned long )r->swp_addr + offset), src, size);
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
		memcpy((void *)((unsigned long)r->swp_addr + offset), src, size);
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
		ret = gmm_memcpy_dtoh((void*)((unsigned long)r->swp_addr),
                (cl_mem)((unsigned long)r->dev_addr), size,off);
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
		ret = gmm_memcpy_htod(r->dev_addr,r->swp_addr, size,off);
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
                    size_t count,size_t dst_off)
{
	int iblock, ifirst, ilast;
	unsigned long off, end;
	void *s = (void *)src;
	char *skipped;
	int ret = 0;
    
	if (r->gmm_flags & HINT_PTARRAY)
		return gmm_htod_pta(r, dst, src, count);
    
	off = (unsigned long)dst -(unsigned long)r->swp_addr+dst_off;
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
		if ((off % BLOCKSIZE) == 0 &&
			(size == BLOCKSIZE || (off + size) == r->size)) {// a change by ERCI offset->off
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
	off = (unsigned long)dst -(unsigned long) r->swp_addr+dst_off;
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
	off = (unsigned long)dst - (unsigned long)r->swp_addr+dst_off;
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
                    cl_mem dst,
                    const void *src,
                    size_t count,size_t dst_off)
{
	unsigned long off, end, size;
	int ifirst, ilast, iblock;
	void *s = (void *)src;
	char *skipped;
	int ret = 0;
    
	gprint(DEBUG, "htod: r(%p %p %ld %d %d) dst(%p) src(%p) count(%lu)\n", \
           r, r->swp_addr, r->size, r->gmm_flags, r->state, dst, src, count);
    
	if (r->gmm_flags & HINT_PTARRAY)
		return gmm_htod_pta(r, dst, (void *)src, count);
	off = (unsigned long)dst -(unsigned long)r->swp_addr+dst_off;
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
	off = (unsigned long)dst -(unsigned long)r->swp_addr+dst_off;
	s = (void *)src;
	for (iblock = ifirst; iblock <= ilast; iblock++) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
            gprint(DEBUG,"we have skipped\n");
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


static int gmm_dtod(struct region *rd,struct region *rs,cl_mem dst,cl_mem src,size_t count,size_t dst_off,size_t src_off)
{
	int ret = 0;
	void *temp;
    
	gprint(DEBUG, "dtod: rd(%p %p %ld %d %d) rs(%p %p %ld %d %d) " \
           "dst(%p) src(%p) count(%lu)\n", \
           rd, rd->swp_addr, rd->size, rd->gmm_flags, rd->state, \
           rs, rs->swp_addr, rs->size, rs->gmm_flags, rs->state, \
           dst, src, count);
    
	temp = malloc(count);
	if (!temp) {
		gprint(FATAL, "malloc failed for temporary dtod buffer: %s\n", \
               strerror(errno));
		return -1;
	}
    
	if (gmm_dtoh(rs, temp, src, count,src_off) < 0) {
		gprint(ERROR, "failed to copy data to temp dtod for dtoh buffer\n");
		free(temp);
		return -1;
	}
    
	if (gmm_htod(rd, dst, temp, count,dst_off) < 0) {
		gprint(ERROR, "failed to copy data from temp htod  buffer\n");
		free(temp);
		return -1;
	}
    
	free(temp);
	return ret;
}


cl_int  gmm_clEnqueueCopyBuffer(cl_command_queue command_queue,cl_mem src,cl_mem dst,size_t src_off,size_t dst_off,size_t count,cl_uint num_events_in_wait_list,const cl_event* event_wait_list,cl_event* event)
{
	struct region *rs, *rd;
         
	if (count <= 0)
		return CL_INVALID_VALUE;
    
	rs = region_lookup(pcontext, src);
	if (!rs) {
		gprint(ERROR, "cannot find src region containing %p in dtod\n", src);
		return CL_INVALID_MEM_OBJECT;
	}
	if (rs->state == STATE_FREEING || rs->state == STATE_ZOMBIE) {
		gprint(ERROR, "src region already freed\n");
		return CL_INVALID_VALUE;
	}
	if (rs->gmm_flags & FLAG_COW) {
		gprint(ERROR, "dtod: src region already tagged COW\n");
		return CL_INVALID_MEM_OBJECT;
	}
	if ((unsigned long)src + count > (unsigned long)rs->swp_addr + rs->size) {
		gprint(ERROR, "dtod out of src boundary\n");
		return CL_INVALID_VALUE;
	}
    
	rd = region_lookup(pcontext, dst);
	if (!rd) {
		gprint(ERROR, "cannot find dst region containing %p in dtod\n", dst);
		return CL_INVALID_MEM_OBJECT;
	}
	if (rd->state == STATE_FREEING || rd->state == STATE_ZOMBIE) {
		gprint(ERROR, "dst region already freed\n");
		return CL_INVALID_VALUE;
	}
	if (rd->gmm_flags & FLAG_COW) {
		gprint(ERROR, "dtod: dst region already tagged COW\n");
		return CL_INVALID_MEM_OBJECT;
	}
	if ((unsigned long)dst + count > (unsigned long)rd->swp_addr + rd->size) {
		gprint(ERROR, "dtod out of dst boundary\n");
		return CL_INVALID_VALUE;
	}
    
	stats_time_begin();
	if (gmm_dtod(rd, rs, dst, src, count,dst_off,src_off) < 0)
		return CL_INVALID_VALUE;
	stats_time_end(&pcontext->stats, time_dtod);
	stats_inc(&pcontext->stats, bytes_dtod, count);
    
    //clSetUserEventStatus(*event,CL_COMPLETE);    
	return CL_SUCCESS;
}





cl_int gmm_clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem dst, cl_bool blocking_write, size_t offset, size_t count, const void *src, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event* event){
    

    struct region * r;
    if (count<=0){
        return CL_INVALID_VALUE;
    }

    r = region_lookup(pcontext, dst);
    if(!r){
        gprint(ERROR,"cannot find the region containing %p in htod\n",dst);
        return CL_INVALID_MEM_OBJECT;
    }

    if(r->state== STATE_FREEING||r->state==STATE_ZOMBIE){
        gprint(WARN,"region already freed\n");
        return CL_INVALID_VALUE;
    }
    if (r->gmm_flags & FLAG_COW){
        gprint(ERROR,"HtoD,the region is tagged on COW\n");
        return CL_INVALID_MEM_OBJECT;
    }
    if((unsigned long)dst+count > (unsigned long)r->swp_addr + r->size){
        gprint(ERROR,"htod out of region boundary\n");
        return CL_INVALID_VALUE;
    }

    stats_time_begin();
   /*if(cow){

        stats_time_begin();
        if(gmm_htod_cow(r,dst,src,count)<0){
            gprint(FATAL,"reasons unknown\n");
            return CL_INVALID_VALUE;
        }
        stats_time_end(&pcontext->stats,time_htod_cow);
        stats_inc(&pcontext->stats,bytes_htod_cow, count);
            
    }
    else{*/
            if(gmm_htod(r,dst,src,count,0)<0)
                return CL_INVALID_MEM_OBJECT;
    // }
    stats_time_end(&pcontext->stats,time_htod);
    stats_inc(&pcontext->stats,bytes_htod, count);
    return CL_SUCCESS;
}

cl_program gmm_clCreateProgramWithSource(cl_context context, cl_uint count, const char**strings, const size_t * lengths, cl_int *errcode_ret){

        if(context!=pcontext->context_kernel){
            gprint(WARN,"different context\n");
            return NULL;
        }
        pcontext->program_kernel=ocl_clCreateProgramWithSource(context,count,strings,lengths,errcode_ret);
        if(*errcode_ret!=CL_SUCCESS){
            gprint(FATAL,"unable to create program\n");        
        }
        return pcontext->program_kernel;
}

cl_int gmm_clBuildProgram(cl_program program,cl_uint num_devices, const cl_device_id *devices_list,const char *options,void(*pfn_notify)(cl_program, void* user_data),void * user_data){
    
    if(program!=pcontext->program_kernel){
        gprint(FATAL,"unmatched program\n");
        return CL_INVALID_PROGRAM;
    }
    if(num_devices!=1){
        gprint(FATAL,"only support 1 device at the moment\n");
        return CL_INVALID_DEVICE;
    }
    if(pcontext->device!=devices_list){
        gprint(FATAL,"unmatched devices\n");
        return CL_INVALID_DEVICE;
    }
    if(ocl_clBuildProgram(program,num_devices,devices_list,options,pfn_notify,user_data)!=CL_SUCCESS){
        gprint(FATAL,"unable to build program\n");
        return CL_BUILD_PROGRAM_FAILURE;
    }
    return CL_SUCCESS;
}
/*
cl_kernel gmm_clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret){
        
        cl_int * errcode_CK=NULL;
        if(program!=pcontext->program_kernel){
            gprint(FATAL,"unmatched kernel\n");
            errcode_ret=CL_INVALID_PROGRAM;
            return NULL;
        }
        pcontext->kernel=ocl_clCreateKernel(program,kernel_name,errcode_CK);
        if(errcode_CK!=CL_SUCCESS){
            errcode_ret=errcode_CK;
            return NULL;
        }
        return pcontext->kernel;

}
*/

/*
cl_int gmm_clSetKernelArg(cl_kernel kernel,cl_uint arg_index,size_t arg_size, const void* arg_value){

        struct region *r;
        int is_dptr=0;
        int iref=0;
        if(kernel!=pcontext->kernel){
            gprint(FATAL,"invalid kernel\n");
            return CL_INVALID_KERNEL;
        }
        gprint(DEBUG,"openCL setup argument nargs(%d), size(%lu) offset(%lu)\n",nargs,arg_size,arg_index);


}
*/

extern int refs[NREFS];
extern int rwflags[NREFS];
extern int nrefs;

static unsigned char kstack[512];
static void *ktop=(void *)kstack;
static struct karg kargs[NREFS];
static int nargs=0;

cl_int gmm_clSetKernelArg(cl_kernel kernel,cl_uint offset,size_t size, const void* arg){
	
    struct region *r;
	int is_dptr = 0;
	int iref = 0;

    /*
    if(kernel!=pcontext->kernel){
            gprint(FATAL,"invalid kernel\n");
            return CL_INVALID_KERNEL;
    }
    */
	gprint(DEBUG, "opencl Setup Argument: nargs(%d) size(%lu) offset(%lu)\n", \
           nargs, size, offset);
    
	// Test whether this argument is a device memory pointer.
	// If it is, record it and postpone its pushing until cudaLaunch.
	// Use reference hints if given. Otherwise, parse automatically
	// (but parsing errors are possible, e.g., when the user passes a
	// long argument that happen to lay within some region's host swap
	// buffer area).
	if (nrefs > 0) {
		for (iref = 0; iref < nrefs; iref++) {
			if (refs[iref] == nargs)
				break;
		}
		if (iref < nrefs) {
			if (size != sizeof(void *)) {
				gprint(ERROR, "argument size (%lu) does not match dptr " \
                       "ocl Reference (%d)\n", size, nargs);
				return CL_INVALID_ARG_SIZE;
				//panic("cudaSetupArgument does not match cudaReference");
			}
            if(arg){
			    r = region_lookup(pcontext, *(cl_mem*)arg);
            }
            else{
		        kargs[nargs].arg.arg1.dptr = arg;
	            kargs[nargs].is_dptr = 2;
	            kargs[nargs].size = size;
	            kargs[nargs].argoff = offset;
	            nargs++;
                return CL_SUCCESS;
            }
			if (!r) {
				gprint(ERROR, "cannot find region containing %p (%d) in " \
                       "clSetkernel\n", *(cl_mem*)arg, nargs);
				return CL_INVALID_ARG_SIZE;
				//panic("region_lookup in cudaSetupArgument");
			}
			is_dptr = 1;
		}
	}
	// TODO: we should assume all memory regions are to be referenced
	// if no reference hints are given.
	else if (size == sizeof(void *)) {
		gprint(WARN, "trying to parse dptr argument automatically\n");
		r = region_lookup(pcontext, *(cl_mem*)arg);
		if (r)
			is_dptr = 1;
	}
    
	if (is_dptr) {
		kargs[nargs].arg.arg1.r = r;
		kargs[nargs].arg.arg1.off =
        (unsigned long)(*(cl_mem*)arg) - (unsigned long)r->swp_addr;
		if (nrefs > 0)
			kargs[nargs].arg.arg1.flags = rwflags[iref];
		else
			kargs[nargs].arg.arg1.flags = HINT_DEFAULT | HINT_PTADEFAULT;
		gprint(DEBUG, "argument is dptr: r(%p %p %ld %d %d)\n", \
               r, r->swp_addr, r->size, r->gmm_flags, r->state);
	}
	else {
		// This argument is not a device memory pointer.
		// XXX: Currently we ignore the case that nv_cudaSetupArgument
		// returns error and CUDA runtime might stop pushing arguments.
        void * temp = malloc(size);
        /*void * temp = malloc(size);
        if(arg)
            memcpy(temp,arg,size);
        else
            temp=arg;
		kargs[nargs].arg.arg2.arg = temp;
        memcpy(ktop, arg, size);
		kargs[nargs].arg.arg2.arg = ktop;
        printf("for a non-dptr your addr will be %p \n",temp);
=======
		*/
        if(arg)
            memcpy(ktop, arg, size);
        else 
            ktop=arg;
        kargs[nargs].arg.arg2.arg = ktop;
		ktop += size;
	}
	kargs[nargs].is_dptr = is_dptr;
	kargs[nargs].size = size;
	kargs[nargs].argoff = offset;
    
	nargs++;
	return CL_SUCCESS;
}


static long regions_referenced(struct region ***prgns, int *pnrgns)
{
	struct region **rgns, *r;
	long total = 0;
	int nrgns = 0;
	int i;
    
	if (nrefs > NREFS || nrefs <= 0)
		panic("nrefs");
	if (nargs <= 0 || nargs > NREFS)
		panic("nargs");
    
	// Get the upper bound of the number of unique regions.
	for (i = 0; i < nargs; i++) {
		if (kargs[i].is_dptr) {
			nrgns++;
			r = kargs[i].arg.arg1.r;
			// Here we assume at most one level of dptr arrays
			if (r->gmm_flags & HINT_PTARRAY)
				nrgns += r->size / sizeof(void *);
		}
	}
	if (nrgns <= 0)
		panic("nrgns");
    
	rgns = (struct region **)malloc(sizeof(*rgns) * nrgns);
	if (!rgns) {
		gprint(FATAL, "malloc failed for region array: %s\n", strerror(errno));
		return -1;
	}
	nrgns = 0;
    
	// Now set the regions to be referenced.
	for (i = 0; i < nargs; i++) {
		if (kargs[i].is_dptr) {
			r = kargs[i].arg.arg1.r;
			if (!is_included((void **)rgns, nrgns, (void*)r)) {
				gprint(DEBUG, "new referenced region(%p %p %ld %d %d) " \
                       "off(%lu)\n", r, r->swp_addr, r->size, r->gmm_flags, \
                       r->state, kargs[i].arg.arg1.off);
				rgns[nrgns++] = r;
				r->rwhint.flags = kargs[i].arg.arg1.flags & HINT_MASK;
				total += r->size;
                
#ifdef GMM_CONFIG_RW
				// Make sure cow region is read-only
				if (r->gmm_flags & FLAG_COW)
					r->rwhint.flags = HINT_READ;
#endif
                
				if (r->gmm_flags & HINT_PTARRAY) {
					void **pdptr = (void **)(r->pta_addr);
					void **pend = (void **)((unsigned long)r->pta_addr + r->size);
					r->rwhint.flags = HINT_READ;	// dptr array is read-only
					// For each device memory pointer contained in this region
					while (pdptr < pend) {
						r = region_lookup(pcontext, (cl_mem)*pdptr);
						if (!r) {
							gprint(WARN, "cannot find region for dptr " \
                                   "%p (%d)\n", *pdptr, i);
							pdptr++;
							continue;
						}
						if (!is_included((void **)rgns, nrgns, (void*)r)) {
							gprint(DEBUG, "\tnew referenced region" \
                                   "(%p %p %ld %d %d) off(%lu)\n", \
                                   r, r->swp_addr, r->size, r->gmm_flags, \
                                   r->state, kargs[i].arg.arg1.off);
							rgns[nrgns++] = r;
							r->rwhint.flags =
                            ((kargs[i].arg.arg1.flags & HINT_PTAREAD) ?
                             HINT_READ : 0) |
                            ((kargs[i].arg.arg1.flags & HINT_PTAWRITE) ?
                             HINT_WRITE : 0);
							total += r->size;
						}
						else {
							gprint(DEBUG, "\told referenced region" \
                                   "(%p %p %ld %d %d) off(%lu)\n", r, \
                                   r->swp_addr, r->size, r->gmm_flags, \
                                   r->state, kargs[i].arg.arg1.off);
							r->rwhint.flags |=
                            ((kargs[i].arg.arg1.flags & HINT_PTAREAD) ?
                             HINT_READ : 0) |
                            ((kargs[i].arg.arg1.flags & HINT_PTAWRITE) ?
                             HINT_WRITE : 0);
						}
						// Make sure cow region is read-only
						if (r->gmm_flags & FLAG_COW)
							r->rwhint.flags = HINT_READ;
						pdptr++;
					}
				}
			}
			else {
				gprint(DEBUG, "old referenced region" \
                       "(%p %p %ld %d %d) off(%lu)\n", r, r->swp_addr, \
                       r->size, r->gmm_flags, r->state, kargs[i].arg.arg1.off);
				r->rwhint.flags |= kargs[i].arg.arg1.flags & HINT_MASK;
#ifdef GMM_CONFIG_RW
				// Make sure cow region is read-only
				if (r->gmm_flags & FLAG_COW)
					r->rwhint.flags = HINT_READ;
#endif
			}
		}
	}
    
	*pnrgns = nrgns;
	if (nrgns > 0)
		*prgns = rgns;
	else {
		free(rgns);
		*prgns = NULL;
	}
    
	return total;
}


cl_int gmm_clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event * event_in_wait_list, cl_event *event)
{
	cl_int ret = CL_SUCCESS;
	struct region **rgns = NULL;
	int nrgns = 0;
	long total = 0;
	int i, ldret;
    
	gprint(DEBUG, "openCL is about to launch\n");
    
	// NOTE: it is possible that nrgns == 0 when regions_referenced
	// returns. Consider a kernel that only uses registers, for
	// example.
	total = regions_referenced(&rgns, &nrgns);
	if (total < 0) {
		gprint(ERROR, "failed to get the regions to be referenced\n");
		ret = CL_INVALID_ARG_SIZE;
		goto finish;
	}
	else if (total > memsize_total()) {
		gprint(ERROR, "kernel requires too much space (%ld)\n", total);
		ret = CL_INVALID_ARG_SIZE;
		goto finish;
	}
    
reload:
	stats_time_begin();
	launch_wait();
	stats_time_end(&pcontext->stats, time_sync);
	stats_time_begin();
	ldret = gmm_attach(rgns, nrgns);
	stats_time_end(&pcontext->stats, time_attach);
	launch_signal();
	if (ldret > 0) {	// attach unsuccessful, retry later
		//stats_inc(&pcontext->stats, num_attach_fail, 1);
		sched_yield();
		goto reload;
	}
	else if (ldret < 0) {	// fatal load error, quit launching
		gprint(ERROR, "attach failed; quitting kernel launch\n");
		ret = CL_INVALID_ARG_SIZE;
		goto finish;
	}
    
	// Transfer data to device memory if they are to be read, and
	// handle WRITE hints. Note that, by this moment, all regions
	// pointed by each dptr array has been attached and pinned to
	// device memory.
	// XXX: What if the launch below failed? Partial modification?
	stats_time_begin();
	ldret = gmm_load(rgns, nrgns);
	stats_time_end(&pcontext->stats, time_load);
	if (ldret < 0) {
		gprint(ERROR, "gmm_load failed\n");
		for (i = 0; i < nrgns; i++)
			region_unpin(rgns[i]);
		ret = CL_INVALID_VALUE;
		goto finish;
	}
    
	// Configure and push all kernel arguments.
    /*
	if (nv_cudaConfigureCall(grid, block, shared, stream_issue)
        != cudaSuccess) {
		gprint(ERROR, "cudaConfigureCall failed\n");
		for (i = 0; i < nrgns; i++)
			region_unpin(rgns[i]);
		ret = cudaErrorUnknown;
		goto finish;
	}
    */
	for (i = 0; i < nargs; i++) {
		if (kargs[i].is_dptr==1) {
			kargs[i].arg.arg1.dptr =(cl_mem)((unsigned long)kargs[i].arg.arg1.r->dev_addr + kargs[i].arg.arg1.off);

            ocl_clSetKernelArg(kernel, kargs[i].argoff,kargs[i].size,&kargs[i].arg.arg1.dptr);
			 gprint(DEBUG, "setup device ptr %p %lu %lu\n", \
             &kargs[i].arg.arg1.dptr, \
             sizeof(void *), \
             kargs[i].arg.arg1.off);
		}
        else if (kargs[i].is_dptr==2){
            ocl_clSetKernelArg(kernel, kargs[i].argoff,kargs[i].size,&kargs[i].arg.arg1.dptr);
            gprint(DEBUG,"set up a null device ptr\n");
        }
		else {
            ocl_clSetKernelArg(kernel, kargs[i].argoff,kargs[i].size,kargs[i].arg.arg2.arg);
			/*gprint(DEBUG, "setup %p %lu %lu\n", \
             kargs[i].arg.arg2.arg, \
             kargs[i].arg.arg2.size, \
             kargs[i].arg.arg2.offset);*/
	         gprint(DEBUG, "setup non-ptr %p \n", \
             kargs[i].arg.arg2.arg);
		}
	}
    
	// Now we can launch the kernel.
	if (gmm_launch(rgns, nrgns,kernel) < 0) {
		for (i = 0; i < nrgns; i++)
			region_unpin(rgns[i]);
		ret = CL_INVALID_KERNEL;
	}
   // clSetUserEventStatus(*event,CL_COMPLETE);    
    
finish:
	if (rgns)
		free(rgns);
	nrefs = 0;
	nargs = 0;
	ktop = (void *)kstack;
	return ret;
}

cl_int gmm_clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel,cl_uint work_dim,const size_t *global_work_offset,const size_t * global_work_size,const size_t * local_work_size, cl_uint num_events_in_wait_list, const cl_event * event_in_wait_list, cl_event *event)
{
	cl_int ret = CL_SUCCESS;
	struct region **rgns = NULL;
	int nrgns = 0;
	long total = 0;
	int i, ldret;
    cl_int errcode_SK;
	gprint(DEBUG, "openCL is about to launch\n");
    
	// NOTE: it is possible that nrgns == 0 when regions_referenced
	// returns. Consider a kernel that only uses registers, for
	// example.
	total = regions_referenced(&rgns, &nrgns);
	if (total < 0) {
		gprint(ERROR, "failed to get the regions to be referenced\n");
		ret = CL_INVALID_ARG_SIZE;
		goto finish;
	}
	else if (total > memsize_total()) {
		gprint(ERROR, "kernel requires too much space (%ld)\n", total);
		ret = CL_INVALID_ARG_SIZE;
		goto finish;
	}
    
reload:
	stats_time_begin();
	launch_wait();
	stats_time_end(&pcontext->stats, time_sync);
	stats_time_begin();
	ldret = gmm_attach(rgns, nrgns);
	stats_time_end(&pcontext->stats, time_attach);
	launch_signal();
	if (ldret > 0) {	// attach unsuccessful, retry later
		//stats_inc(&pcontext->stats, num_attach_fail, 1);
		sched_yield();
		goto reload;
	}
	else if (ldret < 0) {	// fatal load error, quit launching
		gprint(ERROR, "attach failed; quitting kernel launch\n");
		ret = CL_INVALID_ARG_SIZE;
		goto finish;
	}
    
	// Transfer data to device memory if they are to be read, and
	// handle WRITE hints. Note that, by this moment, all regions
	// pointed by each dptr array has been attached and pinned to
	// device memory.
	// XXX: What if the launch below failed? Partial modification?
	stats_time_begin();
	ldret = gmm_load(rgns, nrgns);
	stats_time_end(&pcontext->stats, time_load);
	if (ldret < 0) {
		gprint(ERROR, "gmm_load failed\n");
		for (i = 0; i < nrgns; i++)
			region_unpin(rgns[i]);
		ret = CL_INVALID_VALUE;
		goto finish;
	}
    
	for (i = 0; i < nargs; i++) {
		if (kargs[i].is_dptr==1) {
			kargs[i].arg.arg1.dptr =(cl_mem)((unsigned long)kargs[i].arg.arg1.r->dev_addr + kargs[i].arg.arg1.off);
            errcode_SK=ocl_clSetKernelArg(kernel, kargs[i].argoff,kargs[i].size,&kargs[i].arg.arg1.dptr);
			if(errcode_SK!=CL_SUCCESS)
			    gprint(FATAL, "setup dptr with size%d  err:%d\n",kargs[i].size,errcode_SK);
		}
        else if (kargs[i].is_dptr==2){
            errcode_SK=ocl_clSetKernelArg(kernel, kargs[i].argoff,kargs[i].size,&kargs[i].arg.arg1.dptr);
			if(errcode_SK!=CL_SUCCESS)
                gprint(DEBUG,"set up a null device ptr\n");
        }
		else {
            errcode_SK=ocl_clSetKernelArg(kernel, kargs[i].argoff,kargs[i].size,kargs[i].arg.arg2.arg);
			if(errcode_SK!=CL_SUCCESS)
                gprint(FATAL, "setup  non-dptr %p with size %d err:%d\n",kargs[i].arg.arg2.arg,kargs[i].size,errcode_SK);
		}
	}
    
	// Now we can launch the kernel.
	if (gmm_launch2(rgns, nrgns,kernel,work_dim,global_work_offset,global_work_size,local_work_size,event) < 0) {
		for (i = 0; i < nrgns; i++)
			region_unpin(rgns[i]);
		ret = CL_INVALID_KERNEL;
	}
    
    //clSetUserEventStatus(*event,CL_COMPLETE);    
finish:
	if (rgns)
		free(rgns);
	nrefs = 0;
	nargs = 0;
	ktop = (void *)kstack;
	return ret;
}

static int gmm_load(struct region **rgns, int nrgns)
{
	int i, ret;
    
	for (i = 0; i < nrgns; i++) {
		if (rgns[i]->rwhint.flags & HINT_READ) {
			ret = region_load(rgns[i]);
			if (ret != 0)
				return -1;
		}
		if (rgns[i]->rwhint.flags & HINT_WRITE) {
			region_inval(rgns[i], 1);
			region_valid(rgns[i], 0);
		}
	}
    
	return 0;
}


static int gmm_attach(struct region **rgns, int n)
{
	char *pinned;
	int i, ret;
    
	if (n == 0)
		return 0;
	if (n < 0 || (n > 0 && !rgns))
		return -1;
    
	gprint(DEBUG, "gmm_attach begins: %d regions to attach\n", n);
    
	pinned = (char *)calloc(n, sizeof(char));
	if (!pinned) {
		gprint(FATAL, "malloc failed for pinned array: %s\n", strerror(errno));
		return -1;
	}
    
	for (i = 0; i < n; i++) {
		if (rgns[i]->state == STATE_FREEING ||
            rgns[i]->state == STATE_ZOMBIE) {
			gprint(ERROR, "cannot attach a freed region " \
                   "r(%p %p %ld %d %d)\n", \
                   rgns[i], rgns[i]->swp_addr, rgns[i]->size, \
                   rgns[i]->flags, rgns[i]->state);
			ret = -1;
			goto fail;
		}
		// NOTE: In current design, this locking is redundant
		acquire(&rgns[i]->lock);
		ret = region_attach(rgns[i], 1, rgns, n);//ERCI CHANGE TO ZERO FOR YDB
		release(&rgns[i]->lock);
		if (ret != 0)
			goto fail;
		pinned[i] = 1;
	}
    
	gprint(DEBUG, "gmm_attach succeeded\n");
	free(pinned);
	return 0;
    
fail:
	stats_inc(&pcontext->stats, count_attach_fail, 1);
	for (i = 0; i < n; i++)
		if (pinned[i])
			region_unpin(rgns[i]);
	free(pinned);
	gprint(DEBUG, "gmm_attach failed\n");
	return ret;
}


void CL_CALLBACK gmm_kernel_callback(cl_event event,cl_int status, void *data){


    
    struct kcb *pcb=(struct kcbi*)data;
    int i;
    if(status!=CL_COMPLETE){
        gprint(DEBUG,"kernel failed\n");
    }
    else{
        gprint(DEBUG,"kernel succeed\n");
    }

    for(i=0;i<pcb->nrgns;i++){
    
        if(pcb->flags[i]&HINT_WRITE){
            atomic_dec(&pcb->rgns[i]->writing);
        }
        if(pcb->flags[i]&HINT_READ){
            atomic_dec(&pcb->rgns[i]->reading);
        }
        region_unpin(pcb->rgns[i]);
    }
    free(pcb);
}

static int gmm_launch(struct region **rgns, int nrgns,cl_kernel kernel)
{
	cl_int ret=CL_SUCCESS;
	struct kcb *pcb;
	int i;
    
	if (nrgns > NREFS) {
		gprint(ERROR, "too many regions\n");
		return -1;
	}
    
	pcb = (struct kcb *)malloc(sizeof(*pcb));
	if (!pcb) {
		gprint(FATAL, "malloc failed for kcb: %s\n", strerror(errno));
		return -1;
	}
	if (nrgns > 0)
		memcpy(pcb->rgns, rgns, sizeof(void *) * nrgns);
	for (i = 0; i < nrgns; i++) {
		pcb->flags[i] = rgns[i]->rwhint.flags;
		if (pcb->flags[i] & HINT_WRITE)
			atomic_inc(&rgns[i]->writing);
		if (pcb->flags[i] & HINT_READ)
			atomic_inc(&rgns[i]->reading);
	}
	pcb->nrgns = nrgns;
    
	//stats_time_begin();
	if (ocl_clEnqueueTask(pcontext->commandQueue_kernel,kernel,0,NULL,&pcontext->event_kernel) != CL_SUCCESS) {
		for (i = 0; i < nrgns; i++) {
			if (pcb->flags[i] & HINT_WRITE)
				atomic_dec(&pcb->rgns[i]->writing);
			if (pcb->flags[i] & HINT_READ)
				atomic_dec(&pcb->rgns[i]->reading);
		}
		free(pcb);
		gprint(ERROR, "nv_cudaLaunch failed\n");
		return -1;
	} 
	//nv_cudaStreamAddCallback(stream_issue, gmm_kernel_callback, (void *)pcb, 0);
    clSetEventCallback(pcontext->event_kernel,CL_COMPLETE,gmm_kernel_callback,(void*)pcb); 


    // Uncomment the following expression if kernel time is to be measured
	//if (nv_cudaStreamSynchronize(pcontext->stream_kernel) != cudaSuccess) {
	//	gprint(ERROR, "stream synchronization failed in gmm_launch\n");
	//}
	//stats_time_end(&pcontext->stats, time_kernel);
    
	// TODO: update this client's position in global LRU client list
    
	gprint(DEBUG, "kernel launched\n");
	return 0;
}

static int gmm_launch2(struct region **rgns, int nrgns,cl_kernel kernel,cl_uint work_dim,const size_t*global_work_offset, const size_t * global_work_size, const size_t* local_work_size, cl_event* event)
{
	cl_int ret=CL_SUCCESS;
	struct kcb *pcb;
	int i;
    cl_int * errcode_KL;
	if (nrgns > NREFS) {
		gprint(ERROR, "too many regions\n");
		return -1;
	}
    
	pcb = (struct kcb *)malloc(sizeof(*pcb));
	if (!pcb) {
		gprint(FATAL, "malloc failed for kcb: %s\n", strerror(errno));
		return -1;
	}
	if (nrgns > 0)
		memcpy(pcb->rgns, rgns, sizeof(void *) * nrgns);
	for (i = 0; i < nrgns; i++) {
		pcb->flags[i] = rgns[i]->rwhint.flags;
		if (pcb->flags[i] & HINT_WRITE)
			atomic_inc(&rgns[i]->writing);
		if (pcb->flags[i] & HINT_READ)
			atomic_inc(&rgns[i]->reading);
	}
	pcb->nrgns = nrgns;
    
	//stats_time_begin();
	errcode_KL=ocl_clEnqueueNDRangeKernel(pcontext->commandQueue_kernel,kernel,work_dim,global_work_offset,global_work_size,local_work_size,0,NULL,&pcontext->event_kernel);
    if(errcode_KL!=CL_SUCCESS){
		for (i = 0; i < nrgns; i++) {
			if (pcb->flags[i] & HINT_WRITE)
				atomic_dec(&pcb->rgns[i]->writing);
			if (pcb->flags[i] & HINT_READ)
				atomic_dec(&pcb->rgns[i]->reading);
		}
		free(pcb);
		gprint(ERROR, "clEnqueueNDRangeKernel failed with errcode %d\n",errcode_KL);
		return -1;
	} 
    if(event)
        clSetEventCallback(pcontext->event_kernel, CL_COMPLETE,gmm_kernel_callback,(void*)pcb); 
    else
        clSetEventCallback(pcontext->event_kernel,CL_COMPLETE,gmm_kernel_callback,(void*)pcb); 



    // Uncomment the following expression if kernel time is to be measured
	//if (nv_cudaStreamSynchronize(pcontext->stream_kernel) != cudaSuccess) {
	//	gprint(ERROR, "stream synchronization failed in gmm_launch\n");
	//}
	//stats_time_end(&pcontext->stats, time_kernel);
    
	// TODO: update this client's position in global LRU client list
    
	gprint(DEBUG, "kernel launched\n");
	return 0;
}


static int region_load(struct region *r)
{
	int i, ret = 0;
    
	if (r->gmm_flags & FLAG_COW)
		ret = region_load_cow(r);
	else if (r->gmm_flags & FLAG_MEMSET)
		ret = region_load_memset(r);
	else if (r->gmm_flags & HINT_PTARRAY)
		ret = region_load_pta(r);
	else {
		gprint(DEBUG, "loading region %p\n", r);
		for (i = 0; i < NRBLOCKS(r->size); i++) {
			acquire(&r->blocks[i].lock);	// Though this is useless
			if (!r->blocks[i].dev_valid)
				ret = block_sync(r, i);
			release(&r->blocks[i].lock);
			if (ret != 0) {
				gprint(ERROR, "load failed\n");
				goto finish;
			}
		}
		gprint(DEBUG, "region loaded\n");
	}
    
finish:
	return ret;
}

static int region_load_pta(struct region *r)
{
	cl_mem *pdptr = (cl_mem *)(r->pta_addr);
	cl_mem *pend = (cl_mem*)((unsigned long)r->pta_addr + r->size);
	unsigned long off = 0;
	int i, ret;
    
	gprint(DEBUG, "loading pta region %p\n", r);
    
	while (pdptr < pend) {
		struct region *tmp = region_lookup(pcontext, (cl_mem)*pdptr);
		if (!tmp) {
			gprint(WARN, "cannot find region for dptr %p\n", *pdptr);
			off += sizeof(void *);
			pdptr++;
			continue;
		}
		*(cl_mem*)((unsigned long)r->swp_addr + off) = (unsigned long)tmp->dev_addr +
        (unsigned long)*pdptr - (unsigned long)tmp->swp_addr;
		off += sizeof(void *);
		pdptr++;
	}
    
	region_valid(r, 1);
	region_inval(r, 0);
	for (i = 0; i < NRBLOCKS(r->size); i++) {
		ret = block_sync(r, i);
		if (ret != 0) {
			gprint(ERROR, "load failed\n");
			return -1;
		}
	}
    
	gprint(DEBUG, "region loaded\n");
	return 0;
}


static int region_load_memset(struct region *r)
{
	gprint(DEBUG, "loading region %p with memset flag\n", r);
    
	// Memset in kernel stream ensures correct ordering with the kernel
	// referencing the region. So no explicit sync required.
    if(ocl_clEnqueueFillBuffer(pcontext->commandQueue_kernel, r->dev_addr, &r->value_memset,sizeof(int),0,r->size,0,NULL,NULL)){
        gprint(ERROR,"load failed\n");
        return -1 ;
    }
	region_valid(r, 0);
    
    r->gmm_flags &= ~FLAG_MEMSET;
    r->value_memset = 0;
    
    gprint(DEBUG, "region loaded\n");
    return 0;
}


static int region_load_cow(struct region *r)
{
	int ret;
	gprint(DEBUG, "loading region %p from cow buffer\n", r);
    
	if (!r->usr_addr) {
		gprint(ERROR, "region(%p) cow buffer is null", r);
		return -1;
	}
    
	if (!r->blocks[0].dev_valid) {
		stats_time_begin();
		ret = gmm_memcpy_htod(r->dev_addr, r->usr_addr, r->size,0);
		if (ret < 0) {
			gprint(ERROR, "load failed\n");
			return -1;
		}
		stats_time_end(&pcontext->stats, time_u2d);
		stats_inc(&pcontext->stats, bytes_u2d, r->size);
        
		region_valid(r, 0);
	}
    
#ifndef GMM_CONFIG_RW
    r->gmm_flags &= ~FLAG_COW;
    r->usr_addr = NULL;
#endif
    
    gprint(DEBUG, "region loaded\n");
    return 0;
}


// Allocate device memory to a region (i.e., attach).
static int region_attach(
                         struct region *r,
                         int pin,
                         struct region **excls,
                         int nexcl)
{
	cl_int cret;
	int ret;
    cl_int * errcode_CB=NULL;

	gprint(DEBUG, "attaching%s region %p\n", \
           (r->gmm_flags & FLAG_COW) ? " cow" : "", r);
    
	if (r->state == STATE_EVICTING) {
		gprint(ERROR, "should not see an evicting region\n");
		return -1;
	}
	if (r->state == STATE_ATTACHED) {
		gprint(DEBUG, "region already attached\n");
		if (pin)
			region_pin(r);
		// Update the region's position in the LRU list.
		list_attached_mov(pcontext, r);
		return 0;
	}
	if (r->state != STATE_DETACHED) {
		gprint(ERROR, "attaching a non-detached region\n");
		return -1;
	}
    
	// Attach if current free memory space is larger than region size.
	if (r->size <= memsize_free()) {
		r->dev_addr = ocl_clCreateBuffer(pcontext->context_kernel, CL_MEM_READ_WRITE,r->size,NULL,errcode_CB);
        //gprint(DEBUG,"what is in the dev_addr(%p)\n",r->dev_addr);
        if(errcode_CB==CL_SUCCESS){
			goto attach_success;
        }
		else {
			gprint(DEBUG, "ocl_clCreateBuffer failed inside the gmm_attach");
            return -1;
		}
	}
    
	// Evict some device memory.
	stats_time_begin();
	ret = gmm_evict(r->size, excls, nexcl);
	stats_time_end(&pcontext->stats, time_evict);
	if (ret < 0 || (ret > 0 && memsize_free() < r->size))
		return ret;
    
	// Try to attach again.
    r->dev_addr=ocl_clCreateBuffer(pcontext->context_kernel, CL_MEM_READ_WRITE,r->size,NULL,errcode_CB);
    if(errcode_CB!=CL_SUCCESS){
		r->dev_addr = NULL;
		gprint(DEBUG, "openCL creating buffer failed\n");
		return 1;
	}
    
attach_success:
	latomic_add(&pcontext->size_attached, r->size);
	update_attached(r->size);
	update_detachable(r->size);
	if (pin)
		region_pin(r);
	// Reassure that the dev copies of all blocks are set to invalid.
	region_inval(r, 0);
	r->state = STATE_ATTACHED;
	list_attached_add(pcontext, r);
    
	gprint(DEBUG, "region attached\n");
	return 0;
}


int victim_select(
                  long size_needed,
                  struct region **excls,
                  int nexcl,
                  int local_only,
                  struct list_head *victims)
{
	int ret = 0;
    
#if defined(GMM_REPLACEMENT_LRU)
	ret = victim_select_lru(size_needed, excls, nexcl, local_only, victims);
#elif defined(GMM_REPLACEMENT_LFU)
	ret = victim_select_lfu(size_needed, excls, nexcl, local_only, victims);
#else
	panic("replacement policy not specified");
	ret = -1;
#endif
    
	return ret;
}

// NOTE: When a local region is evicted, no other parties are
// supposed to be accessing the region at the same time.
// This is not true if multiple loadings happen simultaneously,
// but this region has been locked in region_load() anyway.
// A dptr array region's data never needs to be transferred back
// from device to host because swp_valid=0,dev_valid=1 will never
// happen.
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
		ocl_clReleaseMemObject(r->dev_addr);
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

// NOTE: Client %client should have been pinned when this function
// is called.
long remote_victim_evict(int client, long size_needed)
{
	long size_spared;
    
	gprint(DEBUG, "remote eviction in client %d\n", client);
	size_spared = msq_send_req_evict(client, size_needed, 1);
	gprint(DEBUG, "remote eviction returned: %ld\n", size_spared);
	client_unpin(client);
    
	return size_spared;
}

// Similar to gmm_evict, but only select at most one victim from local
// region list, even if it is smaller than required, evict it, and return.
long local_victim_evict(long size_needed)
{
	struct list_head victims;
	struct victim *v;
	struct region *r;
    
	gprint(DEBUG, "local eviction: %ld bytes\n", size_needed);
	INIT_LIST_HEAD(&victims);
    
	if (victim_select(size_needed, NULL, 0, 0, &victims) != 0)
		return -1;
    
	if (list_empty(&victims))
		return 0;
    
	v = list_entry(victims.next, struct victim, entry);
	r = v->r;
	free(v);
	return region_evict(r);
}

// Evict the victim %victim.
// %victim may point to a local region or a remote client that
// may own some evictable region.
// The return value is the size of free space spared during
// this eviction. -1 means error.
long victim_evict(struct victim *victim, long size_needed)
{
	if (victim->r)
		return region_evict(victim->r);
	else if (victim->client != -1)
		return remote_victim_evict(victim->client, size_needed);
	else {
		panic("victim is neither local nor remote");
		return -1;
	}
}

// Evict some device memory so that the size of free space can
// satisfy %size_needed. Regions in %excls[0:%nexcl) should not
// be selected for eviction.
static int gmm_evict(long size_needed, struct region **excls, int nexcl)
{
	struct list_head victims, *e;
	struct victim *v;
	long size_spared;
	int ret = 0;
    
	gprint(DEBUG, "evicting for %ld bytes\n", size_needed);
	INIT_LIST_HEAD(&victims);
	stats_inc(&pcontext->stats, bytes_evict_needed,
              LMAX(size_needed - memsize_free(), 0));
    
	do {
		ret = victim_select(size_needed, excls, nexcl, 1, &victims);
		if (ret != 0)
			return ret;
        
		for (e = victims.next; e != (&victims); ) {
			v = list_entry(e, struct victim, entry);
			if (memsize_free() < size_needed) {
				if ((size_spared = victim_evict(v, size_needed)) < 0) {
					ret = -1;
					goto fail_evict;
				}
				stats_inc(&pcontext->stats, bytes_evict_space, size_spared);
				stats_inc(&pcontext->stats, count_evict_victims, 1);
			}
			else if (v->r) {
				acquire(&v->r->lock);
				if (v->r->state != STATE_FREEING)
					v->r->state = STATE_ATTACHED;
				release(&v->r->lock);
			}
			else if (v->client != -1)
				client_unpin(v->client);
			list_del(e);
			e = e->next;
			free(v);
		}
	} while (memsize_free() < size_needed);
    
	gprint(DEBUG, "eviction finished\n");
	return 0;
    
fail_evict:
	stats_inc(&pcontext->stats, count_evict_fail, 1);
	for (e = victims.next; e != (&victims); ) {
		v = list_entry(e, struct victim, entry);
		if (v->r) {
			acquire(&v->r->lock);
			if (v->r->state != STATE_FREEING)
				v->r->state = STATE_ATTACHED;
			release(&v->r->lock);
		}
		else if (v->client != -1)
			client_unpin(v->client);
		list_del(e);
		e = e->next;
		free(v);
	}
    
	gprint(DEBUG, "eviction failed\n");
	return ret;
}


static int gmm_dtoh_block(
                          struct region *r,
                          void *dst,
                          unsigned long off,
                          unsigned long size,
                          int iblock,
                          int skip,
                          char *skipped)
{
	struct block *b = r->blocks + iblock;
	int ret = 0;
    
    /*	GMM_DPRINT("gmm_dtoh_block: r(%p) dst(%p) off(%lu)" \
     " size(%lu) block(%d) swp_valid(%d) dev_valid(%d)\n", \
     r, dst, off, size, iblock, b->swp_valid, b->dev_valid);
     */
	if (BLOCKIDX(off) != iblock)
		panic("dtoh_block");
    
	if (b->swp_valid) {
		stats_time_begin();
		memcpy(dst, r->swp_addr + off, size);
		stats_time_end(&pcontext->stats, time_s2u);
		stats_inc(&pcontext->stats, bytes_s2u, size);
        
		if (skipped)
			*skipped = 0;
		return 0;
	}
    
	stats_time_begin();
	while (!try_acquire(&b->lock)) {
		if (skip) {
			if (skipped)
				*skipped = 1;
			return 0;
		}
	}
	stats_time_end(&pcontext->stats, time_sync);
    
	if (b->swp_valid) {
		release(&b->lock);
		stats_time_begin();
		memcpy(dst, r->swp_addr + off, size);
		stats_time_end(&pcontext->stats, time_s2u);
		stats_inc(&pcontext->stats, bytes_s2u, size);
	}
	else if (!b->dev_valid) {
		release(&b->lock);
	}
	else if (skip) {
		release(&b->lock);
		if (skipped)
			*skipped = 1;
		return 0;
	}
	else { // dev_valid == 1 && swp_valid == 0
		// We don't need to pin the device memory because we are holding the
		// lock of a swp_valid=0,dev_valid=1 block, which will prevent the
		// evictor, if any, from freeing the device memory under us.
		ret = block_sync(r, iblock);
		release(&b->lock);
		if (ret != 0)
			goto finish;
        
		stats_time_begin();
		memcpy(dst, r->swp_addr + off, size);
		stats_time_end(&pcontext->stats, time_s2u);
		stats_inc(&pcontext->stats, bytes_s2u, size);
	}
    
finish:
	if (skipped)
		*skipped = 0;
	return ret;
}

static int gmm_dtoh_pta(
                        struct region *r,
                        void *dst,
                        const void *src,
                        size_t count)
{
	unsigned long off = (unsigned long)(src - r->swp_addr);
    
	if (off % sizeof(void *)) {
		gprint(ERROR, "offset(%lu) not aligned for pta-to-host memcpy\n", off);
		return -1;
	}
	if (count % sizeof(void *)) {
		gprint(ERROR, "count(%lu) not aligned for pta-to-host memcpy\n", count);
		return -1;
	}
    
	stats_time_begin();
	memcpy(dst, r->pta_addr + off, count);
	stats_time_end(&pcontext->stats, time_s2u);
	stats_inc(&pcontext->stats, bytes_s2u, count);
    
	return 0;
}

// TODO: It is possible to achieve pipelined copying, i.e., copy a block from
// its host swap buffer to user buffer while the next block is being fetched
// from device memory.
int gmm_dtoh(struct region *r, void *dst, const cl_mem src, size_t count,size_t src_off)
{
	unsigned long off = (unsigned long)src - (unsigned long)r->swp_addr;
	unsigned long end = off+src_off+count, size;
	int ifirst = BLOCKIDX(off), iblock;
	void *d = dst;
	char *skipped;
	int ret = 0;
    
	gprint(DEBUG, "dtoh: r(%p %p %ld %d %d) dst(%p) src(%p) count(%lu)\n", \
           r, r->swp_addr, r->size, r->gmm_flags, r->state, dst, src, count);
    
	if (r->gmm_flags & HINT_PTARRAY)
		return gmm_dtoh_pta(r, dst, src, count);
    
	skipped = (char *)malloc(BLOCKIDX(end - 1) - ifirst + 1);
	if (!skipped) {
		gprint(FATAL, "malloc failed for skipped[]: %s\n", strerror(errno));
		return -1;
	}
    
	// First, copy blocks whose swp buffers contain immediate, valid data.
    off=off+src_off;
	iblock = ifirst;
	while (off < end) {
		size = MIN(BLOCKUP(off), end) - off;
		ret = gmm_dtoh_block(r, d, off+src_off, size, iblock, 1,
                             skipped + iblock - ifirst);
		if (ret != 0)
			goto finish;
		d += size;
		off += size;
		iblock++;
	}
    
	// Then, copy the rest blocks.
	off = (unsigned long)src -(unsigned long)r->swp_addr;
    off = src_off+off;
	iblock = ifirst;
	d = dst;
	while (off < end) {
		size = MIN(BLOCKUP(off), end) - off;
		if (skipped[iblock - ifirst]) {
			ret = gmm_dtoh_block(r, d, off, size, iblock, 0, NULL);
			if (ret != 0)
				goto finish;
		}
		d += size;
		off += size;
		iblock++;
	}
    
finish:
	free(skipped);
	return ret;
}




cl_int gmm_clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem src, cl_bool blocking, size_t offset,size_t count, void * dst, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event*event){

	struct region *r;
    
	if (count <= 0)
		return CL_INVALID_VALUE;
    
	r = region_lookup(pcontext, src);
	if (!r) {
		gprint(ERROR, "cannot find region containing %p in dtoh\n", src);
		return CL_INVALID_MEM_OBJECT;
	}
	if (r->state == STATE_FREEING || r->state == STATE_ZOMBIE) {
		gprint(ERROR, "region already freed\n");
		return CL_INVALID_VALUE;
	}
	if (r->gmm_flags & FLAG_COW) {
		gprint(ERROR, "dtoh: region already tagged COW\n");
		return CL_INVALID_VALUE;
	}
	if ((unsigned long)src + count > r->swp_addr + r->size) {
		gprint(ERROR, "dtoh out of region boundary\n");
		return CL_INVALID_VALUE;
	}
    
	stats_time_begin();
	if (gmm_dtoh(r, dst, src, count,0) < 0)
	    return CL_INVALID_VALUE;
	stats_time_end(&pcontext->stats, time_dtoh);
	stats_inc(&pcontext->stats, bytes_dtoh, count);
    
    //clSetUserEventStatus(*event,CL_COMPLETE);    
	return CL_SUCCESS;
}
