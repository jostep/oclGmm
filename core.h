#ifndef _GMM_CORE_H_
#define _GMM_CORE_H_

#include <stdint.h>
#include <sys/time.h>


#include "list.h"
#include "spinlock.h"
#include "atomic.h"
#include "stats.h"

#define NBUFS 2
#define BUFSIZE (512*1024)
#define region_pinned(r) atomic_read(&((r)->pinned));
struct dma_channel{
    struct spinlock lock;
    cl_command_queue commandQueue_chan;//stream
    int ibuf;
    void *stage_bufs[NBUFS];
    cl_event events[NBUFS];
}

struct rwhint{
    int flags;
}

struct gmm_context{
    struct spinlock lock;
    struct list_head list_allocated;
    struct list_head list_attached;
    struct spinlock lock_attached;
    struct spinlock lock_alloced;
    struct dma_channel dma_htod;
    struct dma_channel dma_dtoh;
    cl_platform_id platform;
    cl_device_id *device;
    cl_context context_kernel;
    cl_command_queue commandQueue_kernel;
    struct statistics stats;
}

struct region{
    
    long size;
    cl_mem *dev_addr;
    cl_mem *swp_addr;
    cl_mem *pta_addr;
    cl_mem *usr_addr;
    int  value_memset;
    struct block *blocks;
    struct spinlock lock;
    region_state_t state;
    atomic_t pinned;
    atomic_t reading;
    atomic_t writing;
    cl_mem_flags flags;
    int gmm_flags;
    struct rwhint rwhint;

    struct list_head entry_alloced;
    struct list_head entry_attached;

}

#endif
