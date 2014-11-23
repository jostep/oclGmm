#ifndef _GMM_CORE_H_
#define _GMM_CORE_H_

#include <stdint.h>
#include <sys/time.h>

#include "CL/opencl.h"
#include "list.h"
#include "spinlock.h"
#include "atomic.h"
#include "stats.h"


// State of a device memory region
typedef enum region_state {
	STATE_DETACHED = 0,		// object not allocated with device memory
	STATE_ATTACHED,			// object allocated with device memory
	STATE_FREEING,			// object being freed
	STATE_EVICTING,			// object being evicted
	STATE_ZOMBIE			// object waiting to be GC'ed
} region_state_t;

// RW hint passed to a device memory region.
// TODO: RW hints can be very delicate. For example, if DB knows a kernel
// only modifies part of a region, it can pass GMM a RW hint with the range.
struct rwhint {
	int flags;
};

// Device memory block.
struct block {
	int dev_valid;			// if data copy on device is valid
	int swp_valid;			// if data copy in host swap buffer is valid
	struct spinlock lock;	// r/w lock
};

// Device memory region.
// A device memory region is a virtual memory area allocated by the user
// program through cudaMalloc. It is logically partitioned into an array of
// fixed-length device memory blocks. Due to lack of system support, all blocks
// must be attached/detached together. But the valid and dirty statuses of
// each block are maintained separately.
struct region {
	unsigned long size;				// size of the object in bytes
	cl_mem dev_addr;			// device memory address
	void* swp_addr;			// host swap buffer address
	void* pta_addr;			// dptr array address
	cl_mem usr_addr;			// copy-on-write user address
	int value_memset;		// value of cudaMemset
	struct block *blocks;	// device memory blocks
	struct spinlock lock;	// the lock that protects memory object state
	region_state_t state;	// state of the object
	atomic_t pinned;		// atomic pin counter
	atomic_t writing;		// being written by how many kernels
	atomic_t reading;		// being read by how many kernels

    cl_mem_flags flags;
	int gmm_flags;
	struct rwhint rwhint;	// rw hint

	struct list_head entry_alloced;		// linked to the list of allocated
	struct list_head entry_attached;	// linked to the list of attached
};

// Maximum number of kernel arguments that may be device memory pointers
#define NREFS		32

// A kernel argument that is a device memory pointer
struct dptr_arg {
	struct region *r;		// the region this argument points to
	unsigned long off;		// device pointer offset in the region
	int flags;				// RW hints
	cl_mem  dptr;				// the actual device memory address
};

// A kernel argument that is not a device memory pointer
struct ndptr_arg {
	void *arg;
};

// A kernel argument
struct karg {
	char is_dptr;
	union {
		struct dptr_arg arg1;
		struct ndptr_arg arg2;
	} arg;
	size_t size;
	size_t argoff;
};

// Kernel callback structure
struct kcb {
	struct region *rgns[NREFS];	// Regions referenced by the kernel
	int flags[NREFS];			// is each region read/modified by the kernel?
	int nrgns;					// Number of regions referenced
};

// Staging buffer number and size.
// It seems nbufs=2, bufsize=512k, blocksize=8m performs good
#define NBUFS		2
#define BUFSIZE		(512 * 1024)

// A DMA channel for either HtoD or DtoH data transfer
struct dma_channel {
	struct spinlock lock;
    cl_command_queue commandQueue_chan;
	int ibuf;					// The next staging buffer to be used
	cl_mem stage_bufs[NBUFS];	// Host-pinned staging buffers
	cl_event events[NBUFS];	// Events for syncing staging buffers
};

// The local GMM context
struct gmm_context {
	struct spinlock lock;				// TODO: what's the use of this lock??
	latomic_t size_attached;			// Total size of attached mem regions
	struct list_head list_alloced;		// List of all allocated mem regions
	struct list_head list_attached;		// LRU list of attached mem regions
	struct spinlock lock_alloced;
	struct spinlock lock_attached;
	struct dma_channel dma_htod;		// HtoD DMA channel
	struct dma_channel dma_dtoh;		// DtoH DMA channel
    cl_platform_id *platform;
    cl_device_id * device;
    cl_context context_kernel;
    cl_command_queue commandQueue_kernel;
    cl_program program_kernel;
    cl_event event_kernel;
	struct statistics stats;
};

// A victim region for being evicted
struct victim {
	struct region *r;		// for a local victim
	int client;				// for a remote victim
	struct list_head entry;
};

#define PAGEMASK            ((unsigned long)((unsigned long)sysconf(_SC_PAGE_SIZE) - 1UL))
#define UPPER_PAGE(addr)    ((void *)(((unsigned long)addr + PAGEMASK) & ~(PAGEMASK)))
#define LOWER_PAGE(addr)    ((void *)((unsigned long)addr & ~(PAGEMASK)))

#define MIN(x, y)	((x) < (y) ? (x) : (y))

#define BLOCKSIZE			(8 * 1024 * 1024)
#define BLOCKSHIFT			22
#define BLOCKMASK			(~(BLOCKSIZE - 1))

// Could use BLOCKSHIFT, but it has little impact to system performance.
#define NRBLOCKS(size)		(((size) + BLOCKSIZE - 1) / BLOCKSIZE)
#define BLOCKIDX(offset)	(((unsigned long)(offset)) / BLOCKSIZE)
#define BLOCKUP(offset)		((((offset) + BLOCKSIZE) / BLOCKSIZE) * BLOCKSIZE)

#define ALIGNUP(addr, size)	((void *)(((unsigned long)(addr) + \
		(unsigned long)(size) - 1UL) / ((unsigned long)(size)) * \
		((unsigned long)(size))))

#define region_pinned(r)	atomic_read(&((r)->pinned))


// Invalidate all blocks in a region.
static inline void region_inval(struct region *r, int swp)
{
	int i;

	if (swp) {
		for (i = 0; i < NRBLOCKS(r->size); i++)
			r->blocks[i].swp_valid = 0;
	}
	else {
		for (i = 0; i < NRBLOCKS(r->size); i++)
			r->blocks[i].dev_valid = 0;
	}
}

// Validate all blocks in a region.
static inline void region_valid(struct region *r, int swp)
{
	int i;

	if (swp) {
		for (i = 0; i < NRBLOCKS(r->size); i++)
			r->blocks[i].swp_valid = 1;
	}
	else {
		for (i = 0; i < NRBLOCKS(r->size); i++)
			r->blocks[i].dev_valid = 1;
	}
}

// Whether pointer p is included in pointer array a[0:n)
static inline int is_included(void **a, int n, void *p)
{
	int i;

	for (i = 0; i < n; i++)
		if (a[i] == p)
			return 1;

	return 0;
}


// Functions exposed by GMM core
int gmm_context_init();
void gmm_context_fini();
#endif
