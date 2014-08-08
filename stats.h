#ifndef _GMM_STATS_H_
#define _GMM_STATS_H_

#include "spinlock.h"

struct statistics {
	struct spinlock lock;

	// API-level
	long bytes_mem_alloc;		// # of bytes allocated
	long bytes_mem_peak;		// # of bytes held at peak
	long bytes_mem_freed;		// # of bytes explicitly freed (detect leaks)
	long bytes_htod;			// # of bytes going through the htod API
	double time_htod;
	long bytes_htod_cow;		// # of htod bytes that are marked cow
	double time_htod_cow;
	long bytes_dtoh;			// # of bytes going through the dtoh API
	double time_dtoh;
	long bytes_dtod;			// # of bytes going through the dtod API
	double time_dtod;
	long bytes_memset;			// # of bytes going through the memset API
	double time_memset;

	// Bus-level
	long bytes_u2s;				// user buffer to swap buffer
	double time_u2s;
	long bytes_s2d;				// swap buffer to device
	double time_s2d;
	long bytes_u2d;				// user buffer to device, i.e. cow loading
	double time_u2d;
	long bytes_d2s;				// device to swap buffer
	double time_d2s;
	long bytes_s2u;				// swap buffer to user buffer
	double time_s2u;

	// Replacement
	long bytes_evict_needed;	// # of bytes of free space required
	long bytes_evict_space;		// # of bytes actually freed
	//long bytes_evict_data;		// # of bytes of data evicted
	long count_evict_victims;	// # of evicted victim regions
	long count_evict_fail;		// # of failed evictions in gmm_evict
	long count_attach_fail;		// # of failed attach

	// Other
	double time_sync;			// Time on synchronizations
	double time_kernel;			// Time on kernel executions
	double time_attach;			// Time on attaching (gmm_attach)
	double time_load;			// Time on loading (gmm_load)
	double time_evict;			// Time on eviction (gmm_evict)
};

#define TVAL(t)			(t.tv_sec * 1000.0 + t.tv_usec / 1000.0)
#define TDIFF(t1, t2)	(TVAL(t2) - TVAL(t1))

#ifdef GMM_CONFIG_STATS
extern long bytes_mem_active;

// bytes_mem_alloc and bytes_mem_freed are treated specially,
// because they may need to update bytes_mem_peak.
#define stats_inc_alloc(pstats, delta) \
	do { \
		acquire(&(pstats)->lock); \
		(pstats)->bytes_mem_alloc += (long)delta; \
		bytes_mem_active += (long)delta; \
		if (bytes_mem_active > (pstats)->bytes_mem_peak) \
			(pstats)->bytes_mem_peak = bytes_mem_active; \
		release(&(pstats)->lock); \
	} while (0)

#define stats_inc_freed(pstats, delta) \
	do { \
		acquire(&(pstats)->lock); \
		(pstats)->bytes_mem_freed += (long)delta; \
		bytes_mem_active -= (long)delta; \
		release(&(pstats)->lock); \
	} while (0)

#define stats_inc(pstats, elem, delta)	\
	do { \
		acquire(&(pstats)->lock); \
		(pstats)->elem += (long)(delta); \
		release(&(pstats)->lock); \
	} while (0)

// There can be no break statement or variable declaration in
// the program segment measured by the macros.
#define stats_time_begin() \
	do { \
		struct timeval t1, t2; \
		gettimeofday(&t1, NULL)

#define stats_time_end(pstats, elem) \
		gettimeofday(&t2, NULL); \
		acquire(&(pstats)->lock); \
		(pstats)->elem += TDIFF(t1, t2); \
		release(&(pstats)->lock); \
	} while (0)

#define stats_record_time(pstats, elem, func)	\
	do { \
		struct timeval t1, t2; \
		gettimeofday(&t1, NULL); \
		func; \
		gettimeofday(&t2, NULL); \
		acquire(&(pstats)->lock); \
		(pstats)->elem += TDIFF(t1, t2); \
		release(&(pstats)->lock); \
	} while (0)

#define stats_kernel_start(pstats, pcb)	\
	do { \
		gettimeofday(&(pcb->t_start), NULL); \
	} while (0)

#define stats_kernel_end(pstats, pcb)	\
	do { \
		struct timeval t_end; \
		gettimeofday(&t_end, NULL); \
		acquire(&(pstats)->lock); \
		(pstats)->time_kernel += \
			(double)((t_end.tv_sec - pcb->t_start.tv_sec)*1000 \
			+ (double)(t_end.tv_usec - pcb->t_start.tv_usec)/1000); \
		release(&(pstats)->lock); \
	} while (0)
#else
#define stats_inc_alloc(pstats, delta)
#define stats_inc_freed(pstats, delta)
#define stats_inc(pstats, elem, delta)
#define stats_time_begin()
#define stats_time_end(pstats, elem)
#define stats_record_time(pstats, elem, func) func
#define stats_kernel_start(pstats, pcb)
#define stats_kernel_end(pstats, pcb)
#endif

void stats_init(struct statistics *pstats);
void stats_print(struct statistics *pstats);

#endif
