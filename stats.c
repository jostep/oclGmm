#include <string.h>
#include "common.h"
#include "stats.h"

#ifdef GMM_CONFIG_STATS
// Current # of bytes of active device memory
long bytes_mem_active = 0L;
#endif

void stats_init(struct statistics *pstats)
{
#ifdef GMM_CONFIG_STATS
	memset(pstats, 0, sizeof(*pstats));
#endif
}

void stats_print(struct statistics *pstats)
{
#ifdef GMM_CONFIG_STATS
	acquire(&pstats->lock);
	gprint(STAT, "------------------------------\n");
	gprint(STAT, "bytes_mem_alloc     %ld\n", pstats->bytes_mem_alloc);
	gprint(STAT, "bytes_mem_peak      %ld\n", pstats->bytes_mem_peak);
	gprint(STAT, "bytes_mem_freed     %ld\n", pstats->bytes_mem_freed);
	gprint(STAT, "bytes_htod          %ld\n", pstats->bytes_htod);
	gprint(STAT, "time_htod           %.3lf\n", pstats->time_htod);
	gprint(STAT, "bytes_htod_cow      %ld\n", pstats->bytes_htod_cow);
	gprint(STAT, "time_htod_cow       %.3lf\n", pstats->time_htod_cow);
	gprint(STAT, "bytes_dtoh          %ld\n", pstats->bytes_dtoh);
	gprint(STAT, "time_dtoh           %.3lf\n", pstats->time_dtoh);
	gprint(STAT, "bytes_dtod          %ld\n", pstats->bytes_dtod);
	gprint(STAT, "time_dtod           %.3lf\n", pstats->time_dtod);
	gprint(STAT, "bytes_memset        %ld\n", pstats->bytes_memset);
	gprint(STAT, "time_memset         %.3lf\n", pstats->time_memset);
	gprint(STAT, "------------------------------\n");
	gprint(STAT, "bytes_u2s           %ld\n", pstats->bytes_u2s);
	gprint(STAT, "time_u2s            %.3lf\n", pstats->time_u2s);
	gprint(STAT, "bytes_s2d           %ld\n", pstats->bytes_s2d);
	gprint(STAT, "time_s2d            %.3lf\n", pstats->time_s2d);
	gprint(STAT, "bytes_u2d           %ld\n", pstats->bytes_u2d);
	gprint(STAT, "time_u2d            %.3lf\n", pstats->time_u2d);
	gprint(STAT, "bytes_d2s           %ld\n", pstats->bytes_d2s);
	gprint(STAT, "time_d2s            %.3lf\n", pstats->time_d2s);
	gprint(STAT, "bytes_s2u           %ld\n", pstats->bytes_s2u);
	gprint(STAT, "time_s2u            %.3lf\n", pstats->time_s2u);
	gprint(STAT, "------------------------------\n");
	gprint(STAT, "bytes_evict_needed  %ld\n", pstats->bytes_evict_needed);
	gprint(STAT, "bytes_evict_space   %ld\n", pstats->bytes_evict_space);
	//gprint(STAT, "bytes_evict_data    %ld\n", pstats->bytes_evict_data);
	gprint(STAT, "count_evict_victims %ld\n", pstats->count_evict_victims);
	gprint(STAT, "count_evict_fail    %ld\n", pstats->count_evict_fail);
	gprint(STAT, "count_attach_fail   %ld\n", pstats->count_attach_fail);
	gprint(STAT, "------------------------------\n");
	gprint(STAT, "time_sync           %.3lf\n", pstats->time_sync);
	gprint(STAT, "time_kernel         %.3lf\n", pstats->time_kernel);
	gprint(STAT, "time_attach         %.3lf\n", pstats->time_attach);
	gprint(STAT, "time_load           %.3lf\n", pstats->time_load);
	gprint(STAT, "------------------------------\n");
	release(&pstats->lock);
#endif
}
