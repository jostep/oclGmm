// The interfaces exported to allow user programs to print GMM runtime
// info for debugging purposes.
// The user program should first open libgmm.so with RTLD_NOLOAD flag
// to get the handle to the already loaded shared library. Then use
// dlsym to get the addresses of related interfaces.
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "common.h"
#include "atomic.h"
#include "core.h"

struct region *region_lookup(struct gmm_context *ctx, const void *ptr);
//int gmm_dtoh(struct region *r, void *dst, const void *src, size_t count);

extern struct gmm_context *pcontext;

// For internal use only
void gmm_print_region(void *rgn)
{
	struct region *r = (struct region *)rgn;

	gprint(DEBUG, "printing dptr %p (%p)\n", r->swp_addr, r);
	gprint(DEBUG, "\tsize: %ld\t\tstate: %d\t\tflags: %d\n", \
			r->size, r->state, r->flags);
	gprint(DEBUG, "\tdev_addr: %p\t\tswp_addr: %p\t\tpta_addr: %p\n", \
			r->dev_addr, r->swp_addr, r->pta_addr);
	gprint(DEBUG, "\tpinned: %d\t\twriting: %d\t\treading: %d\n", \
			atomic_read(&r->pinned), atomic_read(&r->writing), \
			atomic_read(&r->reading));
}

void gmm_dump_region(
		const char *filepath,
		struct region *r,
		const void *addr,
		size_t size)
{
	void *temp;
	FILE *f;

	temp = malloc(size);
	if (!temp) {
		gprint(FATAL, "malloc failed for temp dump buffer: %s\n", \
				strerror(errno));
		return;
	}

	/*if (gmm_dtoh(r, temp, addr, size) < 0) {
		gprint(ERROR, "dtoh failed for region dump\n");
		goto finish;
	}*/

	f = fopen(filepath, "wb");
	if (!f) {
		gprint(ERROR, "failed to open file (%s) for region dump\n", filepath);
		goto finish;
	}

	if (fwrite(temp, size, 1, f) == 0) {
		gprint(ERROR, "write error for region dump\n");
		goto finish;
	}

	fclose(f);

finish:
	free(temp);
}

// Print info of the region containing %dptr
GMM_EXPORT
void gmm_print_dptr(const void *dptr)
{
	struct region *r;

	r = region_lookup(pcontext, dptr);
	if (!r) {
		gprint(DEBUG, "failed to look up region containing %p\n", dptr);
		return;
	}
	gmm_print_region(r);
}

GMM_EXPORT
void gmm_dump_dptr(const char *filepath, const void *dptr, const size_t size)
{
	struct region *r;

	if (!filepath) {
		gprint(DEBUG, "bad filepath for gmm_dump\n");
		return;
	}
	if (size <= 0) {
		gprint(DEBUG, "bad size for gmm_dump\n");
		return;
	}

	r = region_lookup(pcontext, dptr);
	if (!r) {
		gprint(DEBUG, "region lookup failed for %p in gmm_dump\n", dptr);
		return;
	}

	if (dptr + size > &(r->swp_addr) + r->size) {//potential bug
		gprint(DEBUG, "bad dump range for gmm_dump\n");
		return;
	}

	gmm_dump_region(filepath, r, dptr, size);
}
