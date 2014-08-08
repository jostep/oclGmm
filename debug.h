#ifndef _GMM_DEBUG_H_
#define _GMM_DEBUG_H_

#ifdef __cplusplus
extern "C" {
#endif

void gmm_print_region(const void *rgn);
void gmm_print_dptr(const void *dptr);
void gmm_dump_region(
		const char *filepath,
		struct region *r,
		const void *addr,
		size_t size);
void gmm_dump_dptr(const char *filepath, const void *dptr, const size_t size);

#ifdef __cplusplus
}
#endif

#endif
