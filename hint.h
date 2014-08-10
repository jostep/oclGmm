#ifndef _GMM_HINT_H_
#define _GMM_HINT_H_

// Read/write hints
#define HINT_READ		1
#define HINT_INPUT		1		// This is the accurate name for hint_read
#define HINT_WRITE		2
#define HINT_OUTPUT		2		// This is the accurate name for hint_write
#define HINT_DEFAULT	(HINT_INPUT | HINT_OUTPUT)
#define HINT_MASK		(HINT_INPUT | HINT_OUTPUT)

// Device pointer array flags
//#define HINT_PTRARRAY	4	// to be deleted
#define HINT_PTARRAY	4
#define HINT_PTAREAD	8
#define HINT_PTAWRITE	16
#define HINT_PTADEFAULT	(HINT_PTAREAD | HINT_PTAWRITE)
#define HINT_PTAMASK	(HINT_PTAREAD | HINT_PTAWRITE)

#define FLAG_COW		32
#define cudaMemcpyHostToDeviceCow	8
#define GMM_BUFFER_COW	8

#define FLAG_MEMSET		64

// Kernel priority hints. Highest priority is 0.
#define PRIO_LOWEST		15
#define PRIO_DEFAULT	PRIO_LOWEST

#endif
