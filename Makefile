include gmm.mk

## Specify CUDA install path here
<<<<<<< HEAD
CUDAPATH = /usr/local/cuda
=======
CUDAPATH = /usr/local/cuda-6.5
>>>>>>> e9a66bdce415aaa1c33e8344ba8b8d3d4c7ed3df

# Name of the GMM shared library
LIBGMM = libgmm.so

# Files needed to create $(LIBGMM)
SRCS = client.c common.c core.c interfaces.c msq.c replacement.c stats.c debug.c

OBJS = $(SRCS:.c=.o)

# Use print buffer?
ifeq ($(USE_PRINT_BUFFER), 0)
FLAG_PRINT_BUFFER :=
else
FLAG_PRINT_BUFFER := -DGMM_PRINT_BUFFER
endif

# The compiler/linker settings
CC := gcc
#NVCC := $(CUDAPATH)/bin/nvcc
CFLAGS := -g -Wall -pthread -fPIC -fvisibility=hidden \
	-I$(CUDAPATH)/include  \
	-DGMM_PRINT_LEVEL=$(PRINT_LEVEL) $(FLAG_PRINT_BUFFER) $(GMM_CONFIGS)
LDFLAGS := -shared -pthread -ldl -fPIC -OpenCL

.DEFAULT_GOAL := all
.PHONY : depend all clean install uninstall

all : depend gmmctl $(LIBGMM)

# Generate dependencies for $(OBJS)
depend : .depend

.depend : $(SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > .depend

-include .depend

# No rules for source files
%.c: ;

gmmctl : server.o
	$(CC) -L$(CUDAPATH)/lib $^ -o  $@ -lpthread -lrt

server.o : server.c protocol.h spinlock.h list.h atomic.h
	$(CC) -c -l OpenCL -Wall -g\
		-I$(CUDAPATH)/include  $< -o $@ -lpthread -lrt

$(LIBGMM): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	-rm -f gmmctl $(LIBGMM) *.o .depend

# TODO
install: ;

# TODO
uninstall: ;
