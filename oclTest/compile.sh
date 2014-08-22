rm -f a.out
gcc -I/usr/local/cuda-6.0/include test.c -L/home/erci/oclGmm/oclGmm -lgmm -lrt -lOpenCL

