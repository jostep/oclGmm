rm -f a.out
export LD_LIBRARY_PATH=/home/erci/oclGmm/oclGmm:$LD_LIBRARY_PATH
gcc -g -I/usr/local/cuda-6.0/include -L/home/erci/oclGmm/oclGmm/ test.c -Xlinker -rpath=/home/erci/oclGmm/oclGmm -lgmm -lrt -lOpenCL
