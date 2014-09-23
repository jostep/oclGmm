cd ../
make clean
make 
./gmmctl --stop
./gmmctl --start
cd ./oclTest
rm -f a.out
export LD_LIBRARY_PATH=/home/erci/oclGmm:$LD_LIBRARY_PATH
#gcc -g -I/usr/local/cuda-6.0/include -L/home/erci/oclGmm test.c -Xlinker -rpath=/home/erci/oclGmm -lgmm -lrt -lOpenCL
gcc -g -I/usr/local/cuda-6.0/include test.c  -lrt -lOpenCL
./a.out
