cd ../
make clean
make 
./gmmctl --stop
./gmmctl --start
cd ./oclTest
rm -f a.out
export LD_LIBRARY_PATH=/home/erci/oclGmm
export LD_LIBRARY_PATH=/usr/local/cuda/lib
<<<<<<< HEAD
gcc -g -I/usr/local/cuda/include -L/home/erci/oclGmm  test.c -Xlinker -rpath=/home/erci/oclGmm -lgmm -lrt -lOpenCL
#gcc -g -I/usr/local/cuda-6.0/include newT.c  -lrt -lOpenCL
./a.out # &./a.out>result1 &./a.out>./result2
=======
gcc -g -I/usr/local/cuda-6.0/include -L/home/erci/oclGmm  test.c -Xlinker -rpath=/home/erci/oclGmm -lgmm -lrt -lOpenCL
#gcc -g -I/usr/local/cuda-6.0/include test.c  -lrt -lOpenCL
time ./a.out  # & ./a.out>result1 & ./a.out>result2
#time ./a.out>result1 #& ./a.out>result1 & ./a.out>result2
#time ./a.out>result2 #& ./a.out>result1 & ./a.out>result2

>>>>>>> e9a66bdce415aaa1c33e8344ba8b8d3d4c7ed3df
