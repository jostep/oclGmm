<<<<<<< HEAD
__kernel void square(__global int* input, int a, int b,__global int*output){
    
    int i=get_global_id(0);
    if(i<10*1024*1024){
=======
#define size 50
__kernel void square(__global int* input, int a, int b,__global int*output){
    
    int i=get_global_id(0);
    if(i<size*1024*1024){
>>>>>>> e9a66bdce415aaa1c33e8344ba8b8d3d4c7ed3df
        output[i]=input[i]*input[i]+a+b+15;
    }
}

__kernel void add(__global int*input,__global int* output){
    
    int i=get_global_id(0);
    if(i<size*1024*1024){
        output[i]=output[i]+input[i];
    }

}


__kernel void longSize(__global int* input, int a, long b,__global int*output){
    
    int i=get_global_id(0);
    if(i<1*1024*1024){
        output[i]=sizeof(b);
    }
}

