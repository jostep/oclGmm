__kernel void add(__global int*input,__global int* output){
    
    int i=get_global_id(0);
    if(i<1*1024*1024){
        output[i]=output[i]+input[i];
    }

}
