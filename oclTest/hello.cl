__kernel void square(__global int* input, __global int*output){
    
    int i=get_global_id(0);
    if(i<10){
        output[i]=input[i]*input[i];
    }
}
