#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <inttypes.h>
#include <string.h>

#define FALSE 1
#define TRUE 0

int main(){

    char vendor[1024];
    cl_uint num_platform; 
    cl_device_id devId[10];
    cl_uint num_Dev;
    unsigned int devUsed;
    cl_context context;
    cl_command_queue cqueue;
    cl_program program;
    cl_kernel kernel;
    cl_ulong mem;
    cl_ulong localMem;
    char devName[1024];
    int i=0; 
    char *nvidia="NVIDIA Corporation";
    int flag=FALSE;
    cl_device_id device;
    cl_device_info param_name;
    cl_int *errcode_CC=NULL;
    clGetPlatformIDs(NULL,0,&num_platform);
    printf("Currently, we have %d platforms;\n",num_platform);
    
    cl_platform_id * platform=(cl_platform_id*) malloc(num_platform*sizeof(cl_platform_id));
    clGetPlatformIDs(num_platform,platform,NULL); 
    
    for (i=0;i<num_platform;i++){
    
        clGetPlatformInfo(platform[i],CL_PLATFORM_VENDOR,sizeof(vendor),vendor,NULL);
        printf("\tPlatform Vendor:\t%s\n",vendor);
        if(strcmp(vendor,nvidia)==0){
            printf("We've got the right one\n");
            flag=TRUE;
        }
        




        clGetDeviceIDs(platform[i],CL_DEVICE_TYPE_ALL,sizeof(devId),devId,&num_Dev);
        printf("number of devices %u\n",num_Dev); 
   
        clGetDeviceInfo(devId[0], CL_DEVICE_NAME, sizeof(devName),devName,NULL);
        printf("\tDevice Name:\t%s\n",devName);
        
        if(flag==TRUE){
            context=clCreateContext(NULL,1,devId,NULL,NULL,errcode_CC);
            if(errcode_CC==CL_SUCCESS){
                printf("Context Creating Success!\n");
            }

        }
    
        clGetDeviceInfo(devId[0],CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem),&mem,NULL);
        printf("The global size is %0.00f \n",(double)mem/1024576);
        flag=FALSE;
    }
    
    return 0;
}
