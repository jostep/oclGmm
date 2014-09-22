#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <inttypes.h>
#include <string.h>



#define FALSE 1
#define TRUE 0
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0X100000)


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
    int data[10000]={5};
    int value=44;
    char *nvidia="NVIDIA Corporation";
    int flag=FALSE;
    cl_device_id device;
    cl_device_info param_name;
    cl_int *errcode_CC=NULL;
    cl_int *errcode_CQ=NULL;
    cl_int *errcode_CP=NULL;
    cl_int *errcode_CK=NULL;
    cl_mem buffer,buffer2;

    FILE*fp;
    char fileName[]="./hello.cl";
    char *source_str;
    size_t source_size;
    fp=fopen(fileName,"r");
    if(!fp){
        printf("fialed to the open the openCL file");
        exit(1);
    }

    source_str=(char *)malloc(MAX_SOURCE_SIZE);
    source_size=fread(source_str,1,MAX_SOURCE_SIZE,fp);
    fclose(fp);




    cl_int *errcode_CB=NULL;
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
           cqueue=clCreateCommandQueue(context,devId[0],CL_QUEUE_PROFILING_ENABLE,errcode_CQ); 
           buffer=clCreateBuffer(context,CL_MEM_READ_WRITE,10000*sizeof(cl_int),NULL,errcode_CB);
            if(errcode_CB!=CL_SUCCESS){
                printf("Buffer Creating failed!  %p\n",errcode_CB);
            }

            program=clCreateProgramWithSource(context,1,(const char**)&source_str,(const size_t*)&source_size, errcode_CP);
            if(errcode_CP!=CL_SUCCESS){
                printf("unable to the load the program  %p\n",errcode_CP);
            }
            if(clBuildProgram(program,1,devId,NULL,NULL,NULL)!=CL_SUCCESS){
                printf("uanble to build the program\n");
            }
            
            kernel=clCreateKernel(program,"hello",errcode_CK);
            if(errcode_CK!=CL_SUCCESS){
                printf("kernel creating failure\n");
            }


/*            if(clEnqueueFillBuffer(cqueue,buffer,&value,sizeof(int),0,100*sizeof(cl_int),0,NULL,NULL)!=CL_SUCCESS){
                printf("Memseting failed\n");
            }*/
            if(clEnqueueWriteBuffer(cqueue,buffer,CL_TRUE,0,sizeof(int)*10000,data,0,NULL,NULL)!=CL_SUCCESS){
                printf("write buffer failed\n");
            }
            if(oclReference(0,2)!=CL_SUCCESS){
                printf("unable to set the arg ref\n");
            }
            if(clSetKernelArg(kernel,0,sizeof(cl_mem),&buffer)!=CL_SUCCESS){
                printf("unable to set the arg\n");
            }
            if(clEnqueueTask(cqueue,kernel,0,NULL,NULL)!=CL_SUCCESS){
                printf("kernel launch failed\n"); 
            }
            if(CL_SUCCESS!=clReleaseMemObject(buffer)){
                printf("Buffer Deleting unsuccessful");
            }
        }
        
        clGetDeviceInfo(devId[0],CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem),&mem,NULL);
        printf("The global size is %d \n",mem);
        flag=FALSE;    
    }
    return 0;
}
