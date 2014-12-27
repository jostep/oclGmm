#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include "../gmm.h"


#define FALSE 1
#define TRUE 0
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0X100000)
#define testSize 50*1024*1024
#define corun 1
int main(){
    
    char vendor[1024];
    cl_uint num_platform; 
    cl_device_id devId[10];
    cl_uint num_Dev;
    cl_context context;
    cl_command_queue cqueue;
    cl_program program,program2;
    cl_kernel kernel[corun];
    cl_ulong mem;
    cl_ulong localMem;
    size_t global=testSize*sizeof(int);
    size_t local=1024;
    char devName[1024];
    int i=0,j=0,k=0; 
    int *data2=(int*)malloc(testSize*sizeof(int));
    int *result=(int*)malloc(testSize*sizeof(int)*corun);
    int value=44;
    char *nvidia="NVIDIA Corporation";
    int flag=FALSE;
    int flag_cal[corun];
    int   para1=3;
    int para2=8;
    cl_device_id device;
    cl_device_info param_name;
    cl_int errcode_CC;
    cl_int errcode_CQ;
    cl_int errcode_CP;
    cl_int errcode_CK;
    cl_int errcode_BP;
    cl_int errcode_DEBUG;
    cl_int errcode_COPY;
    cl_mem buffer[corun];
  
    memset(flag_cal,0,corun);
    memset(result,0,testSize*sizeof(int)*corun);
    for(j=0;j<testSize;j++){
        data2[j]=j;
    }


    FILE*fp;
    char fileName[]="./hello.cl";
    char *source_str;
    size_t source_size;
    fp=fopen(fileName,"r");
    if(!fp){
        printf("failed to the open the openCL file");
        exit(1);
    }
    source_str=(char *)malloc(MAX_SOURCE_SIZE);
    source_size=fread(source_str,1,MAX_SOURCE_SIZE,fp);
    fclose(fp);

    FILE*fp2;
    char fileName2[]="./goodbye.cl";
    char *source_str2;
    size_t source_size2;
    fp2=fopen(fileName2,"r");
    if(!fp2){
        printf("failed to the open the openCL file");
        exit(1);
    }
    struct timeval start,end; 
    gettimeofday(&start,NULL);
    source_str2=(char *)malloc(MAX_SOURCE_SIZE);
    source_size2=fread(source_str2,1,MAX_SOURCE_SIZE,fp2);
    fclose(fp2);




    cl_int errcode_CB;
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
            context=clCreateContext(NULL,1,devId,NULL,NULL,&errcode_CC);
            //printf("short debug: devId[0]%p,errcode_CC%d\n",devId,errcode_CC);
            if(errcode_CC!=CL_SUCCESS){
                printf("Context Creating Failed!\n");
            }
           cqueue=clCreateCommandQueue(context,devId[0],CL_QUEUE_PROFILING_ENABLE,&errcode_CQ); 
            program=clCreateProgramWithSource(context,1,(const char**)&source_str,(const size_t*)&source_size, &errcode_CP);
            if(errcode_CP!=CL_SUCCESS){
                printf("unable to the load the program  %p\n",errcode_CP);
            }
            errcode_BP=clBuildProgram(program,1,devId,NULL,NULL,NULL);
            if(errcode_BP!=CL_SUCCESS){
                printf("unable to build the program\n");
            }

           for(k=0;k<corun;k++){
                printf("current%p, unsigned long%p\n",buffer[2*k],(unsigned long)buffer[2*k]);
                buffer[2*k]=clCreateBuffer(context,CL_MEM_READ_WRITE,testSize*sizeof(int),NULL,&errcode_CB);
                if(errcode_CB!=CL_SUCCESS){
                    printf("Buffer Creating failed!  %d \n",errcode_CB);
                }
                printf("current%p, unsigned long%p\n",buffer[2*k],(unsigned long)buffer[2*k]);
                buffer[2*k+1]=clCreateBuffer(context,CL_MEM_READ_WRITE,testSize*sizeof(int),NULL,&errcode_CB);
                if(errcode_CB!=CL_SUCCESS){
                    printf("Buffer Creating failed!  %d \n",errcode_CB);
                }
                kernel[k]=clCreateKernel(program,"square",&errcode_CK);
                if(errcode_CK!=CL_SUCCESS){
                printf("kernel creating failure\n");
                }
                if(clEnqueueWriteBuffer(cqueue,buffer[2*k],CL_TRUE,0,sizeof(int)*testSize,data2,0,NULL,NULL)!=CL_SUCCESS){
                    printf("write buffer failed\n");
                }
                if(clEnqueueWriteBuffer(cqueue,buffer[2*k+1],CL_TRUE,0,sizeof(int)*testSize,(result+k*testSize),0,NULL,NULL)!=CL_SUCCESS){
                    printf("write buffer failed\n");
                }
                if(clReference(0,2)!=CL_SUCCESS){
                    printf("unable to set the arg ref\n");
                }
                if(clReference(3,1)!=CL_SUCCESS){
                    printf("unable to set the arg ref\n");
                }
                if(clSetKernelArg(kernel[k],0,sizeof(cl_mem),&buffer[2*k])!=CL_SUCCESS){
                    printf("unable to set the arg 0\n");
                }   
                if(clSetKernelArg(kernel[k],1,sizeof(int),&para1)!=CL_SUCCESS){
                    printf("unable to set the arg 1\n");
                }   
                if(clSetKernelArg(kernel[k],2,sizeof(int),&para2)!=CL_SUCCESS){
                    printf("unable to set the arg 2\n");
                }   
                if(clSetKernelArg(kernel[k],3,sizeof(cl_mem),&buffer[2*k+1])!=CL_SUCCESS){
                    printf("unable to set the arg 3\n");
                }
                /*errcode_DEBUG=clGetKernelWorkGroupInfo(kernel[k],devId[0],CL_KERNEL_WORK_GROUP_SIZE,sizeof(local),&local,NULL);
                printf("the local is %d\n",local);
                if(errcode_DEBUG!=CL_SUCCESS){
                    printf("Failed to get the work info\n");
                }*/
                if(clEnqueueNDRangeKernel(cqueue,kernel[k],1,NULL,&global,&local,0,NULL,NULL)!=CL_SUCCESS){
                    printf("kernel launched failed\n");
                }

                errcode_COPY=clEnqueueCopyBuffer(cqueue,buffer[2*k+1],buffer[2*k],0,0,sizeof(int)*testSize,0,NULL,NULL);
                if(errcode_COPY!=CL_SUCCESS){
                    printf("incorrect copying, with err%d\n",errcode_COPY);
                }
                if(clEnqueueReadBuffer(cqueue,buffer[2*k+1],CL_TRUE,0,sizeof(int)*testSize,(result+k*testSize),0,NULL,NULL)!=CL_SUCCESS){
                    printf("read buffer error\n");
                }
                //answer check
                for(j=0;j<testSize;j++){
                    if(*(result+k*testSize+j)!=data2[j]*data2[j]){
                        flag_cal[k]=FALSE;
                    }     
                }
                printf("show %dth result %d\n",k,*(result+k*testSize+5));
                if(flag_cal==FALSE){
                    printf("the %dth calculation is incorrect\n",k);
                }
           }
            /*program2=clCreateProgramWithSource(context,1,(const char**)&source_str2,(const size_t*)&source_size2, errcode_CP);
            if(errcode_CP!=CL_SUCCESS){
                printf("unable to the load the program  %p\n",errcode_CP);
            }
            errcode_BP=clBuildProgram(program2,1,&devId[0],NULL,NULL,NULL);
            if(errcode_BP==CL_INVALID_DEVICE){
                printf("unable to build the program, INVALID DEV\n");
            }*/
           for(k=0;k<corun;k++){
            
                
                kernel[k]=clCreateKernel(program,"add",errcode_CK);
                if(clReference(0,2)!=CL_SUCCESS){
                    printf("unable to set the arg ref\n");
                }
                if(clReference(1,1)!=CL_SUCCESS){
                    printf("unable to set the arg ref\n");
                }
                if(clSetKernelArg(kernel[k],0,sizeof(cl_mem),&buffer[2*k])!=CL_SUCCESS){
                    printf("unable to set the arg\n");
                }   
                if(clSetKernelArg(kernel[k],1,sizeof(cl_mem),&buffer[2*k+1])!=CL_SUCCESS){
                    printf("unable to set the arg\n");
                }
                /*errcode_DEBUG=clGetKernelWorkGroupInfo(kernel[k],devId[0],CL_KERNEL_WORK_GROUP_SIZE,sizeof(local),&local,NULL);
                if(errcode_DEBUG!=CL_SUCCESS){
                    printf("Failed to get the work info\n");
                }*/
                if(clEnqueueNDRangeKernel(cqueue,kernel[k],1,NULL,&global,&local,0,NULL,NULL)!=CL_SUCCESS){
                    printf("kernel launched failed\n");
                }
                errcode_COPY=clEnqueueCopyBuffer(cqueue,buffer[2*k+1],buffer[2*k],0,0,sizeof(int)*testSize,0,NULL,NULL);
                if(errcode_COPY!=CL_SUCCESS){
                    printf("incorrect copying, with err%d\n",errcode_COPY);
                }
                if(clEnqueueReadBuffer(cqueue,buffer[2*k],CL_TRUE,0,sizeof(int)*testSize,(result+k*testSize),0,NULL,NULL)!=CL_SUCCESS){
                    printf("read buffer error\n");
                }
                for(j=0;j<testSize;j++){
                    if(*(result+k*testSize+j)!=data2[j]*data2[j]){
                        flag_cal[k]=FALSE;
                    }     
                }
                printf("show %dth result %d\n",k,*(result+k*testSize+5));
                if(flag_cal==FALSE){
                    printf("the %dth calculation is incorrect\n",k);
                }

           
           }


                
            if(CL_SUCCESS!=clFinish(cqueue)){
                printf("unsuccessfully quited\n");
            }


            for(k=0;k<2*corun;k++){
                if(CL_SUCCESS!=clReleaseMemObject(buffer[k])){
                    printf("Buffer Deleting unsuccessful\n");
                }
            }
        }
        //clGetDeviceInfo(devId[0],CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem),&mem,NULL);
        flag=FALSE;    
    }
    gettimeofday(&end,NULL);
    unsigned long total=(end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec;
    printf("========[Total time]========= |%ld \n",total);
    return 0;
}
