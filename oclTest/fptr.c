#include <stdio.h>
#include <stdlib.h>


int main(){


    int a;
    long b;
    void * k = malloc(512);

    printf("lets take a look int %d, long %d and void*i %p %p %p",sizeof(int),sizeof(long),&a,&b,k);
}
