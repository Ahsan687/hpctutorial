#include<stdio.h>

__global__ void add( int a, int b, int *c )
{ 
	*c = a + b; 
} 
int main (void)
{
	int sum;
	int *sum_dev;
	cudaMalloc((void**)&sum_dev, sizeof(int));
	add<<<1,1>>>(1,10,sum_dev);	  
	cudaMemcpy(&sum,sum_dev,sizeof(int),cudaMemcpyDeviceToHost);
	printf("Addition of 1 + 10 = %d\n",sum);
	cudaFree(sum_dev);
}