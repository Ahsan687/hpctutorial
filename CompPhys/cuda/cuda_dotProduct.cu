#include<stdio.h>

__global__ void dotProduct(int*a,int*b,int*c,int n)
{
	int tid=0;
	//get thread id
	tid=threadIdx.x+(blockIdx.x*blockDim.x);
	if(tid<n)
	{
		c[tid]=a[tid]*b[tid];
	}
}

int main(int argc,char*argv[])
{
	int *a,*b,*dotprodtmp,n=1000,i=0,dotproduct=0;
	int *a_dev,*b_dev,*dotprod_dev;
	int numblocks=0,threadsPerBlock=256;
	size_t datasize=sizeof(int)*n;
	clock_t start, stop;

	a=(int*)malloc(datasize);
	b=(int*)malloc(datasize);
	dotprodtmp=(int*)malloc(datasize);

	//allocate memory on device
	cudaMalloc(&a_dev,datasize);
	cudaMalloc(&b_dev,datasize);
	cudaMalloc(&dotprod_dev,datasize);

	//init arrays
	for(i=0;i<n;i++)
	{
		a[i]=i;
		b[i]=i;
	}
        start = clock();
	
	//copy data from CPU(host) to device
	cudaMemcpy(a_dev,a,datasize,cudaMemcpyHostToDevice);	
	cudaMemcpy(b_dev,b,datasize,cudaMemcpyHostToDevice);	
	//get number of blocks
	numblocks=(int)(n/threadsPerBlock)+1;

	//call gpu function to compute dot product
	dotProduct<<<numblocks,threadsPerBlock>>>(a_dev,b_dev,dotprod_dev,n);
	//copy results from device to host
	cudaMemcpy(dotprodtmp,dotprod_dev,datasize,cudaMemcpyDeviceToHost);
	
	//sum partial dot products
	dotproduct=0;
	for(i=0;i<n;i++)
	{
		dotproduct+=dotprodtmp[i];
	}
	stop = clock();
	printf("Dot Product is :%d\n",dotproduct);
	printf("GPU Execution time %f \n", (stop-start)/(float)CLOCKS_PER_SEC);
}
