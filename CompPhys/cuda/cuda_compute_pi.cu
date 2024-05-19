#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#define N 1000000000




//compute value of PI on GPU
__global__ void compute_pi(float *estimate, curandState *states,int subN) 
{
	//find the index of the thread
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int inside = 0;
	float x, y;


	//Initialize CURAND
	curand_init(1234, tid, 0, &states[tid]);  

	//generate defined random pair of x and y
	for(int i = 0; i < subN; i++) 
	{
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);

		// increment the count if x & y is in the circle.
		inside += (x*x + y*y <= 1.0f); 
	}
	// compute PI on each thread and return the value
	estimate[tid] = 4.0f * inside / (float) subN; 
}


int main (int argc, char *argv[]) {
	clock_t start, stop;

	//define number of blocks and threads per block
	int blocks=100,threads_per_block=100;
	float pi_host[blocks * threads_per_block];
	float *pi_dev,pi=0;


	int subN=N/(blocks*threads_per_block);
	curandState *devStates;

	//start the timer
	start = clock();

	// allocate device mem. for counts
	cudaMalloc((void **) &pi_dev, blocks * threads_per_block * sizeof(float)); 
	cudaMalloc( (void **)&devStates, threads_per_block * blocks * sizeof(curandState) );

	//call GPU function to estimate value of PI
	compute_pi<<<blocks, threads_per_block>>>(pi_dev, devStates,subN);

	//copy the results from GPU (device) to CPU (host)
	cudaMemcpy(pi_host, pi_dev, blocks * threads_per_block * sizeof(float), cudaMemcpyDeviceToHost); // return results 
	
	//find average of all values of PI from each thread
	for(int i = 0; i < blocks * threads_per_block; i++) 
	{
		pi += pi_host[i];
	}
	pi /= (blocks * threads_per_block);

	//stop the timer
	stop = clock();


	
	printf("PI:%f\tError:%f\n", pi, pi - M_PI);
	
	printf("Execution time %f \n", (stop-start)/(float)CLOCKS_PER_SEC);
	return 0;
}

