#include<stdio.h>
#include<omp.h>
#define N 10
int main (int argc,char*argv[])
{
	int a[N],sum=0,i=0;
	for(i=0;i<N;i++)
		a[i]=i;
	omp_set_num_threads(4);
	#pragma omp parallel
	{
		//reduction operation, sum will contain 
		//sum of all values by each threads
		#pragma omp parallel for reduction(+:sum)
		for(i=0;i<N;i++)
		{
		  sum+=a[i]; // note these sum variables will contain partial sums in each thread
		}
	}
	printf("Sum:%d\n",sum);
}

