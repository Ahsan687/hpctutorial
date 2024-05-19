#include<stdio.h>
#include<unistd.h>
#include<omp.h>

int main(int argc,char*argv[])
{
	int id=0;
	omp_set_num_threads(4);
	#pragma omp parallel private(id)
	{
		id=omp_get_thread_num();
		//thread sleeps for thread id seconds
		sleep(id);
		//All threads wait here
		#pragma omp barrier
		printf("Thread Id:%d\n",id);
	}
}
