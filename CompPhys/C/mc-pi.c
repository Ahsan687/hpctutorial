#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<omp.h>
//declare number of random points to generate
#define N 1000000000
int main(int argc,char*argv[])
{

	//variable declaration
	int id,inside=0,numthreads=0,subN=0,start=0,end=0,tmp=0;
	double etime=0.0,pi=0.0,x=0.0,y=0.0;
	time_t startt, endt;
	long int i=0;	
	//start timer
	time(&startt); 

	//init seed
	srand(time(NULL)); // should be inside parallel block

	//set number of threads
	omp_set_num_threads(8);

	//start parallel region
	#pragma omp parallel private(id,numthreads,start,end,tmp,x,y,i) shared(inside,subN)
	{
		
		//get thread id
		id=omp_get_thread_num();
		//get total number of threads
		numthreads=omp_get_num_threads();
		//compute total number of points processed by each thread
		subN=N/numthreads;
		//compute start and end for each thread
		start=id*subN;
		end=(id+1)*subN;
		tmp=0;
		//generate subN points for each thread
		for(i=start;i<end;i++)
		{
			//generate x and y randomly
			x=((double)rand_r(&id)/RAND_MAX);
			y=((double)rand_r(&id)/RAND_MAX);
			//check point is inside the circle or not
			if( (x*x+y*y)<1.0)
			{
				//increment the count
				tmp++;
			}


		}
		//add number of points inside the circle 
		#pragma omp critical
		inside+=tmp; // can be replaced using reduction
	}
	//compute value of pi
	pi=4*((double)inside/N);

	//stop timer and compute elapsed time
	time(&endt); 
	etime=(double)(endt - startt); 
	
	//print value of pi
	printf("PI:%f\tError:%f\n",pi,M_PI-pi);
	printf("Execution time :%f\n",etime);
}

