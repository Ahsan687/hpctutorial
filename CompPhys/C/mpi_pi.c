#include <mpi.h>
#include <stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#define N 1000000000
int main(int argc, char** argv) {
    // Initialize the MPI environment
	MPI_Init(&argc, &argv);
    	// Get the number of processes
    	int size=0,rank=0,subN=0,inside=0,tmp=0,start=0,end=0,i=0,id=0;
	double x=0.0,y=0.0,pi=0.0,etime=0.0;
	time_t startt, endt;
	//start timer
	time(&startt);

    	MPI_Comm_size(MPI_COMM_WORLD, &size);
    	// Get the rank of the process
    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	subN=N/size; // no broadcast

	start=rank*subN;
	end=(rank+1)*subN;
	srand(time(NULL)*(rank+1));

	tmp=0;
	//generate subN points for each process
	for(i=start;i<end;i++)
	{
		//generate x and y randomly
		x=((double)rand()/RAND_MAX);
		y=((double)rand()/RAND_MAX);
		//check point is inside the circle or not
		if( (x*x+y*y)<1.0)
		{
			//increment the count
			tmp++;
		}
	}
	printf("Rank %d\t Points Inside %d\n",rank,tmp);
	//reduce all results from all processors to processor 0
	MPI_Reduce(&tmp,&inside,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD); // 1 is common length of arrays
	
	if(rank==0)
	{

		printf("\nRank %d\t Points Inside %d\n",rank,inside);
		pi=4*((double)inside/N);

		printf("PI:%f\t Error:%f\n",pi,M_PI-pi);
	}

	
	time(&endt); 
	etime=(double)(endt - startt); 
	//printf("Execution time :%f\n",etime);	
    // Clear/delete the MPI environment.
    MPI_Finalize();
}
