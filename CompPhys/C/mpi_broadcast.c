#include<mpi.h>
#include<stdio.h>

int main(int argc,char*argv[])
{

	int N=0,rank,size;
	MPI_Init(&argc,&argv);
	//get rank and number of processes
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	//accept inputs on rank 0				
	if(rank==0)
	{
		printf("Enter value of N\n");
		scanf("%d",&N);	
	}

	//Wait till all processes come here
	MPI_Barrier(MPI_COMM_WORLD);
	//Braodcast value of N; It is blocking call
	MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);

	printf("Rank:%d\tN:%d\n",rank,N);
	//clear MPI env
	MPI_Finalize();
}

