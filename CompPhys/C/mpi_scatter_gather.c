#include<mpi.h>
#include<stdio.h>
#define N 10
int main(int argc,char*argv[])
{

	int data[N],rank=0,size=0,subN=0,i=0;
	int data1[N];
	MPI_Init(&argc,&argv);
	MPI_Status status;
	//get rank and number of processes
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	//accept inputs on rank 0				
	if(rank==0)
	{
		for(i=0;i<N;i++)
		{
			data[i]=i;
		}		
		subN=N/size;
	}

	//Wait till all processes come here

	//Braodcast value of N; It is blocking call
	MPI_Bcast(&subN,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scatter(&data,subN,MPI_INT,&data1,subN,MPI_INT,0,MPI_COMM_WORLD);


	for(i=0;i<subN;i++)
	{
		printf("Rank:%d\t %d\n",rank,data1[i]);	
		data1[i]*=10;
	}


	MPI_Gather(&data1,subN,MPI_INT,&data,subN,MPI_INT,0,MPI_COMM_WORLD); // 0 is to collect on rank 0
	/* MPI_Allgather(&data1,subN,MPI_INT,&data,subN,MPI_INT,MPI_COMM_WORLD); */
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank==0)
	{
		for(i=0;i<N;i++)
		  printf("Rank:%d\t %d\n",rank,data[i]);	
	}
	//clear MPI env
	MPI_Finalize();
}

