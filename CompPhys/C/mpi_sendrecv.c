#include<mpi.h>
#include<stdio.h>
#define N 10
int main(int argc,char*argv[])
{

	int data[N],rank=0,size=0,subN=0,i=0;
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
		for(i=1;i<size;i++)
			MPI_Send(&data[i*subN],subN,MPI_INT,i,0,MPI_COMM_WORLD);
	}

	//Wait till all processes come here
	MPI_Barrier(MPI_COMM_WORLD);
	//Braodcast value of N; It is blocking call
	MPI_Bcast(&subN,1,MPI_INT,0,MPI_COMM_WORLD);

	if(rank>0)
	{
		MPI_Recv(&data,subN,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		for(i=0;i<subN;i++)
		{
			printf("%d\t%d\n",rank,data[i]);
			// note that this creates size copies of data, most of which is wasted. 
			// correct way is to use malloc for each process.
		}
	}
	//clear MPI env
	MPI_Finalize();
}


