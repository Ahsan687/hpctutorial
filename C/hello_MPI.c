#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
   int rank, size;
   int data = 0;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   if (size < 2) {
      fprintf(stderr,"This program requires at least 2 processes\n");
      MPI_Finalize();
      return 1;
   }

   if (rank == 0) {
      printf ("This program is running on %d nodes\n", size);

      data = 7;
   }

   printf ("I am process: %d, my data = %d\n", rank, data);
   fflush(stdout);
   MPI_Barrier(MPI_COMM_WORLD);
   if (rank == 0) printf ("==============\n");
   MPI_Barrier(MPI_COMM_WORLD);
   

   if (rank == 0) {
      MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
   } else if (rank == 1) {
      MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   }

   printf ("I am process: %d, my data = %d\n", rank, data);

   MPI_Finalize();
   return 0;
}

