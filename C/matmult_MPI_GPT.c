#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 3 // Size of the matrices

void matrix_mult(int **A, int **B, int **C, int rows_per_proc, int rank) {
   int i, j, k;
   for (i = 0; i < rows_per_proc; i++) {
      for (j = 0; j < N; j++) {
         C[i][j] = 0;
         for (k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
         }
      }
   }
}

int main(int argc, char *argv[]) {
   int rank, size;
   int i, j;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   int **A, **B, **C;
   int rows_per_proc = N / size;

   // Allocate memory for matrices A, B, and C
   A = (int **)malloc(rows_per_proc * sizeof(int *));
   B = (int **)malloc(N * sizeof(int *));
   C = (int **)malloc(rows_per_proc * sizeof(int *));
   for (i = 0; i < rows_per_proc; i++) {
      A[i] = (int *)malloc(N * sizeof(int));
      C[i] = (int *)malloc(N * sizeof(int));
   }
   for (i = 0; i < N; i++) {
      B[i] = (int *)malloc(N * sizeof(int));
   }

   // Initialize matrices A and B
   for (i = 0; i < rows_per_proc; i++) {
      for (j = 0; j < N; j++) {
         A[i][j] = rank * rows_per_proc + i; // some initial values
      }
   }
   for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
         B[i][j] = i + j; // some initial values
      }
   }

   // Matrix multiplication
   matrix_mult(A, B, C, rows_per_proc, rank);

   // Gather results
   int *recvbuf = (int *)malloc(N * rows_per_proc * sizeof(int));
   MPI_Gather(C[0], rows_per_proc * N, MPI_INT, recvbuf, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

   // Print the result in rank order
   if (rank == 0) {
      printf("Result:\n");
      for (i = 0; i < N; i++) {
         for (j = 0; j < N; j++) {
            printf("%d ", recvbuf[i * N + j]);
         }
         printf("\n");
      }
   }

   // Clean up
   free(recvbuf);
   for (i = 0; i < rows_per_proc; i++) {
      free(A[i]);
      free(C[i]);
   }
   for (i = 0; i < N; i++) {
      free(B[i]);
   }
   free(A);
   free(B);
   free(C);

   MPI_Finalize();
   return 0;
}

