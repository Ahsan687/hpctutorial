// Program to compute a matrix whose each element is row! x col!
#include <stdio.h>
#include <stdlib.h>

void usage(char *progName) {
   printf ("USAGE: %s <dimention of matrix>\n", progName);
}

double fact (int n) {
   return (n == 0? 1.0 : (double)n * fact (n-1));
}

int main (int argc, char **argv) {

   if (argc != 2) {
      usage(argv[0]);
      return (1);
   }

   int N = atoi (argv[1]);
   printf ("Dimension of the matrix: %d\n",N);

   /* printf ("Factorial of %d is: %lf\n", N, fact(N)); */

   // Allocate memory for the matrix
   double **M = malloc (N * sizeof *M);
   M[0] = malloc (N * N * sizeof **M);

   int row, col;
   for (row=1; row<N; row++)
      M[row] = M[0] + (sizeof **M)*row;

   // Compute the matrix
#pragma omp parallel for private(col)
   for (row=0; row<N; row++) {
      for (col=0; col<N; col++) {
         M[row][col] = fact(row+1) * fact(col+1); // row, col start 0 -> 1
      }
   }
   
   // Print the matrix
   for (row=0; row<N; row++) {
      for (col=0; col<N; col++) {
         printf ("%12g\t",M[row][col]);
      }
      printf ("\n");
   }

   return 0;
}

