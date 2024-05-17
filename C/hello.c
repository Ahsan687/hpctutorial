#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main (int argc, char *argv[]) {

#pragma omp parallel
   {
      int n = omp_get_num_threads();
      int i = omp_get_thread_num();

      printf ("This is thread %d of %d\n",i,n);
   }

   exit(0);
}

