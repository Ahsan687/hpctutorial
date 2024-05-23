#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void usage (char *progName) {
   fprintf (stderr, "USAGE: %s <number>\n", progName);
}

int main (int argc, char **argv) {

   if (argc != 2) {
      usage (argv[0]);
      return (1);
   }

   int n = atoi (argv[1]);
   printf ("To check if this number is prime or not is: %d\n",n);
   
   int d = 1;
   int q = n;
   int isPrime = 1;

   if (n <= 3) {
      printf ("The number %d is prime\n",n);
      return;
   } 

   while (q >= d) {

      d = d + 1;
      q = (int)(n/d);

      if (n%d == 0) {
         isPrime = 0;
         /* printf ("divider = %d, quotient = %d\n",d,q); */
         break;
      }
   }

   if (isPrime)
      printf ("The number %d is prime\n",n);
   else
      printf ("The number %d is NOT prime, it is divisible by %d\n",n,d);

   return;
}
