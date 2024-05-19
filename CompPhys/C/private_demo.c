#include<stdio.h>
#include<omp.h>

int main(int argc,char*argv[])
{
  int fprivate=10,lprivate=0,priv=100,i=0;
  
  omp_set_num_threads(8);
  //private will create separate instance of a variable for each thread
  //first private will create separate instance of a variable by assigning default value to it for each thread
#pragma omp parallel private(priv) firstprivate(fprivate) 
  {
  printf("Private:%d\tFirstPrivate:%d\n",priv, fprivate);
  //last private will copy value of the last thread 
  // outside the parallel region
#pragma omp parallel for lastprivate(lprivate)
  for(i=0;i<10;i++)
    {
      lprivate=i;
    }
  
}
  
  printf("Last Private:%d\n",lprivate);
}

