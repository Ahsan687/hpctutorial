#include<stdio.h>
#include<omp.h>
//include omp header file
int main(int argc,char*argv[])
{
  omp_set_num_threads(4);
  //#pragma omp parallel num_threads(10) //use ten threads in below parallel region
#pragma omp parallel //use no of threads given globally either by commandline or by omp_set_num_threads(4)
  {
    //get id of each thread
    int id=omp_get_thread_num();
    
    //print id of each thread
    printf("%d\t Hello World\n",id);
  }
}
