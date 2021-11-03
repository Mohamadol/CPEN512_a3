#ifndef MATR_H
#define MATR_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


#if !defined(N)
#define N 4096
#endif

#if !defined(PARALLEL)
#define PARALLEL 0
#endif


#if !defined(N)
#define THREADS_IN_BLOCK 4096
#endif


float *create_matrix(){
  float *matrix = (float *)malloc(sizeof(float) * N * N);
  return matrix;
}


void initialize_matrix(float *data, int random_init, float init_val){
  float max_random_init = 1000.0;
  int max_random_init_int = 1000;

  for(int i=0 ; i<N ; i++){
    for(int j=0 ; j<N ; j++){
      if(random_init)
        data[i * N + j] = (float)rand() / (float)(RAND_MAX/max_random_init) + 1;
      else
        data[i * N + j] = init_val;
    }
  }
}


void initialize_matrix_from_another_matrix(float *dest, float *source){
  for(int i=0 ; i<N ; i++){
    for(int j=0 ; j<N ; j++){
      dest[i * N + j] = source[i * N + j];
    }
  }
}


void print_matrix(float *data){
  for (int i=0 ; i<N ; i++){
    for(int j=0 ; j<N ; j++)
      printf("%.2f  ", (data)[i*N+j]);
    printf("\n");
  }
}


void free_matrix(float *data){
  free(data);
}

#endif
