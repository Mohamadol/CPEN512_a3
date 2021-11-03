#include <sys/time.h>
#include "matr.h"


#define THREADS_IN_BLOCK 1024


__global__ void row_normalization_gpu(float * A, int i){
  int threadId = blockIdx.y * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= N)
    return;
  float alpha = A[i * N + i];
  __syncthreads();
  A[i * N + i * threadId] /= alpha;
  return;
}


__global__ void row_elimination_gpu(float *A, int i){
  int threadId = blockIdx.y * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  int r = i + threadId / (N - i) + 1;
  int c = i + threadId % (N - i);
  //Check if thread should actually perform work
  int total_elements = (N - i - 1) * (N - i);
  if(threadId >= total_elements)
    return;
  A[r * N + c] = A[r * N + i] / A[i * N + i] * A[i * N + c];
  return;
}


double GE_cuda(float *B){
  float *A;

  //Allocate memory on device
  cudaMalloc(&A, sizeof(float) * N * N);
  //copy the matrix from host to device
  cudaMemcpy(A, (void *)B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

  struct timeval begin, end;
  gettimeofday(&begin, 0);

  int i=0;
  for(; i<N ; i++){
    int blocks = (int)((N - i)/(float)THREADS_IN_BLOCK);
    row_normalization_gpu<<<blocks, THREADS_IN_BLOCK>>>(A, i);
    cudaDeviceSynchronize();
    blocks = (int)((N - i - 1) * (N - i)/(float)THREADS_IN_BLOCK);
    row_elimination_gpu<<<blocks, THREADS_IN_BLOCK>>>(A, i);
    cudaDeviceSynchronize();
  }

  gettimeofday(&end, 0);
  double duration = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;

  cudaMemcpy(B, (void *)A, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
  cudaFree(A);

  return duration;
}


int main(){
  float *A = create_matrix();
  float *B = create_matrix();
  initialize_matrix(A, 1, 0);
  initialize_matrix_from_another_matrix(B, A);
  double duration_gpu = GE_cuda(B);
  printf("duration was %.4f", duration);
  return 0;
}
