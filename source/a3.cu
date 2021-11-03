#include "matr.h"


#define THREADS_IN_BLOCK 1024

__global__ void row_normalization_gpu(float * A, int i){
  int threadId = blockIdx.y * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
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


void GE_cuda(float *B){
  float *A;

  //Allocate memory on device
  cudaMalloc(&A, sizeof(float) * N * N);
  //copy the matrix from host to device
  cudaMemcpy(A, (void *)B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

  int i=0;
  int j;
  for(; i<N ; i++){
    j=i;
    int blocks = (int)((N - i)/(float)THREADS_IN_BLOCK);
    row_normalization_gpu<<blocks, THREADS_IN_BLOCK>>(A, i);
    cudaDeviceSynchronize();
    blocks = (int)((N - i - 1) * (N - i)/(float)THREADS_IN_BLOCK);
    row_elimination_gpu<<<num_blocks, THREADS_IN_BLOCK>>>(A, i);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(B, (void *)A, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
  cudaFree(A);
}


int main(){
  float *A = create_matrix();
  float *B = create_matrix();
  initialize_matrix(A, 1, 0);
  initialize_matrix_from_another_matrix(B, A);
  GE_cuda(B);
  return 0;
}