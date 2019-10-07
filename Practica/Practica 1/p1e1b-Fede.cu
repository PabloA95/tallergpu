/* 
Realizar un programa CUDA que dado un vector V de N números enteros multiplique a 
cada número por una constante C. Realizar dos implementaciones:
a.
C y N deben ser pasados como parámetros al kernel.
b.
C y N deben estar almacenados en la memoria de constantes de la GPU
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <sys/resource.h>

double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

	__constant__ int d_n = 100000000;
	__constant__ int d_c = 456456;
__global__ void cuadradoV_kernel_cuda(int *const arrayV){


  unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id < d_n)
    arrayV[global_id] = arrayV[global_id]*d_c;

}

void cuadradoV_GPU( int arrayV[], const int blk_size, const int n, const int c){
double timetick;

  // Número de bytes de cada uno de nuestros vectores
  int numBytes = n * sizeof(int);

  // Reservamos memoria global del device (GPU) para el array y lo copiamos
  int *cV;
  
  timetick = dwalltime();
  cudaMalloc((void **) &cV, numBytes);
  printf("-> Tiempo de alocacion en memoria global de GPU %f\n", dwalltime() - timetick);  
  timetick = dwalltime();
  cudaMemcpy(cV, arrayV, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  printf("-> Tiempo de copia de memoria CPU =>> GPU %f\n", dwalltime() - timetick);
  
  // Bloque unidimensional de hilos (*blk_size* hilos)
  dim3 dimBlock(blk_size);

  // Grid unidimensional (*ceil(n/blk_size)* bloques)
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

  // Lanzamos ejecución del kernel en la GPU
  //timestamp(start);            // Medimos tiempo de cálculo en GPU
  timetick = dwalltime();
  cuadradoV_kernel_cuda<<<dimGrid, dimBlock>>>(cV);
  cudaError_t error = cudaDeviceSynchronize();
	printf(" Synchronize error %d\n", error);
  printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
  //timestkernelamp(end);

  // Movemos resultado: GPU -> CPU
  timetick = dwalltime();
  cudaMemcpy(arrayV, cV, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);
  
  // Liberamos memoria global del device utilizada
  cudaFree (cV);


}


void init_CPU_array(int array[], const int n)
{
  unsigned int i;
  for(i = 0; i < n; i++) {
    array[i] = (int)i;
  }
}


void cuadradoV_CPU(int arrayV[], const int n, const int c) {
 unsigned int i;
 for(i = 0; i < n; i++) {
     arrayV[i] = arrayV[i]*c;
 }
}


int main(int argc, char *argv[]){

	double timetick;
	int cb = 128;
	const int n = 100000000;
	const int c = 456456;	
	int numBytes = n * sizeof(int);
	timetick = dwalltime();	
	int *vectorV = (int *) malloc(numBytes);
	init_CPU_array(vectorV, n);
  printf("-> Tiempo de alocar memoria e inicializar vectores en CPU %f\n", dwalltime() - timetick);
  timetick = dwalltime();
	cuadradoV_CPU(vectorV, n, c);
  printf("-> Tiempo de ejecucion en CPU %f\n", dwalltime() - timetick);	


	init_CPU_array(vectorV, n);
  cuadradoV_GPU(vectorV,  cb, n , c);


}













