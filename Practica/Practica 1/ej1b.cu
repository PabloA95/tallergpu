/* Realizar un programa CUDA que dado un vector V de N números enteros multiplique a
* cada número por una constante C. Realizar dos implementaciones:
* a. C y N deben ser pasados como parámetros al kernel.
* b. C y N deben estar almacenados en la memoria de constantes de la GPU.
*/

#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef _INT_
typedef int basetype;     // Tipo para elementos: int
#define labelelem    "ints"
#elif _DOUBLE_
typedef double basetype;  // Tipo para elementos: double
#define labelelem    "doubles"
#else
typedef float basetype;   // Tipo para elementos: float     PREDETERMINADO
#define labelelem    "floats"
#endif

/*
void kernel_multiplicar(const int c, int *vector)
{
	int n=0;
	for (int i=1;)
}
*/


double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


/*
  Función para inicializar el vector que vamos a utilizar
*/
void init_CPU_array(basetype array[], const unsigned int n)
{
  unsigned int i;
  for(i = 0; i < n; i++) {
    array[i] = (basetype)i;
  }
}

void multiplicarV_CPU(basetype vec[], const unsigned int c, const unsigned int n) {
	for(int i=0; i<n; i++){
		vec[i]=vec[i]*c;
	}
}

__constant__ int d_n=102400;
__constant__ int d_c=9;
__global__ void multiplicacionV_kernel_cuda(basetype *const arrayV){

  unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id < d_n)
    arrayV[global_id] = arrayV[global_id]*d_c;

}



void multiplicacionV_GPU( basetype arrayV[], const unsigned int n, const unsigned int blk_size, const int c){
	double timetick;

  // Número de bytes de cada uno de nuestros vectores
  unsigned int numBytes = n * sizeof(basetype);

	cudaError_t error;

  // Reservamos memoria global del device (GPU) para el array y lo copiamos
  basetype *cV;
  
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
  multiplicacionV_kernel_cuda<<<dimGrid, dimBlock>>>(cV);
  error = cudaDeviceSynchronize();
  printf("Synchronyse error: %d\n", error);
  printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
  //timestamp(end);

  // Movemos resultado: GPU -> CPU
  timetick = dwalltime();
  cudaMemcpy(arrayV, cV, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);
  
  // Liberamos memoria global del device utilizada
  cudaFree (cV);


}




int main(int argc, char **argv)
{
	double timetick;

	
	//Ejecucion en CPU
	basetype *vec;
	int c=9;
	int n=102400;
	
	timetick = dwalltime();
	vec=(basetype *) malloc(sizeof(basetype)*n);
	init_CPU_array(vec, n);
  printf("-> Tiempo de inicializacion de vector en CPU %f\n", dwalltime() - timetick);
  
  timetick = dwalltime();
	multiplicarV_CPU(vec,c,n);
  printf("-> Tiempo de ejecucion en CPU %f\n", dwalltime() - timetick);



	//Ejecucion en GPU
	int cb=64;
	 //Inicializa nuevamente el vector para realizar la ejecucion en GPU
  init_CPU_array(vec, n);
  
  // Ejecutamos cuadradoV en GPU
  multiplicacionV_GPU(vec, n,  cb, c);

  //Chequea si el resultado obtenido en la GPU es correcto
  //check_array(vec,n); 
  
  free(vec);
	
	return 0;
}
