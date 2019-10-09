#include <stdio.h>
#include <stdlib.h>

#ifdef _FLOAT_
typedef float basetype;
#define labelelem    "floats"
#elif _DOUBLE_
typedef double basetype;
#define labelelem    "doubles"
#else
typedef int basetype;// Tipo para elementos: Int     PREDETERMINADO
#define labelelem    "ints"
#endif

/* Cosas para calcular el tiempo */
#include <sys/time.h>
#include <sys/resource.h>

double dwalltime(){
        double sec;
        struct timeval tv;
        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

/* Fin de cosas para calcular el tiempo */


void init_CPU_array(basetype array[], const unsigned int n)
{
        unsigned int i;
        for(i = 0; i < n; i++) {
                array[i] = (basetype)i;
        }
}

void transponer_matriz_CPU(basetype *m, basetype *t, int n)
{
	for (int i = 0; i < n; ++i){
		for (int j = 0; j < n; ++j){
			t[i+j*n] = m[i*n+j];
		}
	}

}

__global__ void transponer_kernel_cuda(basetype * m, basetype * t, const int n){

	unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_id < n*n){
		//printf("soy %d\n", global_id);
		t[global_id] = m[global_id];

	}
}

int transponer_matriz_GPU(basetype *m,basetype *t, int n)
{
	double timetick;
int blk_size=64;
cudaError_t error;
	// Número de bytes de cada uno de nuestros vectores
	unsigned int numBytes = n *n * sizeof(basetype);

	// Reservamos memoria global del device (GPU) para el array y lo copiamos
	basetype *gm, *gt;
	
	timetick = dwalltime();
	cudaMalloc((void **) &gm, numBytes);
	cudaMalloc((void **) &gt, numBytes);

	printf("-> Tiempo de alocacion en memoria global de GPU %f\n", dwalltime() - timetick);	
	timetick = dwalltime();
	cudaMemcpy(gm, m, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

	printf("-> Tiempo de copia de memoria CPU =>> GPU %f\n", dwalltime() - timetick);
	


	// Bloque unidimensional de hilos (*blk_size* hilos)
	dim3 dimBlock(blk_size);

	// Grid unidimensional (*ceil(n/blk_size)* bloques)
	dim3 dimGrid((((n + dimBlock.x - 1) / dimBlock.x)));

	//printf("%d %d",dimBlock.x,dimGrid.x);

	// Lanzamos ejecución del kernel en la GPU
	//timestamp(start);						// Medimos tiempo de cálculo en GPU
	timetick = dwalltime();
	transponer_kernel_cuda<<<dimGrid, dimBlock>>>(gm,gt,n);
	error=cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(error));
	printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
	//timestamp(end);

	// Movemos resultado: GPU -> CPU
	timetick = dwalltime();
	cudaMemcpy(t, gt, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
	printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);
	
	// Liberamos memoria global del device utilizada
	cudaFree (gm);
cudaFree (gt);

return 0;
}



int main(int argc, char const *argv[])
{
	basetype *m;
	basetype *t;

	int b=3; //=1024;
	int n=b*b;

	double timetick;
	
	// Aloco memoria para los vectores
	timetick = dwalltime();
	m=(basetype *) malloc(n*sizeof(basetype));
	t=(basetype *) malloc(n*sizeof(basetype));
	printf("-> Tiempo de allocacion de vectores en CPU %f\n", dwalltime() - timetick);

	// Inicializo los arreglos
	// timetick = dwalltime();
	init_CPU_array(m,n);
	// printf("-> Tiempo de inicializacion de vectores en CPU %f\n", dwalltime() - timetick);

	// Sumo los arreglos
	timetick = dwalltime();
	transponer_matriz_CPU(m,t,b);
	printf("-> Tiempo de suma de vectores en CPU %f\n", dwalltime() - timetick);


	transponer_matriz_GPU(m,t,b);

/*	for (int i = 0; i < n; ++i)
	{
		printf("%d\n", m[i]);
	}
	printf("%p\n", t);
	for (int i = 0; i < n; ++i)
	{
		printf("%d\n", t[i]);
	}
*/

	return 0;
}
