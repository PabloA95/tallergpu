#include <stdio.h>
#include <stdlib.h>

#ifdef _INT_
typedef int basetype;		 // Tipo para elementos: int
#define labelelem		"ints"
#elif _DOUBLE_
typedef double basetype;	// Tipo para elementos: double
#define labelelem		"doubles"
#else
typedef float basetype;	 // Tipo para elementos: float		 PREDETERMINADO
#define labelelem		"floats"
#endif

#include <sys/time.h>
#include <sys/resource.h>

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

void init2_CPU_array(basetype array[], const unsigned int n)
{
	unsigned int i;
	for(i = 0; i < n; i++) {
		array[i] = ((basetype)i) * 2;
	}
}

int sum_CPU_array(basetype *a, basetype *b, basetype *c, int n)
{
	for (int i=0; i<n; i++){
		c[i]=a[i]+b[i];
	}
	return 0;
}

int print_CPU_array(basetype *c, int n)
{
	for (int i=0; i<n; i++){
		printf("%f\n",c[i]);
	}
	return 0;
}


__global__ void suma_kernel_cuda(basetype * a, basetype * b, basetype *c,	 const int n){

	unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_id < n)
		c[global_id] = a[global_id]+b[global_id];
	 else
	printf("%d\n",global_id);
}

int sum_GPU_array(basetype *a,basetype *b,basetype *c,int n,int cb)
{


double timetick;
int blk_size=cb;
cudaError_t error;
	// Número de bytes de cada uno de nuestros vectores
	unsigned int numBytes = n * sizeof(basetype);

	// Reservamos memoria global del device (GPU) para el array y lo copiamos
	basetype *ga, *gb, *gc;
	
	timetick = dwalltime();
	cudaMalloc((void **) &ga, numBytes);
cudaMalloc((void **) &gb, numBytes);
cudaMalloc((void **) &gc, numBytes);
	printf("-> Tiempo de alocacion en memoria global de GPU %f\n", dwalltime() - timetick);	
	timetick = dwalltime();
	cudaMemcpy(ga, a, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
	cudaMemcpy(gb, b, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU

	printf("-> Tiempo de copia de memoria CPU =>> GPU %f\n", dwalltime() - timetick);
	
	// Bloque unidimensional de hilos (*blk_size* hilos)
	dim3 dimBlock(blk_size);

	// Grid unidimensional (*ceil(n/blk_size)* bloques)
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	//printf("%d %d",dimBlock.x,dimGrid.x);

	// Lanzamos ejecución del kernel en la GPU
	//timestamp(start);						// Medimos tiempo de cálculo en GPU
	timetick = dwalltime();
	suma_kernel_cuda<<<dimGrid, dimBlock>>>(ga,gb,gc,n);
	error=cudaDeviceSynchronize();
	printf("%s\n", cudaGetErrorString(error));
	printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
	//timestamp(end);

	// Movemos resultado: GPU -> CPU
	timetick = dwalltime();
	cudaMemcpy(c, gc, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
	printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);
	
	// Liberamos memoria global del device utilizada
	cudaFree (ga);
cudaFree (gb);
cudaFree (gc);
//for (int i=0;i<n;i++){ printf("%f+%f=%f\n", a[i],b[i],c[i]); }

return 0;
}

int main()
{
	int n=1024;
	int size= sizeof(basetype);
	basetype *a, *b, *c;
	double timetick;
	int cb=64;

// Reservamos e inicializamos el vector en CPU
	timetick = dwalltime();
	a = (basetype *) malloc(n*size);
	b = (basetype *) malloc(n*size);
	c = (basetype *) malloc(n*size);
	init_CPU_array(a,n);
	init2_CPU_array(b,n);


printf("-> Tiempo de alocar memoria e inicializar vectores en CPU %f\n", dwalltime() - timetick);

timetick = dwalltime();
	sum_CPU_array(a,b,c,n);
	//print_CPU_array(c,n);
printf("-> Tiempo de alocar memoria e inicializar vectores en CPU %f\n", dwalltime() - timetick);
//init_CPU_array(c,n);
//print_CPU_array(c,n);

	sum_GPU_array(a,b,c,n,cb);
	
	
	return 0;
}





