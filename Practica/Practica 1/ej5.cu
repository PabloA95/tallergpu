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
        for(i = 0; i < n*n; i++) {
                array[i] = (basetype)3;
        }
}

void funcion_CPU(basetype vec[], const unsigned int c, const unsigned int n) {
        // Codigo
}

__global__ void multiplication_kernel_cuda(basetype *ga,basetype *gb,basetype *gc,   const int n){
        //unsigned long int col = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned long int fila = blockIdx.y * blockDim.y + threadIdx.y;        
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < n && j<n){
	gc[i*n+j]=0;
	for(int k=0; k<n; k++){
		gc[i*n+j]+=ga[i*n+k]*gb[k*n+j];
}
	}

}

void multiplicacion_GPU( basetype *ca,basetype *cb,basetype *cc, const unsigned int n, const unsigned int blk_size){
        double timetick;
        // Número de bytes de cada uno de nuestros vectores
        unsigned int numBytes = n * n * sizeof(basetype);
        cudaError_t error;

        // Reservamos memoria global del device (GPU) para el array y lo copiamos
        basetype *ga;
basetype *gb;
basetype *gc;
        timetick = dwalltime();
        cudaMalloc((void **) &ga, numBytes);
        cudaMalloc((void **) &gb, numBytes);
        cudaMalloc((void **) &gc, numBytes);
        printf("-> Tiempo de alocacion en memoria global de GPU %f\n", dwalltime() - timetick);  
        timetick = dwalltime();
        cudaMemcpy(ga, ca, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
        cudaMemcpy(gb, cb, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
        printf("-> Tiempo de copia de memoria CPU =>> GPU %f\n", dwalltime() - timetick);

        // Bloque unidimensional de hilos (*blk_size* hilos)
        dim3 dimBlock(blk_size,blk_size);
        // Grid unidimensional (*ceil(n/blk_size)* bloques)
        dim3 dimGrid((n + dimBlock.y - 1) / dimBlock.y,(n + dimBlock.x - 1) / dimBlock.x);
	printf("%d- %d\n",dimGrid.x, dimGrid.y);
        
        // Lanzamos ejecución del kernel en la GPU
        // timestamp(start);            // Medimos tiempo de cálculo en GPU
        timetick = dwalltime();
        multiplication_kernel_cuda<<<dimGrid, dimBlock>>>(ga,gb,gc, n);
        error = cudaDeviceSynchronize();
        printf("Synchronyse error: %d\n", error);
        printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
        //timestamp(end);

        // Movemos resultado: GPU -> CPU
        timetick = dwalltime();
        cudaMemcpy(cc, gc, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
        printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);

        // Liberamos memoria global del device utilizada
        cudaFree (ga);
cudaFree (gb);
cudaFree (gc);
}



int main(int argc, char **argv)
{
        double timetick;
        // Ejecucion en CPU
        unsigned int blk_size=32;
        int n=64;
	basetype *ca = (basetype *) malloc(n*n*sizeof(basetype));
	basetype *cb = (basetype *) malloc(n*n*sizeof(basetype));
	basetype *cc = (basetype *) malloc(n*n*sizeof(basetype));
	
        timetick = dwalltime();
        // Alocar e inicializar vector y variables
init_CPU_array(ca,n);
init_CPU_array(cb,n);
init_CPU_array(cc,n);
        printf("-> Tiempo de inicializacion de vector en CPU %f\n", dwalltime() - timetick);

        timetick = dwalltime();
        // Funcion que se ejecuta en la CPU
printf("%s\n", "Despues implementamos esto");
        printf("-> Tiempo de ejecucion en CPU %f\n", dwalltime() - timetick);

        // Ejecucion en GPU
        // Inicializa nuevamente el vector para realizar la ejecucion en GPU
        // Funcion que se ejecuta en la GPU
multiplicacion_GPU( ca,cb,cc,n,blk_size);
        // Chequea si el resultado obtenido en la GPU es correcto
        // check_array(vec,n);
for(int i=0; i<n*n; i++) printf("%g\n",cc[i]);
        free(ca);
free(cb);
free(cc);
        return 0;
}
