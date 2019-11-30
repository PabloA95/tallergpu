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
        for(i = 0; i < n; i++) {
                array[i] = (basetype) i;
        }
}

void funcion_CPU(basetype vec[], const unsigned int c, const unsigned int n) {
        // Codigo
}

__global__ void suma_vesctor_kernel_cuda(basetype *vec, const int n, int i){
        unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned long int fila = blockIdx.y * blockDim.y + threadIdx.y;
	if (global_id<i){
		vec[global_id]+=vec[global_id+i];
	}
}

void suma_vector_GPU( basetype *vec,basetype *res, const unsigned int n, const unsigned int blk_size, basetype *gvec){
        double timetick;
        // Número de bytes de cada uno de nuestros vectores
        unsigned int numBytes = n * sizeof(basetype);
        cudaError_t error;

        // Reservamos memoria global del device (GPU) para el array y lo copiamos
        
//	basetype *gres;

        timetick = dwalltime();

//        cudaMalloc((void **) &gres, numBytes);

	printf("ACA");

        printf("-> Tiempo de alocacion en memoria global de GPU %f\n", dwalltime() - timetick);  
        timetick = dwalltime();
        cudaMemcpy(gvec, vec, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
        
printf("-> Tiempo de copia de memoria CPU =>> GPU %f\n", dwalltime() - timetick);

        // Bloque unidimensional de hilos (*blk_size* hilos)
        dim3 dimBlock(blk_size);
        // Grid unidimensional (*ceil(n/blk_size)* bloques)
        dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);
	

        // Lanzamos ejecución del kernel en la GPU
        // timestamp(start);            // Medimos tiempo de cálculo en GPU
        timetick = dwalltime();
for(int i=n/2;i>0;i=i/2){
printf("%d",i);

        suma_vesctor_kernel_cuda<<<dimGrid, dimBlock>>>(gvec,n,i);
        error = cudaDeviceSynchronize();
        printf("Synchronyse error: %d\n", error);
} 
       printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
        //timestamp(end);

        // Movemos resultado: GPU -> CPU
        timetick = dwalltime();
        cudaMemcpy(res, gvec, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
        printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);
printf("%g\n",res[0]);
        // Liberamos memoria global del device utilizada
        //cudaFree (gres);
	//cudaFree (gvec);


}

basetype suma_vector_CPU(basetype *vec, int n){
	int cont=0;
	for(int i=0; i<n ; i++){
		cont+=vec[i];
//printf("%d\n",cont);

	}
	return cont;
}

int main(int argc, char **argv)
{
        double timetick;
        // Ejecucion en CPU
        unsigned int blk_size=32;
        int n=1024;
	basetype *vec = (basetype *) malloc(n*sizeof(basetype));
	basetype *res = (basetype *) malloc(n*sizeof(basetype));

	basetype *gvec;
        cudaMalloc((void **) &gvec, n*sizeof(basetype));
	
        timetick = dwalltime();
        // Alocar e inicializar vector y variables
init_CPU_array(vec,n);

        printf("-> Tiempo de inicializacion de vector en CPU %f\n", dwalltime() - timetick);
        timetick = dwalltime();
        // Funcion que se ejecuta en la CPU
int a=suma_vector_CPU(vec,n);
printf("%d\n",a);

       printf("-> Tiempo de ejecucion en CPU %f\n", dwalltime() - timetick);

        // Ejecucion en GPU
        // Inicializa nuevamente el vector para realizar la ejecucion en GPU
        // Funcion que se ejecuta en la GPU
suma_vector_GPU(vec, res, n, blk_size, gvec);
        // Chequea si el resultado obtenido en la GPU es correcto
        // check_array(vec,n);
/*for(int i=0; i<n*n; i++) printf("%g\n",cc[i]);
        free(ca);
free(cb);
free(cc);
*/        
return 0;
}
