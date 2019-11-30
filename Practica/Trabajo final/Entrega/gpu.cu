#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef WIN
#include <time.h>
#define DECLARACION clock_t start,finish;
#define START start = clock();
#define STOP duration=((double)(clock()-start)/CLOCKS_PER_SEC);
#else
#include <sys/time.h>
#include <sys/resource.h>
#define DECLARACION double timetick;
#define START timetick = dwalltime();
#define STOP duration = dwalltime() - timetick;
double dwalltime(){
        double sec;
        struct timeval tv;
        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}
#endif




__global__ void contar_kernel_cuda(int * input, int * output, const int n, int size_array, int thr_x_blk, int limite,int cant_x_hilo){

        //unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
        extern __shared__ int datos_shared[];
        if (threadIdx.x < thr_x_blk){
                int comienzo_arreglo=blockIdx.x*size_array;
                int comienzo_en_output=blockIdx.x*(limite);
                int module=thr_x_blk-1;
                for(int i=0; i<cant_x_hilo;i++) if((comienzo_arreglo+i*thr_x_blk+threadIdx.x)<n) datos_shared[threadIdx.x+i*thr_x_blk]=input[comienzo_arreglo+i*thr_x_blk+threadIdx.x];
                __syncthreads();
                /*Se recorre todo el segmento del arreglo asignado al bloque*/
                for(int i=0;i<size_array;i++){
                        /*Toma el valor de la entrada en la posicion del inicio del segmento actual(blockId.x*size_array) mas el desplazamiento i , calcula el modulo con respecto a la cantidad de hilos, y si el modulo es igual a su id entonces le corresponde actualizarlo*/ 
                        if(((datos_shared[i]) & (module)) == threadIdx.x && ((comienzo_arreglo+i)<n)) output[comienzo_en_output+(datos_shared[i])]++;
                }
        }
}


__global__ void reducir_kernel_cuda(int *gapariciones,int limite, int i){
        unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_id < limite) gapariciones[global_id] += gapariciones[(i*limite)+global_id];
}


int main(int argc, char const *argv[])
{

        /*Declaracion de las variables*/
        int n = (argc>1)?atoi(argv[1]):131072;
        int thr_x_blk = (argc>2)?atoi(argv[2]):128;
        unsigned int cant_x_hilo=(n<131072)?4:64; // Cantidad de posiciones que revisa cada hilo 

        DECLARACION
        double duration;

        int limite = 65536;
        int *secuencia;
        int *apariciones;

        cudaError_t error;
        int *gsecuencia;
        int *gapariciones;
        /* Fin de declaracion de las variables */


        /* Creacion de los hilos y los bloques a usar en el primer kernel */
        // Bloque unidimensional de hilos (*blk_size* hilos)
        dim3 dimBlock(thr_x_blk);
        // Grid unidimensional (*ceil(n/blk_size)* bloques)
        dim3 dimGrid((n + (dimBlock.x*cant_x_hilo) - 1) / (dimBlock.x*cant_x_hilo));
        //printf("dimGrid %d -- thread %d\n", dimGrid.x, dimBlock.x);
        /* Fin de creacion de los hilos y los bloques a usar en el primer kernel */

        int size_buffers= (n + dimGrid.x -1) / dimGrid.x ;

        /* Inicializacion de los vectores en CPU */
        START
        secuencia = (int *) malloc(n*sizeof(int));
        apariciones = (int *) malloc(limite*sizeof(int)*(dimGrid.x));
        //INICIALIZAMOS LOS VECTORES EN CPU
        for (int i = 0; i < n; ++i) secuencia[i] = i & (limite-1);   
        STOP     
        printf("-> Tiempo de inicializacion de los vectores en CPU %f\n", duration);
        /* Fin de inicializacion de los vectores en CPU */

        /* Alocacion de memoria en el device */
        START
        cudaMalloc((void **) &gsecuencia, n*sizeof(int));
        cudaMalloc((void **) &gapariciones, limite*sizeof(int)*dimGrid.x);
        STOP
        printf("-> Tiempo de inicializacion y copia de los vectores en GPU %f\n", duration);
        /* Fin de alocacion de memoria en el device */


        /* Memset y copia de datos de host a device */
//ESTE VA
        START
        cudaMemcpy(gsecuencia, secuencia, n*sizeof(int), cudaMemcpyHostToDevice); // CPU -> GPU
        cudaMemset(gapariciones,0,(limite*sizeof(int)*dimGrid.x));        

        STOP
        printf("-> Tiempo de memset de los vectores en GPU %f\n", duration);
        /* Fin de memset y copia de datos de host a device */


        /* Ejecucion del primer kernel para hacer las sumas parciales*/
        START
        contar_kernel_cuda<<<dimGrid, dimBlock, n*sizeof(int)/dimGrid.x>>>(gsecuencia, gapariciones, n, size_buffers,thr_x_blk,limite,cant_x_hilo);
        error = cudaDeviceSynchronize();
        printf("%s\n", cudaGetErrorString(error));
        //for (int i=0;i<65536;i++) printf("%d",apariciones[i]);
        /* Fin de ejecucion del primer kernel para hacer las sumas parciales*/


        /* Creacion de los hilos y los bloques a usar en el segundo kernel */
        int aux = dimGrid.x;
        // Bloque unidimensional de hilos (*blk_size* hilos)
        dimBlock.x=thr_x_blk;
        // Grid unidimensional (*ceil(n/blk_size)* bloques)    
        dimGrid.x = (limite + dimBlock.x - 1)/dimBlock.x; //dimBlock.x
        /* Fin de creacion de los hilos y los bloques a usar en el primer kernel */

        /* Ejecucion del kernel para hacer las reducciones */
        for(int i=1; i < aux; i++){
                reducir_kernel_cuda<<<dimGrid, dimBlock>>>(gapariciones,limite,i);
                error = cudaDeviceSynchronize();
                //if(error) printf("Synchronyse error: %d -- %s\n", error, cudaGetErrorString(error));        
        }
        STOP
        printf("-> Tiempo de ejecucion en GPU %f\n", duration);
        /* Fin de ejecucion del kernel para hacer las reducciones */

        
        /* Copia de datos del device al host*/
        START
        cudaMemcpy(apariciones, gapariciones, limite*sizeof(int), cudaMemcpyDeviceToHost); // GPU -> CPU
        STOP
        printf("-> Tiempo de copia del arreglo de apariciones de GPU a CPU %f\n", duration);
        /* Copia de datos del device al host*/


        /*
        printf("Primeros 16 valores-----------\n");
        for (int i=0;i<16;i++) printf("%d\n",apariciones[i]);
        printf("Ultimos 16 valores-----------\n");
        for (int i=65520;i<65536;i++) printf("%d\n",apariciones[i]);
        */

        /* Liberar la memoria de la CPU y de la GPU */
        START
        cudaFree (gapariciones);
        cudaFree (gsecuencia);
        free(secuencia);
        free(apariciones);
        STOP
        printf("-> Tiempo de liberacion de las memorias %f\n", duration);
        /* Fin de liberar la memoria de la CPU y de la GPU */

        return 0;
}
