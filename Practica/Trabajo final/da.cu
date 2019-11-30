#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

double dwalltime(){
        double sec;
        struct timeval tv;
        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


__global__ void contar_kernel_cuda(int * input, int * output, const int n, int size_array, int thr_x_blk, int limite){

        //unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;
        //REVISAR QUE CUENTAS MERECEN TENER UNA VARIABLE Y CUALES SE HACEN POCAS VECES COMO PARA JUSTIFICAR LA MEMORIA

	/*if (global_id == 712) {
		printf("%d--------%d---------%d ----- /// %d ******%d\n",blockIdx.x,blockDim.x ,threadIdx.x,(blockIdx.x*(limite))+(input[blockIdx.x*size_array+5]),(((input[blockIdx.x*size_array+size_array-1]) & (thr_x_blk-1))));
//id =1   ///0--------256---------1
//id =223 ///0--------256---------223
//id =712 ///2--------256---------200
	}*/

        if (global_id < n){
                /*Se recorre todo el segmento del arreglo asignado al bloque*/
                for(int i=0;i<size_array;i++){
                        /*Toma el valor de la entrada en la posicion del inicio del segmento actual(blockId.x*size_array) mas el desplazamiento i , calcula el modulo con respecto a la cantidad de hilos, y si el modulo es igual a su id entonces le corresponde actualizarlo
                        */ 
                        if(((input[blockIdx.x*size_array+i]) & (thr_x_blk-1)) == threadIdx.x){
//printf("%daaaaaaaaaa%d\n",global_id,output[(blockIdx.x*(limite))+(input[blockIdx.x*size_array+i])]);
                                output[(blockIdx.x*(limite))+(input[blockIdx.x*size_array+i])]++;
                        }
                }
        }
}


__global__ void reducir_kernel_cuda(int * gapariciones,int limite, int i){
        unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_id < limite){

                gapariciones[global_id] += gapariciones[(i*limite)+global_id];



        }
}

int main(int argc, char const *argv[])
{



        /*
Se 'divide' el arreglo de n numeros en tantos pedasos como bloques se utilices.
Se genera un arreglo de m lugares, siendo m el producto entre 65536 por la cantidad de bloques usados, para 
 que cada bloque use una seccion para dejar los resultados del sector analizado.
En un primer kernel, a cada hilo se le asignan una cantidad de numeros (de los posibles para la secuencia de entrada)
 segun la cantidad de hilos por bloque, de forma que los 65536 numeros esten divididos equitativamente entre los 
 k hilos del bloque, ...de forma que todos los hilos de un bloque recorren todo los valores de la seccion de las n valores
 asignados a ese bloque y cuando encuentran un valor que le pertenece, incrementa el contador de la seccion del arreglo de
 m posiciones correspondiente a ese numero.
 AL finalizar quedaran un arreglo que sera sera visto como multiples arreglos de 65536 lugares, que tendran de forma
 parcial la cantidad de apariciones de cada numero.
En un segundo kernel, que se llamara de forma iterativa, se deben ir sumando los arreglos generados en las distintas
 secciones del arreglo de m posiciones, para en sqrt(nro_bloques) pasos finalizar con un unico vector que posea la cantidad
 total para cada numero. Para esto, en el segundo kernes, los hilos generados deberan ir sumando todos los valores a una
 distancia de 65536 lugares.


        */
        /* code */
        /*Cada hilo puede escribir una variable local y despues volcarla a la global en la PGU*/
        int n;
        if (argc>1) {
                n = atoi(argv[1]);
        } else {
                n = 131072;
        }

        int thr_x_blk;
        if (argc>2) {
                thr_x_blk = atoi(argv[2]);
        } else {
                thr_x_blk = 256;
        }

        double timetick;
        int limite = 65536;//65536;
        int *secuencia;
        int *apariciones;

        ////////////////////////int *nvector;

        cudaError_t error;
        int *gsecuencia;
        int *gapariciones;
        int cant_x_hilo=16; ///QUE HACIA ESTO?  

        // Bloque unidimensional de hilos (*blk_size* hilos)
        dim3 dimBlock(thr_x_blk);
        // Grid unidimensional (*ceil(n/blk_size)* bloques)
        dim3 dimGrid((((n + dimBlock.x - 1) / (dimBlock.x*cant_x_hilo))));
        printf("dimGrid %d\n", dimGrid.x);

        int size_buffers=n/dimGrid.x;
        //printf("%d\n", size_buffers);


        /* Inicializo los vectores en CPU */
        timetick = dwalltime();
        secuencia = (int *) malloc(n*sizeof(int));
        apariciones = (int *) malloc(limite*sizeof(int)*(dimGrid.x));
        for (int i = 0; i < n; ++i)
        {
                secuencia[i] = i & (limite-1);        
        }

        for(int i=0; i<limite; i++) apariciones[i] = 0;
        printf("-> Tiempo de inicializacion de los vectores en CPU %f\n", dwalltime() - timetick);

        /* Inicializo en la GPU */
        cudaMalloc((void **) &gsecuencia, n*sizeof(int));
        /* QUITAR COPIA DE DATOS DE APARICIONES Y QUE LOS HILOS INICIALICEN CON 0 EL VECTOR*/
        cudaMalloc((void **) &gapariciones, n*sizeof(int)*dimGrid.x);
        cudaMemcpy(gsecuencia, secuencia, n*sizeof(int), cudaMemcpyHostToDevice); // CPU -> GPU
        cudaMemcpy(gapariciones, apariciones, (limite*sizeof(int)*dimGrid.x), cudaMemcpyHostToDevice); // CPU -> GPU
timetick = dwalltime();
        /* GAPARICIONES tiene que ser de tamano n * dimGrid */
        contar_kernel_cuda<<<dimGrid, dimBlock>>>(gsecuencia, gapariciones, n, size_buffers,thr_x_blk,limite);
        error = cudaDeviceSynchronize();
        printf("Synchronyse error: %d\n", error);
	
//reducir_kernel_cuda(int * input, int * output, const int n, int pos_i, int pos_f)

	int aux = dimGrid.x;
	// Bloque unidimensional de hilos (*blk_size* hilos)
        dimBlock.x=thr_x_blk;
        // Grid unidimensional (*ceil(n/blk_size)* bloques)
printf("Aux %d\n", aux);
//REVISAR REDONDEOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
	int t = limite / thr_x_blk; // no le gusto la cuenta        
	printf("%d,,%d,,%d,, \n",t, limite, thr_x_blk);	
	dimGrid.x = t;
	for(int i=1; i < aux; i++){
		reducir_kernel_cuda<<<dimGrid, dimBlock>>>(gapariciones,limite,i);
		error = cudaDeviceSynchronize();
       		//printf("Synchronyse error: %d\n", error);		
	}
printf("-> Tiempo de ejecucion en CPU %f\n", dwalltime() - timetick);
        cudaMemcpy(apariciones, gapariciones, limite*sizeof(int), cudaMemcpyDeviceToHost); // GPU -> CPU
	//COPIAR SOLO EL COMIENZO DEL ARREGLO UNA VEZ QUE LO REDUCIMOS!!!!!!!!! :D
//cudaMemcpy(apariciones, gapariciones, (limite*sizeof(int)*dimGrid.x), cudaMemcpyDeviceToHost); // GPU -> CPU

/*        for(int i =0;i<limite;i++){
	printf("pos %d: %d\n",i, apariciones[i]);
}*/
        cudaFree (gapariciones);
        cudaFree (gsecuencia);

        free(secuencia);
        free(apariciones);
        return 0;
}
