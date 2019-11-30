#include <stdio.h>
#include <stdlib.h>
// #include <sys/time.h>
// #include <sys/resource.h>

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


int main(int argc, char const *argv[])
{
        /*Declaracion de las variables*/
        int n = (argc>1)?atoi(argv[1]):131072;
        int limite = 65536;
        int *secuencia;
        int *apariciones;
        DECLARACION
        double duration;
        /* Fin de declaracion de las variables */

        /* Inicializacion de los vectores */
        START
        secuencia = (int *) malloc(n*sizeof(int));
        apariciones = (int *) malloc(limite*sizeof(int));
        for (int i = 0; i < n; ++i) secuencia[i] = i & (limite-1);
        for(int i=0; i<limite; i++) apariciones[i] = 0;
        STOP
        printf("-> Tiempo de inicializacion de los vectores en CPU %f\n", duration);
        /* Fin de inicializacion de los vectores */

        /* Calculo de cantidad de cada numero */
        START
        /* Recorro el vector incrementando la posicion del numero que aparece */
        for (int i = 0; i < n; ++i) apariciones[secuencia[i]]++;
        STOP
        printf("-> Tiempo de ejecucion en CPU %f\n", duration);
        /* Fin de calculo de cantidad de cada numero */

        /* Imprimo los resultados */
        // for (int i = 0; i < 100; ++i) printf("%d tiene %d apariciones\n",i,apariciones[i]);

        free(secuencia);
        free(apariciones);
        return 0;
}
