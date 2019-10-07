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

void sumar_arreglos_CPU(basetype *a, basetype *b, basetype *c,int n){
	for (int i = 0; i < n; ++i)
	{
		c[i]=a[i]+b[i];
	}

}

int main(int argc, char const *argv[])
{
	/* code */

	basetype *a;
	basetype *b;
	basetype *c;
	int n=1024*1024;

	double timetick;
	
	// Aloco memoria para los vectores
	timetick = dwalltime();
	a=(basetype *) malloc(n*sizeof(basetype));
	b=(basetype *) malloc(n*sizeof(basetype));
	c=(basetype *) malloc(n*sizeof(basetype));
	printf("-> Tiempo de allocacion de vectores en CPU %f\n", dwalltime() - timetick);

	// Inicializo los arreglos
	timetick = dwalltime();
	init_CPU_array(a,n);
	init_CPU_array(b,n);
	printf("-> Tiempo de inicializacion de vectores en CPU %f\n", dwalltime() - timetick);

	// Sumo los arreglos
	timetick = dwalltime();
	sumar_arreglos_CPU(a,b,c,n);
	printf("-> Tiempo de suma de vectores en CPU %f\n", dwalltime() - timetick);

	// for (int i = 0; i < 15; ++i)
	// {
	// 	printf("%d\n", c[i]);
	// }


	return 0;
}