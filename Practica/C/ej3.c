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

void transponer_matriz(basetype *m, basetype *t, int b, int h)
{
	int n=h*b;
	for (int i = 0; i < b; ++i){
		for (int j = 0; j < h; ++j){
			t[i+j*n] = m[i*n+j];
		}
	}

}

int main(int argc, char const *argv[])
{
	basetype *m;
	basetype *t;

	int b=5; //=1024;
	int h=3; //=2048;
	int n=b*h;

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
	transponer_matriz(m,t,b,h);
	printf("-> Tiempo de suma de vectores en CPU %f\n", dwalltime() - timetick);

	for (int i = 0; i < n; ++i)
	{
		printf("%d\n", m[i]);
	}
	printf("%p\n", t);
	for (int i = 0; i < n; ++i)
	{
		printf("%d\n", t[i]);
	}


	return 0;
}