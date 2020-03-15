/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include <math.h>

#ifdef _MPI_
#include <mpi.h>
#endif

using namespace std;

void printMat(const char mesg[], double *E, int m, int n);




// Initialize E_prev and R
// right half-plane of E_prev to 1.0, the left half plane to 0
// botthom half-plane of R to 1.0, the top half plane to 0
void allInit(double *E,double *E_prev,double *R,int m,int n){
    int i;

    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
	    continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
	    continue;

        R[i] = 1.0;
    }

#if 0
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
#endif
}


// Distribute initial conditions from process 0 to all other processes
void myInit(double *E,double *E_prev,double *R,int m,int n){
    int nprocs=1, myrank=0;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    int dcount = (m+2)*(n+2);

    if(!myrank)
        allInit(E, E_prev, R, m, n);
    
    MPI_Bcast(
        E_prev,
        dcount,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );


    MPI_Bcast(
        R,
        dcount,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

// check if initialized
#if 0
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
#endif
}


//
// Initialization
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
#ifdef _MPI_
    myInit(E, E_prev, R, m, n);
#else
    allInit(E, E_prev, R, m, n);
#endif
}


double *alloc1D(int m,int n){
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}


// We only print the meshes if they are small enough
void printMat(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}

// compute stats on (bm,bn) region starting from pointer E
void myStats(const double *E, const int stride, int bm, int bn, double *_mx, double *sumSq){
    double mx = -1;
    double _sumSq = 0;
    for(int i = 0; i < bm; ++i){
        for(int j = 0; j < bn; ++j){
            double num = fabs(E[i*stride + j]);
            _sumSq += num*num;
            if (num > mx) mx = num;
        }
    }
    *_mx = mx;
    *sumSq = _sumSq;
}