/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;


void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
double *alloc1D(int m,int n);

void myStats(const double *E, const int stride, int bm, int bn, double *_mx, double *sumSq);

extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}


// copy a submatrix from src to dst inner block
static inline void copy_submatrix(const double *src, const int src_d,
                                  double *dst, const int dst_d, const int bm, const int bn){
    for(int i = 0; i < bm; ++i){
        memcpy(dst + (dst_d + 1) + i*dst_d, src + i*src_d, sizeof(double) * bn);
    }
}


// Use MPI to run on multiple processors,
// which collectively solve E and R
//
//
void mySolve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

    // double *E = *_E, *E_prev = *_E_prev;
    // double *R_tmp = R;
    // double *E_tmp = *_E;
    // double *E_prev_tmp = *_E_prev;

    int nprocs=1, myrank=0;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    int m = cb.m, n=cb.n;
    int px = cb.px, py = cb.py;

    int ipy = myrank / px; // ipy-th row processor
    int ipx = myrank % px; // ipx-th col processor

    int bm_base = m / py, bm_rem = m % py, bm = bm_base + (ipy < bm_rem);
    int bn_base = n / px, bn_rem = n % px, bn = bn_base + (px - ipx <= bn_rem);

    int iLeftUp_glob = (ipy * bm_base + min(bm_rem, ipy) + 1) * (n + 2) + (ipx * bn_base + max(ipx - px + bn_rem, 0) + 1);
    
    /* copy E, E_prev, R blocks */
    double *E_loc       = alloc1D(bm + 2, bn + 2);
    double *E_prev_loc  = alloc1D(bm + 2, bn + 2);
    double *R_loc       = alloc1D(bm + 2, bn + 2);
    copy_submatrix(
        *_E + iLeftUp_glob,
        (n+2),
        E_loc,
        (bn+2),
        bm,
        bn
    );
    copy_submatrix(
        *_E_prev + iLeftUp_glob,
        (n+2),
        E_prev_loc,
        (bn+2),
        bm,
        bn
    );
    copy_submatrix(
        R + iLeftUp_glob,
        (n+2),
        R_loc,
        (bn+2),
        bm,
        bn
    );
    // tmp vars used as pointers for traversal
    double *R_loc_tmp = R_loc;
    double *E_loc_tmp = E_loc;
    double *E_prev_loc_tmp = E_prev_loc;
    // 4 inner corner local indexes
    int iLeftUp    = (bn + 2) + 1;
    int iRightUp   = iLeftUp + bn - 1;
    int iLeftDown  = iLeftUp + (bm - 1) * (bn + 2);
    int iRightDown = iLeftDown + bn - 1;

    // to save results
    double mx, sumSq;


    MPI_Request recvReqs[4];
    MPI_Request sendReqs[4];
    // udt column and row vector
    MPI_Datatype Col_t, Row_t, Matrix_t;
    MPI_Type_vector(
        bm,            // # of blocks
        1,             // block length
        (bn+2),        // stride
        MPI_DOUBLE,
        &Col_t
    );
    MPI_Type_vector(
        1,
        bn,
        (bn+2),
        MPI_DOUBLE,
        &Row_t
    );
    MPI_Type_commit( &Col_t );
    MPI_Type_commit( &Row_t );

    if(cb.plot_freq){
        MPI_Type_vector(
            bm,
            bn,
            (bn+2),
            MPI_DOUBLE,
            &Matrix_t
        );
        MPI_Type_commit(&Matrix_t);
    }
    
    for (int niter = 0; niter < cb.niters; niter++){

//////////////////////////////////// Exchange ghost cells using MPI ////////////////////////////////////

        int reqCnt = 0;

        // Communicate top cells
        if(ipy==0)  for (int i = iLeftUp - (bn+2); i <= iRightUp - (bn+2); i++)  E_prev_loc[i] = E_prev_loc[i + (bn+2)*2];
        else if( !cb.noComm){
            MPI_Irecv(
                E_prev_loc + iLeftUp - (bn+2),
                1,
                Row_t,
                myrank - px,
                0,
                MPI_COMM_WORLD,
                recvReqs + reqCnt
            );
            MPI_Isend(
                E_prev_loc + iLeftUp,
                1,
                Row_t,
                myrank - px,
                0,
                MPI_COMM_WORLD,
                sendReqs + reqCnt
            );
            ++reqCnt;
        }

        // Communicate left cells
        if(ipx==0)  for (int i = iLeftUp - 1; i <= iLeftDown - 1; i+=(bn+2))  E_prev_loc[i] = E_prev_loc[i + 2];
        else if( !cb.noComm){
            MPI_Irecv(
                E_prev_loc + iLeftUp - 1,
                1,
                Col_t,
                myrank - 1,
                0,
                MPI_COMM_WORLD,
                recvReqs + reqCnt
            );
            MPI_Isend(
                E_prev_loc + iLeftUp,
                1,
                Col_t,
                myrank - 1,
                0,
                MPI_COMM_WORLD,
                sendReqs + reqCnt
            );
            ++reqCnt;
        }

        // Communicate down cells
        if(ipy==py-1) for (int i = iLeftDown + (bn+2); i <= iRightDown + (bn+2); i++) E_prev_loc[i] = E_prev_loc[i - (bn+2)*2];
        else if( !cb.noComm){
            MPI_Irecv(
                E_prev_loc + iLeftDown + (bn+2),
                1,
                Row_t,
                myrank + px,
                0,
                MPI_COMM_WORLD,
                recvReqs + reqCnt
            );
            MPI_Isend(
                E_prev_loc + iLeftDown,
                1,
                Row_t,
                myrank + px,
                0,
                MPI_COMM_WORLD,
                sendReqs + reqCnt
            );
            ++reqCnt;
        }

        // Communicate right cells
        if(ipx==px-1) for (int i = iRightUp + 1; i <= iRightDown + 1; i+=(bn+2)) E_prev_loc[i] = E_prev_loc[i-2];
        else if( !cb.noComm){
            MPI_Irecv(
                E_prev_loc + iRightUp + 1,
                1,
                Col_t,
                myrank + 1,
                0,
                MPI_COMM_WORLD,
                recvReqs + reqCnt
            );
            MPI_Isend(
                E_prev_loc + iRightUp,
                1,
                Col_t,
                myrank + 1,
                0,
                MPI_COMM_WORLD,
                sendReqs + reqCnt
            );
            ++reqCnt;
        }

        // Synchronize
        if( !cb.noComm ){
            MPI_Waitall(reqCnt, sendReqs, MPI_STATUS_IGNORE);
            MPI_Waitall(reqCnt, recvReqs, MPI_STATUS_IGNORE);
        }
////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////// Compute on E and R //////////////////////////////////////////

    #define FUSED 1
    #ifdef FUSED
        // Solve for the excitation, a PDE
        for(int j = iLeftUp; j <= iLeftDown; j+=(bn+2)) {
            E_loc_tmp = E_loc + j;
            E_prev_loc_tmp = E_prev_loc + j;
            R_loc_tmp = R_loc + j;
        #ifdef AVX_ENABLE
	        #pragma ivdep
        #endif
            for(int i = 0; i < bn; i++) {
                E_loc_tmp[i] = E_prev_loc_tmp[i]+alpha*(E_prev_loc_tmp[i+1]+E_prev_loc_tmp[i-1]-4*E_prev_loc_tmp[i]+E_prev_loc_tmp[i+(bn+2)]+E_prev_loc_tmp[i-(bn+2)]);
                E_loc_tmp[i] += -dt*(kk*E_prev_loc_tmp[i]*(E_prev_loc_tmp[i]-a)*(E_prev_loc_tmp[i]-1)+E_prev_loc_tmp[i]*R_loc_tmp[i]);
                R_loc_tmp[i] += dt*(epsilon+M1* R_loc_tmp[i]/( E_prev_loc_tmp[i]+M2))*(-R_loc_tmp[i]-kk*E_prev_loc_tmp[i]*(E_prev_loc_tmp[i]-b-1));
            }
        }
    #else
        // Solve for the excitation, a PDE
        for(int j = iLeftUp; j <= iLeftDown; j+=(bn+2)) {
                E_loc_tmp = E_loc + j;
                E_prev_loc_tmp = E_prev_loc + j;
                for(int i = 0; i < bn; i++) {
                    E_loc_tmp[i] = E_prev_loc_tmp[i]+alpha*(E_prev_loc_tmp[i+1]+E_prev_loc_tmp[i-1]-4*E_prev_loc_tmp[i]+E_prev_loc_tmp[i+(bn+2)]+E_prev_loc_tmp[i-(bn+2)]);
                }
        }

        /* 
        * Solve the ODE, advancing excitation and recovery variables
        *     to the next timtestep
        */

        for(int j = iLeftUp; j <= iLeftDown; j+=(bn+2)) {
            E_loc_tmp = E_loc + j;
            R_loc_tmp = R_loc + j;
            E_prev_loc_tmp = E_prev_loc + j;
            for(int i = 0; i < bn; i++) {
                E_loc_tmp[i] += -dt*(kk*E_prev_loc_tmp[i]*(E_prev_loc_tmp[i]-a)*(E_prev_loc_tmp[i]-1)+E_prev_loc_tmp[i]*R_loc_tmp[i]);
                R_loc_tmp[i] += dt*(epsilon+M1* R_loc_tmp[i]/( E_prev_loc_tmp[i]+M2))*(-R_loc_tmp[i]-kk*E_prev_loc_tmp[i]*(E_prev_loc_tmp[i]-b-1));
            }
        }
    #endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////
   
        if(cb.plot_freq){
            if (!(niter % cb.plot_freq)){
                // process 0 collect data from all other processes and update global E
                MPI_Request sendEReqs;
                MPI_Isend(
                    E_loc + iLeftUp, 
                    1,
                    Matrix_t,
                    0,
                    0,
                    MPI_COMM_WORLD,
                    &sendEReqs
                );
                MPI_Request_free(&sendEReqs);
                if(ipx == 0){
                    // send the left ghost cell
                    MPI_Isend(
                        E_loc + iLeftUp - 1,
                        1,
                        Col_t,
                        0,
                        1,
                        MPI_COMM_WORLD,
                        &sendEReqs
                    );
                    MPI_Request_free(&sendEReqs);
                }
                if(ipx == px - 1){
                    // send the right ghost cell
                    MPI_Isend(
                        E_loc + iRightUp + 1,
                        1,
                        Col_t,
                        0,
                        2,
                        MPI_COMM_WORLD,
                        &sendEReqs
                    );
                    MPI_Request_free(&sendEReqs);
                }
                if(ipy == 0){
                    // send the top ghost cell
                    MPI_Isend(
                        E_loc + iLeftUp - (bn+2),
                        1,
                        Col_t,
                        0,
                        3,
                        MPI_COMM_WORLD,
                        &sendEReqs
                    );
                    MPI_Request_free(&sendEReqs);
                }
                if(ipy == py - 1){
                    // send the down ghost cell
                    MPI_Isend(
                        E_loc + iLeftDown + (bn+2),
                        1,
                        Row_t,
                        0,
                        4,
                        MPI_COMM_WORLD,
                        &sendEReqs
                    );
                    MPI_Request_free(&sendEReqs);
                }

                if(!myrank){
                    MPI_Request recvEInnerReqs[nprocs];
                    MPI_Request recvEGhostCol[py*2];
                    MPI_Request recvEGhostRow[px*2];
                    for(int i = 0; i < nprocs; i++){
                        int ipy_recv = i / px; 
                        int ipx_recv = i % px;
                        int bm_recv = bm_base + (ipy_recv < bm_rem);
                        int bn_recv = bn_base + (px - ipx_recv <= bn_rem);
                        int iLeftUp_glob_recv = (ipy_recv * bm_base + min(bm_rem, ipy_recv) + 1) * (n + 2) + (ipx_recv * bn_base + max(ipx_recv - px + bn_rem, 0) + 1);
                        MPI_Datatype Matrix_recv_t, Col_recv_t, Row_recv_t;
                        MPI_Type_vector(
                            bm_recv,
                            bn_recv,
                            (n+2),
                            MPI_DOUBLE,
                            &Matrix_recv_t
                        );
                        MPI_Type_vector(
                            bm_recv,
                            1,
                            (bn_recv+2),
                            MPI_DOUBLE,
                            &Col_recv_t
                        );
                        MPI_Type_vector(
                            1,
                            bn,
                            (bn_recv+2),
                            MPI_DOUBLE,
                            &Row_recv_t
                        );

                        MPI_Type_commit(&Matrix_recv_t);
                        MPI_Type_commit(&Col_recv_t);
                        MPI_Type_commit(&Row_recv_t);

                        MPI_Irecv(
                            *_E + iLeftUp_glob_recv,
                            1,
                            Matrix_recv_t,
                            i,
                            0,
                            MPI_COMM_WORLD,
                            recvEInnerReqs + i
                        ); // receive inner matrix

                        if(ipx_recv == 0){
                            // receive the left ghost cell
                            MPI_Irecv(
                                *_E + iLeftUp_glob_recv -1,
                                1,
                                Col_recv_t,
                                i,
                                1,
                                MPI_COMM_WORLD,
                                recvEGhostCol + ipy_recv
                            );
                        }

                        if(ipx_recv == px - 1){
                            // receive the right ghost cell
                            MPI_Irecv(
                                *_E + iLeftUp_glob_recv + 1,
                                1,
                                Col_recv_t,
                                i,
                                2,
                                MPI_COMM_WORLD,
                                recvEGhostCol + ipy_recv + py
                            );

                        }
                        if(ipy_recv == 0){
                            // receive the top ghost cell
                            MPI_Irecv(
                                *_E + iLeftUp_glob_recv - (n + 2),
                                1,
                                Row_recv_t,
                                i,
                                3,
                                MPI_COMM_WORLD,
                                recvEGhostRow + ipx_recv
                            );
                        }
                        if(ipy_recv == py - 1){
                            MPI_Irecv(
                                *_E + iLeftUp_glob_recv + (n + 2),
                                1,
                                Row_recv_t,
                                i,
                                4,
                                MPI_COMM_WORLD,
                                recvEGhostRow + ipx_recv + px
                            );
                        }
                    }
                    MPI_Waitall(nprocs, recvEInnerReqs, MPI_STATUS_IGNORE);
                    MPI_Waitall(py*2, recvEGhostCol, MPI_STATUS_IGNORE);
                    MPI_Waitall(px*2, recvEGhostRow, MPI_STATUS_IGNORE);
                    plotter->updatePlot(*_E, niter, m, n);
                }

                
            }
        }

        // Swap current and previous meshes
        double *tmp = E_loc; E_loc = E_prev_loc; E_prev_loc = tmp;
    }



    /*
     * Reduce L2 and Linf
     * Gather results on process 0
     * 
     */
    
    MPI_Barrier(MPI_COMM_WORLD);
    stats(E_prev_loc, bm, bn, &Linf, &sumSq);
    
    if(! cb.noComm){
        // shall only be relevant to root process
        double globalLinf;
        double globalSumSq;
        
        MPI_Reduce(
            &sumSq,         // send data
            &globalSumSq,   // recv data
            1,              // data count
            MPI_DOUBLE,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );
        MPI_Reduce(
            &Linf,
            &globalLinf,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        if( !myrank){ // process 0
            L2 = L2Norm(globalSumSq);
            Linf = globalLinf;
        }
    }
    else{ // Communication shut off
        L2 = L2Norm(sumSq);
    }
}


// Entry point of solution
void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){
    mySolve(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
}


void printMat2(const char mesg[], double *E, int m, int n){
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
