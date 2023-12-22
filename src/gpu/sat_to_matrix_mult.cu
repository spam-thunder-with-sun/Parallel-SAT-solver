using namespace std;

//Librerie standard
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <bitset>
#include <stdio.h>
//Librerie per il calcolo parallelo
#include <cuda_runtime.h>
#include "cublas_v2.h"
//Librerie fatte
#include "read_data.h"
#include "constant.h"

#define M 6
#define N 5

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

int cublas_test()
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* devPtrA;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * N + j + 1);
        }
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }
    free(a);

    return EXIT_SUCCESS;
}

int cublas_allocation(INT_TYPE literals, INT_TYPE clauses, DATA_TYPE* matrix)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    DATA_TYPE* devPtr;

    if (!matrix) {
        cerr << "Host memory allocation failed" << endl;
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&devPtr, literals*clauses*sizeof(*matrix)<<1);
    if (cudaStat != cudaSuccess) {
        cerr << "Device memory allocation failed: " << cudaStat << endl;
        return EXIT_FAILURE;
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "CUBLAS initialization failed: " << stat << endl;
        return EXIT_FAILURE;
    }

    stat = cublasSetMatrix (literals<<1, clauses, sizeof(*matrix), matrix, literals<<1, devPtr, literals<<1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data download failed: " << stat << endl;
        cudaFree (devPtr);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    /*
    modify (handle, devPtr, literals<<1, N, 1, 2, 16.0f, 12.0f);
    
    */

    stat = cublasGetMatrix (literals<<1, clauses, sizeof(*matrix), devPtr, literals<<1, matrix, literals<<1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data upload failed: " << stat << endl;
        cudaFree (devPtr);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaFree(devPtr);
    cublasDestroy(handle);

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) 
{
    string filename = "../input/dimacs/small.cnf";

    if(argc > 1)
        filename = argv[1];

    INT_TYPE literals, clauses;
    DATA_TYPE* matrix;
    std::tie(literals, clauses, matrix) = readDimacsFile2Column(filename);
    print_matrix(literals, clauses, matrix);

    if(cublas_allocation(literals, clauses, matrix))
        cout << "Cublas test failed" << endl;
    else
        cout << "Cublas test passed" << endl;

    cout << endl;
    cout << "After cublas" << endl;
    print_matrix(literals, clauses, matrix);

    free(matrix);

    return 0;
}
