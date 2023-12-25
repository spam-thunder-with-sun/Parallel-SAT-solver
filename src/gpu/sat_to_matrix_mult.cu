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
#include "print_data.h"
#include "constant.h"

DATA_TYPE* createSolutionMatrix(INT_TYPE literals)
{
    //Alloco la matrice di soluzione
    DATA_TYPE *solution_matrix = (DATA_TYPE*)calloc(1<<(literals+1)*literals, sizeof(*solution_matrix));
    if (!solution_matrix) {
        cerr << "Host memory allocation failed" << endl;
        return NULL;
    }

    //Riempio la matrice di soluzione
    for(INT_TYPE i = 0; i < 1<<literals; i++)
    {
        for(INT_TYPE j = 0; j < literals; j++)
        {
            solution_matrix[IDX2C(i, literals-1-j, 1<<literals)] = i>>j & 1;
            solution_matrix[IDX2C(i, (literals<<1)-1-j, 1<<literals)] = !(i>>j & 1);
        }
    }

    return solution_matrix;
}

int cublas(INT_TYPE literals, INT_TYPE clauses, DATA_TYPE* problem_matrix, DATA_TYPE* solution_matrix)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    DATA_TYPE *devPtr_problem, *devPtr_solution;

    //Controllo se le matrici sono state allocate
    if (!problem_matrix || !solution_matrix) {
        cerr << "Host memory allocation failed" << endl;
        return EXIT_FAILURE;
    }

    //--------------------------------------

    //Allocazione memoria device
    cudaStat = cudaMalloc ((void**)&devPtr_problem, literals*clauses*sizeof(*problem_matrix)<<1);
    if (cudaStat != cudaSuccess) {
        cerr << "Device memory allocation (problem) failed: " << cudaStat << endl;
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&devPtr_solution, 1<<(literals+1)*literals*sizeof(*solution_matrix));
    if (cudaStat != cudaSuccess) {
        cerr << "Device memory allocation (solution) failed: " << cudaStat << endl;
        return EXIT_FAILURE;
    }

    //Creazione handle
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "CUBLAS initialization failed: " << stat << endl;
        return EXIT_FAILURE;
    }

    //Copia dei dati sul device
    stat = cublasSetMatrix (literals<<1, clauses, sizeof(*problem_matrix), problem_matrix, literals<<1, devPtr_problem, literals<<1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data download failed (problem): " << stat << endl;
        cudaFree (devPtr_problem);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    stat = cublasSetMatrix (1<<literals, literals<<1, sizeof(*solution_matrix), solution_matrix, 1<<literals, devPtr_solution, 1<<literals);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data download failed (solution): " << stat << endl;
        cudaFree (devPtr_solution);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //Definizione dei parametri per la moltiplicazione
    DATA_TYPE alpha = 1;
    DATA_TYPE beta = 0;

    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1<<literals, literals<<1, clauses, 
        &alpha, devPtr_problem, CUDA_R_32F, literals<<1, devPtr_solution, CUDA_R_32F, 1<<literals, &beta, devPtr_solution, CUDA_R_32F, 1<<literals, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const void     *alpha,
                           const void     *A,
                           cudaDataType   Atype,
                           int lda,
                           const void     *B,
                           cudaDataType   Btype,
                           int ldb,
                           const void     *beta,
                           void           *C,
                           cudaDataType   Ctype,
                           int ldc,
                           cudaDataType   computeType,
                           cublasGemmAlgo_t algo)


    stat = cublasGetMatrix (literals<<1, clauses, sizeof(*problem_matrix), devPtr_problem, literals<<1, problem_matrix, literals<<1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data upload failed (problem): " << stat << endl;
        cudaFree (devPtr_problem);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaFree(devPtr_problem);
    cudaFree(devPtr_solution);
    cublasDestroy(handle);

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) 
{
    cout << endl << "START" << endl << "-------------------------------------------------------------------" << endl << endl;

    string filename = "../input/dimacs/small.cnf";
    if(argc > 1)
        filename = argv[1];

    INT_TYPE literals, clauses;
    DATA_TYPE *problem_matrix, *solution_matrix;
    //Leggo i dati dal file di input
    std::tie(literals, clauses, problem_matrix) = readDimacsFile2Column(filename);
    //Alloco la matrice di soluzione
    solution_matrix = createSolutionMatrix(literals);

    //Stampo la matrice in input
    printInputMatrix(literals, clauses, problem_matrix);
    //Stampo la matrice di soluzione
    printSolutionMatrix(literals, solution_matrix);

    cout << endl << endl;
    if(cublas(literals, clauses, problem_matrix, solution_matrix))
        cout << "Test failed" << endl;
    else
        cout << "Test passed" << endl;

    
    free(problem_matrix);
    free(solution_matrix);

    cout << endl << "-------------------------------------------------------------------" << endl << "END" << endl;

    return 0;
}
