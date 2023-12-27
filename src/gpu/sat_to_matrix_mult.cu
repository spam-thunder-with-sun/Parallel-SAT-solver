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

int cublas(INT_TYPE literals, INT_TYPE clauses, DATA_TYPE* solution_matrix, DATA_TYPE* problem_matrix, RESULT_TYPE* result_matrix)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    DATA_TYPE *devPtr_problem, *devPtr_solution;
    RESULT_TYPE *devPtr_result;
    //Definizione dei parametri per la moltiplicazione
    const RESULT_TYPE alpha = (RESULT_TYPE)1;
    const RESULT_TYPE beta = (RESULT_TYPE)0;

    //Controllo se le matrici sono state allocate
    if (!problem_matrix || !solution_matrix || !result_matrix) {
        cerr << "Host memory allocation failed" << endl;
        return EXIT_FAILURE;
    }

    //--------------------------------------
    //Creazione handle
    //--------------------------------------
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "CUBLAS initialization failed: " << stat << endl;
        return EXIT_FAILURE;
    }

    //--------------------------------------
    //Allocazione memoria device
    //--------------------------------------
    //Matrice soluzione (1<<literals x (literals<<1))
    cudaStat = cudaMalloc ((void**)&devPtr_solution, (1<<literals)*(literals<<1)*sizeof(*solution_matrix));
    if (cudaStat != cudaSuccess) {
        cerr << "Device memory allocation (solution) failed: " << cudaStat << endl;
        return EXIT_FAILURE;
    }

    //Matrice problema ((literals<<1) x clauses)
    cudaStat = cudaMalloc ((void**)&devPtr_problem, (literals<<1)*clauses*sizeof(*problem_matrix));
    if (cudaStat != cudaSuccess) {
        cerr << "Device memory allocation (problem) failed: " << cudaStat << endl;
        return EXIT_FAILURE;
    }

    //Matrice risultato (1<<literals x clauses)
    cudaStat = cudaMalloc ((void**)&devPtr_result, (1<<literals)*clauses*sizeof(*result_matrix));
    if (cudaStat != cudaSuccess) {
        cerr << "Device memory allocation (result) failed: " << cudaStat << endl;
        return EXIT_FAILURE;
    }

    //--------------------------------------
    //Copia dei dati sul device
    //--------------------------------------
    //Matrice soluzione (1<<literals x (literals<<1))
    stat = cublasSetMatrix (1<<literals, literals<<1, sizeof(*solution_matrix), solution_matrix, 1<<literals, devPtr_solution, 1<<literals);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data download failed (solution): " << stat << endl;
        cudaFree (devPtr_solution);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //Matrice problema ((literals<<1) x clauses)
    stat = cublasSetMatrix (literals<<1, clauses, sizeof(*problem_matrix), problem_matrix, literals<<1, devPtr_problem, literals<<1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data download failed (problem): " << stat << endl;
        cudaFree (devPtr_problem);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //Matrice risultato (1<<literals x clauses)
    //Non serve copiare la matrice risultato sul device
    /*
    stat=cublasSetMatrix (1<<literals, clauses, sizeof(*result_matrix), result_matrix, 1<<literals, devPtr_result, 1<<literals);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data download failed (result): " << stat << endl;
        cudaFree (devPtr_result);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    */
    //--------------------------------------
    //Moltiplicazione con GemmEx
    //--------------------------------------
    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, //cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    1<<literals, clauses, literals<<1, //int m, int n, int k,
    &alpha, devPtr_solution, CUDA_DATA_TYPE, 1<<literals, //const void *alpha, const void *A, cudaDataType Atype, int lda,
    devPtr_problem, CUDA_DATA_TYPE, literals<<1, //const void *B, cudaDataType Btype, int ldb,
    &beta, devPtr_result, CUDA_RESULT_TYPE, 1<<literals, //const void *beta, void *C, cudaDataType Ctype, int ldc,
    CUDA_ALGO, CUBLAS_GEMM_DEFAULT); //cudaDataType computeType, cublasGemmAlgo_t algo

    //Moltiplicazione con Sgemm
    /*
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, //cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    1<<literals, clauses, literals<<1, //int m, int n, int k,
    &alpha, devPtr_solution, 1<<literals,  //const float *alpha, const float *A, int lda,
    devPtr_problem, literals<<1, //const float *B, int ldb,
    &beta, devPtr_result, 1<<literals); //const float *beta, float *C, int ldc)
    */

    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Kernel execution failed: " << stat << endl;
        //Stampo il tipo di errore
        if(stat == CUBLAS_STATUS_NOT_INITIALIZED)
            cerr << "CUBLAS_STATUS_NOT_INITIALIZED" << endl;
        else if(stat == CUBLAS_STATUS_INVALID_VALUE)
            cerr << "CUBLAS_STATUS_INVALID_VALUE" << endl;
        else if(stat == CUBLAS_STATUS_ARCH_MISMATCH)
            cerr << "CUBLAS_STATUS_ARCH_MISMATCH" << endl;
        else if(stat == CUBLAS_STATUS_EXECUTION_FAILED)
            cerr << "CUBLAS_STATUS_EXECUTION_FAILED" << endl;
        else if(stat == CUBLAS_STATUS_NOT_SUPPORTED)
            cerr << "CUBLAS_STATUS_NOT_SUPPORTED" << endl;
        else cerr << "Unknown error" << endl;
        
        cudaFree (devPtr_problem);
        cudaFree (devPtr_solution);
        cudaFree (devPtr_result);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //--------------------------------------
    //Copia dei risultati sul device
    //--------------------------------------
    //Matrice risultato (1<<literals x clauses)
    stat = cublasGetMatrix (1<<literals, clauses, sizeof(*result_matrix), devPtr_result, 1<<literals, result_matrix, 1<<literals);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data upload failed (problem): " << stat << endl;
        //Stampo il tipo di errore
        if(stat == CUBLAS_STATUS_MAPPING_ERROR)
            cerr << "CUBLAS_STATUS_MAPPING_ERROR" << endl;
        else if(stat == CUBLAS_STATUS_INVALID_VALUE)
            cerr << "CUBLAS_STATUS_INVALID_VALUE" << endl;

        cudaFree (devPtr_problem);
        cudaFree (devPtr_solution);
        cudaFree (devPtr_result);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //--------------------------------------
    //Deallocazione memoria device
    //--------------------------------------
    cudaFree(devPtr_problem);
    cudaFree(devPtr_solution);
    cudaFree(devPtr_result);
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
    RESULT_TYPE *result_matrix;
    //Leggo i dati dal file di input
    std::tie(literals, clauses, problem_matrix) = readDimacsFile2Column(filename);
    //Alloco la matrice di soluzione
    solution_matrix = createSolutionMatrix(literals);
    //Alloco la matrice di risultato
    result_matrix = (RESULT_TYPE*)calloc((1<<literals)*clauses, sizeof(*result_matrix));

    //Stampo la matrice in input
    printInputMatrix(literals, clauses, problem_matrix);
    //Stampo la matrice di soluzione
    printSolutionMatrix(literals, solution_matrix);

    cout << endl << endl;
    if(cublas(literals, clauses, solution_matrix, problem_matrix, result_matrix))
        cout << "Test failed" << endl;
    else
        cout << "Test passed" << endl;

    //Stampo la matrice di risultato
    printResultMatrix(literals, clauses, result_matrix);
    
    free(problem_matrix);
    free(solution_matrix);
    free(result_matrix);

    cout << endl << "-------------------------------------------------------------------" << endl << "END" << endl;

    return 0;
}
