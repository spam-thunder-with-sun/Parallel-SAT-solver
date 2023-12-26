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

int cublas(INT_TYPE literals, INT_TYPE clauses, DATA_TYPE* problem_matrix, DATA_TYPE* solution_matrix, RESULT_TYPE* result_matrix)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    DATA_TYPE *devPtr_problem, *devPtr_solution;
    RESULT_TYPE *devPtr_result;

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

    cudaStat = cudaMalloc ((void**)&devPtr_result, 1<<literals*clauses*sizeof(*result_matrix));
    if (cudaStat != cudaSuccess) {
        cerr << "Device memory allocation (result) failed: " << cudaStat << endl;
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
    RESULT_TYPE alpha = 1;
    RESULT_TYPE beta = 0;

    //Moltiplicazione
    stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1<<literals, clauses, literals<<1, &alpha, devPtr_problem, CUDA_R_8I, 1<<literals, devPtr_solution, CUDA_R_8I, literals<<1, &beta, result_matrix, CUDA_R_32I, 1<<literals, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Kernel execution failed: " << stat << endl;
        cudaFree (devPtr_problem);
        cudaFree (devPtr_solution);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    //Copia dei risultati sul device
    stat = cublasGetMatrix (1<<literals, clauses, sizeof(*result_matrix), devPtr_result, 1<<literals, result_matrix, 1<<literals);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cerr << "Data upload failed (problem): " << stat << endl;
        cudaFree (devPtr_problem);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

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
    result_matrix = (RESULT_TYPE*)calloc(1<<literals*clauses, sizeof(*result_matrix));

    //Stampo la matrice in input
    printInputMatrix(literals, clauses, problem_matrix);
    //Stampo la matrice di soluzione
    printSolutionMatrix(literals, solution_matrix);

    cout << endl << endl;
    if(cublas(literals, clauses, problem_matrix, solution_matrix, result_matrix))
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
