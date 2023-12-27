#ifndef CONSTANT
    #define CONSTANT

    #define INT_TYPE int
    #define DATA_TYPE float
    #define RESULT_TYPE float
    #define PRINT_DATA float

    #define CUDA_DATA_TYPE CUDA_R_32F
    #define CUDA_RESULT_TYPE CUDA_R_32F
    #define CUDA_ALGO CUBLAS_COMPUTE_32F 
    
    #define IDX2C(i,j,ld) (((j)*(ld))+(i))
#endif