#ifndef CONSTANT
    #define CONSTANT

    #define INT_TYPE int32_t

    #define COMPUTATION_FLOAT 1
    #define COMPUTATION_HALF_FLOAT 2
    #define COMPUTATION_INTEGER 3
    #define COMPUTATION_INT8_FLOAT 4

    #define COMPUTATION_TYPE COMPUTATION_FLOAT

    #if COMPUTATION_TYPE == COMPUTATION_FLOAT
        //FLOAT COMPUTATION 32 bits
        #define DATA_TYPE float
        #define RESULT_TYPE float
        #define PRINT_DATA float
        #define CUDA_DATA_TYPE CUDA_R_32F
        #define CUDA_RESULT_TYPE CUDA_R_32F
        #define CUDA_ALGO CUBLAS_COMPUTE_32F 
    #elif COMPUTATION_TYPE == COMPUTATION_HALF_FLOAT
        //HALF FLOAT COMPUTATION 16 bits
        #define DATA_TYPE half
        #define RESULT_TYPE half
        #define PRINT_DATA float
        #define CUDA_DATA_TYPE CUDA_R_16F
        #define CUDA_RESULT_TYPE CUDA_R_16F
        #define CUDA_ALGO CUBLAS_COMPUTE_16F
    #elif COMPUTATION_TYPE == COMPUTATION_INTEGER
        //INT8 INPUT and INT32 COMPUTATION and INT32 results
        #define DATA_TYPE int8_t
        #define RESULT_TYPE int32_t
        #define PRINT_DATA int
        #define CUDA_DATA_TYPE CUDA_R_8I
        #define CUDA_RESULT_TYPE CUDA_R_32I
        #define CUDA_ALGO CUBLAS_COMPUTE_32I
    #elif COMPUTATION_TYPE == COMPUTATION_INT8_FLOAT
        //INT8 INPUT and FLOAT COMPUTATION and FLOAT results
        #define DATA_TYPE int8_t
        #define RESULT_TYPE float
        #define PRINT_DATA float
        #define CUDA_DATA_TYPE CUDA_R_8I
        #define CUDA_RESULT_TYPE CUDA_R_32F
        #define CUDA_ALGO CUBLAS_COMPUTE_32F
    #else
        #error "COMPUTATION_TYPE not defined"
    #endif
    
    #define IDX2C(i,j,ld) (((j)*(ld))+(i))
#endif