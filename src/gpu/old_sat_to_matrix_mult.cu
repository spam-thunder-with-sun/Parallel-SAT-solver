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
//Librerie selfmade
#include "create_matrix.h"


#define mydatatype unsigned short
#define mysolutiontype unsigned long long int

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0') 

pair<mydatatype *, mysolutiontype *> copy_to_gpu(vector<vector<int>> &, mysolutiontype, int, int);
mysolutiontype copy_from_gpu(mydatatype *, mysolutiontype *);
__host__ __device__  void printCompressMatrix(mydatatype *, int, int);
__global__ void mykernel(mydatatype *, mysolutiontype *, int, int, int);

//Non testato
void test_SAT_parallelo(mydatatype *matrix_h, int literals, int clauses)
{
    cout << endl << "Test SAT parallelo" << endl;
    /*
        bool val = (sol >> ((nChunksEveryHalfClause - chunk - 1) * (sizeof(mydatatype) * 8))) & (-1 << (sizeof(mydatatype) * 8)); 
        booly |= matrix[nChunksEveryClause * clause + chunk] & val;
    */
    int chunkSize = sizeof(mydatatype) * 8;
    int nChunksEveryHalfClause = ceil((float)literals / (sizeof(mydatatype) * 8));
    int nChunksEveryClause = nChunksEveryHalfClause * 2; 
    cout << "nChunksEveryHalfClause: " << nChunksEveryHalfClause << endl;
    cout << "nChunksEveryClause: " << nChunksEveryClause << endl;
    cout << "chunkSize: " << chunkSize << endl;
    cout << endl; 
    
    //Test SAT Parallelo
    for(unsigned long long sol = ((unsigned long long)1 << literals) - 1; sol < (unsigned long long)1 << literals; ++sol)
    {
        bitset<42> x(sol);
        cout << "Sol " << sol << ": " << x << endl;
        bool booly = true;
        for(int clause = 0; clause < clauses; ++clause)
        {
            booly = false;
            for(int chunk = 0; chunk < nChunksEveryHalfClause; ++chunk)
            {
                //Cosa devo fare qui??
                //Prendere il chunk corrispondente dalla matrice e fare l'AND con la soluzione
                //Ho in matrix_h[nChunksEveryClause * clause + chunk] il chunk attuale
                //Devo estrarre dalla soluzione una porzione di bit pari alla dimensione del chunk
                
                //Estraggo la soluzione corrispondente al chunk:
                //Creo una maschera di tutti bit a 1 di chunkSize bit
                unsigned mask = ((1 << chunkSize) - 1); 
                //Spingo a destra la soluzione di chunkSize * chunk bit
                unsigned sol_shifted = sol >> (chunkSize * (nChunksEveryHalfClause - chunk - 1));
                //Calcolo l'AND tra la soluzione spostata e la maschera
                unsigned sol_masked = sol_shifted & mask;
                //Calcolo l'AND tra la soluzione mascherata e il chunk della matrice
                unsigned result = sol_masked & matrix_h[nChunksEveryClause * clause + chunk];
                //Se il risultato è diverso da 0, allora la clausola è vera
                bitset<16> y(result);
                cout << y  << " ";
            }
            cout << endl;
        }
    }
    

    /*
    int startBit = (nChunksEveryHalfClause - chunk - 1) * chunkSize;
    unsigned mask = ((1 << chunkSize) - 1) << startBit;
    std::bitset<16> y(mask);
    cout << y  << "(" << startBit << ")";
    booly |= sol & mask;
                    bitset<16> y(matrix_h[nChunksEveryClause * clause + chunk]);
    cout << y  << " ";
    */
}


int main() 
{
    cout << "-----------------------------------------------" << endl;
    cout << endl << "*** Start ***" << endl << endl;
    //input/dimacs/jnh1.cnf
    //input/3sat/uf20-01.cnf
    //input/small.cnf
    //input/tutorial.cnf
    //input/hole6.cnf
    CreateMatrix *matrix = new CreateMatrix("input/hole6.cnf", true);
    //CreateMatrix *matrix = new CreateMatrix("input/small.cnf", true);
    if (matrix->get_error())  return(1);
    vector<vector<bool>> bool_matrix = matrix->get_boolean_matrix();
    vector<vector<int>> int_matrix = matrix->get_int_matrix();
    int literals = matrix->get_literals();
    int clauses = matrix->get_clauses();

    //Soluzione seriale
    //find_solution(bool_matrix);

    //Copio in gpu la matrice e lo spazio per le soluzioni
    cout << "Copy to gpu: ";
    pair<mydatatype *, mysolutiontype *> gpu_pointer = copy_to_gpu(int_matrix, 0, literals, clauses);
    cout << "[DONE]" << endl;
    mydatatype *matrix_d = gpu_pointer.first;
    mysolutiontype *solutions_d = gpu_pointer.second;

    unsigned long long threadsPerBlock = min((unsigned long long)1024, ((unsigned long long)1 << literals));
    unsigned long long blocksPerGrid = min((((unsigned long long)1 << literals) + threadsPerBlock - 1) / threadsPerBlock, (unsigned long long)65535);
    unsigned long long nSolutionsToCompute = ceil(((unsigned long long)1 << literals) / (double)(threadsPerBlock * blocksPerGrid));
    cout << "Alcune informazioni: "  << endl;
    cout << "\tliterals: " << literals << endl;
    cout << "\tclauses: " << clauses << endl;
    cout << "\tthreadsPerBlock: " << threadsPerBlock << endl;
    cout << "\tblocksPerGrid: " << blocksPerGrid << endl;
    cout << "\tsolutionPerThead: " << nSolutionsToCompute << endl;
    cout << "\tsoluzioni da trovare: " << ((unsigned long long)1 << literals) << endl;

    //Invocazione del kernel
    cout << "Invocazione del kernel:" << endl;
    //mykernel<<<blocksPerGrid, threadsPerBlock>>>(matrix_d, solutions_d, literals, clauses, nSolutionsToCompute);

    //Copia da gpu le soluzioni e libero la memeoria gpu
    mysolutiontype nSoluzioni = copy_from_gpu(matrix_d, solutions_d);

    cout << "Numero di soluzioni trovate: " << nSoluzioni << endl;
    cout << endl << "*** End ***" << endl;
    cout << "-----------------------------------------------" << endl;

    return 0;
}

//Quello che c'è al momento funziona
__global__ void mykernel(mydatatype *matrix, mysolutiontype *solutions, int literals, int clauses, int nSolutionsToCompute)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int nChunksEveryHalfClause = ceil((float)literals / (sizeof(mydatatype) * 8));
    //int nChunksEveryClause = nChunksEveryHalfClause * 2;  

    //Al più ci sono 67107840 threads
    
    //Mi fermo quando ho finito di verificare tutte le nSolutionsToCompute soluzioni
    unsigned long long start = (unsigned long long)id * nSolutionsToCompute;
    unsigned long long stop = min(start + (unsigned long long)nSolutionsToCompute, ((unsigned long long)1 << literals));
    for(unsigned long long sol = start; sol < stop; ++sol)
    {
        bool booly = true;
        for(int clause = 0; clause < clauses && booly; ++clause)
        {
            booly = false;
            for(int chunk = 0; chunk < nChunksEveryHalfClause && !booly; ++chunk)
            {
                //TODO
                ;
            }
        }
    }

   if(id == 0)
   {
        printf("*****************************************\n");
        printf("HELLO (EVIL) WORLD FROM GPU\n");
        //printCompressMatrix(matrix, literals, clauses);
        atomicAdd(&solutions[0], 1);
        printf("*****************************************\n");
   }

   return;
}

//Testato funzionante
//I dati sono salvati in formato big endian, quindi il bit più significativo è il primo
pair<mydatatype *, mysolutiontype *> copy_to_gpu(vector<vector<int>> &matrix, mysolutiontype nSolutions, int literals, int clauses)
{
    //int chunkSize = sizeof(mydatatype) * 8;
    int nChunksEveryHalfClause = ceil((float)literals / (sizeof(mydatatype) * 8));
    int nChunksEveryClause = nChunksEveryHalfClause * 2;
    size_t sizeMatrix = nChunksEveryClause * clauses * (sizeof(mydatatype));
    mydatatype *matrix_h, *matrix_d;
    mysolutiontype *solutions_h, *solutions_d;

    //Copia matrice
    //---------------------------------------------------

    //Alloco il vettore lato host
    matrix_h = (mydatatype*)malloc(sizeMatrix);
    //Riempio il vettore lato host
    for(int i = 0; i < matrix.size(); ++i)
    {
        //Inizializzo a tutti 0
        for(int j = 0; j < nChunksEveryClause; ++j)
            matrix_h[nChunksEveryClause * i + j] = 0;
        //Setto a 1 i bit corrispondenti
        for(int j = 0; j < matrix[i].size(); ++j)
        {
            int _value = abs(matrix[i][j]);
            int offset = (int)(_value-1) % (int)(sizeof(mydatatype) * 8);
            int chunk = nChunksEveryHalfClause - 1 - ((int)(_value-1) / (int)(sizeof(mydatatype) * 8));
            if(matrix[i][j] > 0)
                matrix_h[nChunksEveryClause * i + chunk] |= 1 << offset;
            else if(matrix[i][j] < 0)
                matrix_h[nChunksEveryClause * i + nChunksEveryHalfClause + chunk ] |= 1 << offset;
        }
    }  

    //Stampo il vettore lato host
    //printCompressMatrix(matrix_h, literals, clauses);

    //Alloco il vettore lato device copiandolo da quello lato host
    cudaMalloc((void**)&matrix_d, sizeMatrix);
    cudaMemcpy(matrix_d, matrix_h, sizeMatrix, cudaMemcpyHostToDevice);

    //Test SAT Parallelo
    test_SAT_parallelo(matrix_h, literals, clauses);

    //Libero lo spazio lato host
    free(matrix_h);

    //Copia soluzioni
    //---------------------------------------------------

    //Alloco il vettore lato host
    solutions_h = (mysolutiontype*)malloc(sizeof(mysolutiontype));
    //Riempio il vettore lato host
    solutions_h[0] = nSolutions;

    //Alloco il vettore lato device copiandolo da quello lato host
    cudaMalloc((void**)&solutions_d, sizeof(mysolutiontype));
    cudaMemcpy(solutions_d, solutions_h, sizeof(mysolutiontype), cudaMemcpyHostToDevice);

    //Libero lo spazio lato host
    free(solutions_h);

    return make_pair(matrix_d, solutions_d);
}

//Testato funzionante
mysolutiontype copy_from_gpu(mydatatype *matrix_d, mysolutiontype *solutions_d)
{
    //Libero la memoria lato device della matrice
    cudaFree(matrix_d);

    //Alloco il vettore lato host
    mysolutiontype *solutions_h = (mysolutiontype*)malloc(sizeof(mysolutiontype));

    //Copio il numero di soluzioni
    cudaMemcpy(solutions_h, solutions_d, sizeof(mysolutiontype), cudaMemcpyDeviceToHost);

    //Copio il numero di soluzioni
    mysolutiontype nSolutions = solutions_h[0];

    //Libero la memoria lato device delle soluzioni
    cudaFree(solutions_d);
    //Libero la memoria lato host delle soluzioni
    free(solutions_h);

    return nSolutions;
}

//Testato funzionante (solo con tipo di dato 2 byte)
__host__ __device__ void printCompressMatrix(mydatatype *matrix, int literals, int clauses)
{
    int nChunksEveryHalfClause = ceil((float)literals / (sizeof(mydatatype) * 8));
    int nChunksEveryClause = nChunksEveryHalfClause * 2;

    printf("Print compress matrix: \n");
    printf("nChunksEveryHalfClause: %d\n", nChunksEveryHalfClause);
    printf("nChunksEveryClause: %d\n", nChunksEveryClause);
    printf("literals: %d\n", literals);
    printf("clauses: %d\n", clauses);
    printf("sizeof(mydatatype): %d\n", sizeof(mydatatype));
    printf("\n");

    for(int i = 0; i < clauses; ++i)
    {
        for(int j = 0; j < nChunksEveryClause; ++j)
        {
            //bitset<sizeof(mydatatype) * 8> x(matrix[nChunksEveryClause * i + j]);
            //cout << x << " ";
            //cout << x << "(" << matrix[nChunksEveryClause * i + j] << ") ";
            //printf("%s", x.to_string().c_str());
            printf(""BYTE_TO_BINARY_PATTERN""BYTE_TO_BINARY_PATTERN" ", 
            BYTE_TO_BINARY(matrix[nChunksEveryClause * i + j]>>8), BYTE_TO_BINARY(matrix[nChunksEveryClause * i + j]));
        }
        printf("\n");
    }
    printf("\n");
}