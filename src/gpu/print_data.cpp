#include "print_data.h"

using namespace std;

void printInputMatrix(INT_TYPE literals, INT_TYPE clauses, vector<vector<INT_TYPE>> matrix)
{
    cout << "Literals: " << literals << endl;
    cout << "Clauses: " << clauses << endl;
    cout << "Matrix:" << endl;
    for(long unsigned int i = 0; i < matrix.size(); ++i)
    {
        vector<INT_TYPE> row = matrix[i];
        for (long unsigned int j = 0; j < row.size(); ++j)
            cout << matrix[i][j] << " ";
        cout << endl;
    }
}

void printInputMatrix(INT_TYPE literals, INT_TYPE clauses, vector<unordered_set<INT_TYPE>> matrix)
{
    cout << "Literals: " << literals << endl;
    cout << "Clauses: " << clauses << endl;
    cout << "Matrix:" << endl;
    for(long unsigned int i = 0; i < matrix.size(); ++i)
    {
        unordered_set<INT_TYPE> row = matrix[i];
        for(auto it = row.begin(); it != row.end(); ++it)
            cout << *it << " ";
        cout << endl;
    }
}

void printInputMatrix(INT_TYPE literals, INT_TYPE clauses, DATA_TYPE* matrix)
{
    cout << "Literals: " << literals << endl;
    cout << "Clauses: " << clauses << endl;
    cout << "Input Matrix: " << endl;

    for(INT_TYPE i = 0; i < literals<<1; ++i)
    {
        for(INT_TYPE j = 0; j < clauses; ++j)
        {           
            cout << (int)matrix[IDX2C(i, j, literals<<1)] << " ";
        }
        
        cout << endl;
        if(i == literals - 1)
        {
            for(INT_TYPE j = 0; j < clauses; ++j)
                cout << "- ";
            cout << endl;
        }
    }
}

void printSolutionMatrix(INT_TYPE literals, DATA_TYPE* solution_matrix)
{
    cout << "Solution matrix:" << endl;
    for(INT_TYPE i = 0; i < 1<<literals; i++)
    {
        for(INT_TYPE j = 0; j < literals<<1; j++)
        {
            cout << (int)solution_matrix[IDX2C(i, j, 1<<literals)] << " ";

            if(j == literals-1)
                cout << "| ";
        }
        cout << endl;
    }
}

void printResultMatrix(INT_TYPE literals, INT_TYPE clauses, RESULT_TYPE* result_matrix)
{
    cout << "Result matrix:" << endl;
    for(INT_TYPE i = 0; i < 1<<literals; i++)
    {
        for(INT_TYPE j = 0; j < clauses; j++)
        {
            cout << (int)result_matrix[IDX2C(i, j, 1<<literals)] << " ";
        }
        cout << endl;
    }
}
