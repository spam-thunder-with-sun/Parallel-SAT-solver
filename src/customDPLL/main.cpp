#include <iostream>
#include <unordered_map>
#include <stack>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <thread>

#include "read_data.h"
#include "dpll.h"
#include "constant.h"


using namespace std;

void wrap_DPLL(vector<unordered_set<INT_TYPE>>* matrix, bool *ris)
{
    *ris = DPLL(matrix);

    /*
    stack<INT_TYPE> s;
    bool result = DPLLModel(&matrix, s);
    */
}

int main(int argc, char *argv[]) 
{
    string filename = "../input/dimacs/small.cnf";

    if(argc > 1)
        filename = argv[1];

    /*
    INT_TYPE literals, clauses;
    vector<unordered_set<INT_TYPE>> matrix;
    std::tie(literals, clauses, matrix) = readDimacsFile2Hashset(filename);
    printInputMatrix(literals, clauses, matrix);

    bool ris = false;
    const clock_t c_start = clock();
    auto t_start = chrono::high_resolution_clock::now();
    thread t1(wrap_DPLL, &matrix, &ris);
    t1.join();
    const clock_t c_end = clock();
    const auto t_end = chrono::high_resolution_clock::now();

    cout << "Literals: " << literals << " Clauses: " << clauses << endl;
    cout << "SAT: " << ris << endl;
    cout << fixed << setprecision(3) << "CPUTime: " << (double)(c_end - c_start) / CLOCKS_PER_SEC << " s" << endl;
    */

    INT_TYPE literals, clauses;
    DATA_TYPE* matrix;

    std::tie(literals, clauses, matrix) = readDimacsFile2Column(filename);
    printInputMatrix(literals, clauses, matrix);

    return 0;
}

