#ifndef PRINT_DATA_H
    #define PRINT_DATA_H

    #include <iostream>
    #include <string>
    #include <vector>
    #include <tuple>
    #include <unordered_set>
    #include "constant.h"

    void printInputMatrix(INT_TYPE, INT_TYPE, std::vector<std::vector<INT_TYPE>>);
    void printInputMatrix(INT_TYPE, INT_TYPE, std::vector<std::unordered_set<INT_TYPE>>);
    void printInputMatrix(INT_TYPE, INT_TYPE, DATA_TYPE*);

    void printSolutionMatrix(INT_TYPE, DATA_TYPE*);

    void printResultMatrix(INT_TYPE, INT_TYPE, RESULT_TYPE*);

#endif

