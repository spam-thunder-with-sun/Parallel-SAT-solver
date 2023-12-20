#ifndef READ_DATA_H
    #define READ_DATA_H

    #include <iostream>
    #include <string>
    #include <sstream>
    #include <fstream>
    #include <vector>
    #include <tuple>
    #include <unordered_set>
    #include "constant.h"

    typedef void (*FuncPtrResizeMatrix)(INT_TYPE, INT_TYPE, void*);
    typedef void (*FuncPtrAddLiteral)(INT_TYPE, INT_TYPE, INT_TYPE, INT_TYPE, void*);

    std::tuple<INT_TYPE, INT_TYPE, std::vector<std::vector<INT_TYPE>>> readDimacsFile2Vec(std::string);
    std::tuple<INT_TYPE, INT_TYPE, std::vector<std::unordered_set<INT_TYPE>>> readDimacsFile2Hashset(std::string);
    std::tuple<INT_TYPE, INT_TYPE, DATA_TYPE *> readDimacsFile2Column(std::string);
    std::tuple<INT_TYPE, INT_TYPE> readDimacsFile_generic(std::string, void*, FuncPtrResizeMatrix, FuncPtrAddLiteral);
    void print_matrix(INT_TYPE, INT_TYPE, std::vector<std::vector<INT_TYPE>>);
    void print_matrix(INT_TYPE, INT_TYPE, std::vector<std::unordered_set<INT_TYPE>>);
    void print_matrix(INT_TYPE, INT_TYPE, DATA_TYPE *);

#endif

