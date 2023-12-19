#ifndef DPLL_H
    #define DPLL_H

    #include <iostream>
    #include <unordered_map>
    #include <unordered_set>
    #include <vector>
    #include <stack>
    #include "constant.h"

    std::vector<std::unordered_set<INT_TYPE>>* CreateNewMatrix(const std::vector<std::unordered_set<INT_TYPE>>*, INT_TYPE);
    bool DPLLModel(const std::vector<std::unordered_set<INT_TYPE>>*, std::stack<INT_TYPE>);
    bool DPLL(const std::vector<std::unordered_set<INT_TYPE>>*);

#endif