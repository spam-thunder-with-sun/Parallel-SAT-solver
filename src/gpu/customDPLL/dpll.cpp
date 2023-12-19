#include "dpll.h"

using namespace std;

vector<unordered_set<INT_TYPE>>* CreateNewMatrix(const vector<unordered_set<INT_TYPE>>* matrix, INT_TYPE literal)
{
    vector<unordered_set<INT_TYPE>>* new_matrix = new vector<unordered_set<INT_TYPE>>;

    for(INT_TYPE i = 0; i < (*matrix).size(); ++i)
    {
        unordered_set<INT_TYPE> new_row;
        bool found = false;

        for(auto it = (*matrix)[i].begin(); it != (*matrix)[i].end(); ++it)
        {
            //Se trovo il letterale 
            if(*it == literal)
            {
                found = true;
                break;
            }//Se non trovo il letterale negato
            else if(*it != -literal)
                new_row.insert(*it);
        }              

        if(!found)
        {
            if(new_row.size() > 0)
                (*new_matrix).push_back(new_row);
            else 
                return nullptr;
        }
    }

    return new_matrix;
}

bool DPLLModel(const vector<unordered_set<INT_TYPE>>* matrix, stack<INT_TYPE> literals_stack)
{
    if(!matrix)
        return false;
    else if((*matrix).empty())
    {
        cout << "Literals: ";
        while(!literals_stack.empty())
        {
            cout << literals_stack.top() << " ";
            literals_stack.pop();
        }
        cout << endl;
        return true;
    }
    else
    {
        //Find a unit clause
        for(INT_TYPE i = 0; i < (*matrix).size(); ++i)
            if((*matrix)[i].size() == 1)
            {
                literals_stack.push(*(*matrix)[i].begin());
                #if DPLL_DEBUG == true
                    cout << "Unit clause: " << *(*matrix)[i].begin() << endl;
                #endif
                return DPLLModel(CreateNewMatrix(matrix, *((*matrix)[i].begin())), literals_stack);
            }
                
        //Insert all literals in the map
        unordered_map<INT_TYPE, INT_TYPE> literals;
        for(INT_TYPE i = 0; i < (*matrix).size(); ++i)
            for(auto it = (*matrix)[i].begin(); it != (*matrix)[i].end(); ++it)
            {
                if(literals.find(*it) == literals.end())
                    literals.insert({*it, 1});
                else
                    literals[*it]++;
            }

        //Find a pure literal
        for(auto it = literals.begin(); it != literals.end(); ++it)
            if(literals.find(-(it->first)) == literals.end())
            {
                literals_stack.push(it->first);
                #if DPLL_DEBUG == true
                cout << "Pure literal: " << it->first << endl;
                #endif
                return DPLLModel(CreateNewMatrix(matrix, it->first), literals_stack);
            }

        //Find the literal with the highest frequency
        INT_TYPE literal;
        INT_TYPE max = 0;
        for(auto it = literals.begin(); it != literals.end(); ++it)
            if(it->second > max)
            {
                max = it->second;
                literal = it->first;
            }

        #if DPLL_DEBUG == true
        cout << "Highest frequency literal: " << literal << endl;
        #endif

        literals_stack.push(literal);
        if(DPLLModel(CreateNewMatrix(matrix, literal), literals_stack))
            return true;
        else
        {
            literals_stack.pop();
            literals_stack.push(-literal);
            return DPLLModel(CreateNewMatrix(matrix, -literal), literals_stack);
        }
        return false;

        //return DPLL(CreateNewMatrix(matrix, literal)) || DPLL(CreateNewMatrix(matrix, -literal));
    }
}

bool DPLL(const vector<unordered_set<INT_TYPE>>* matrix)
{
    if(!matrix)
        return false;
    else if((*matrix).empty())
        return true;
    else
    {
        //Find a unit clause
        for(INT_TYPE i = 0; i < (*matrix).size(); ++i)
            if((*matrix)[i].size() == 1)
            {
                #if DPLL_DEBUG == true
                    cout << "Unit clause: " << *(*matrix)[i].begin() << endl;
                #endif
                return DPLL(CreateNewMatrix(matrix, *((*matrix)[i].begin())));
            }
                
        //Insert all literals in the map
        unordered_map<INT_TYPE, INT_TYPE> literals;
        for(INT_TYPE i = 0; i < (*matrix).size(); ++i)
            for(auto it = (*matrix)[i].begin(); it != (*matrix)[i].end(); ++it)
            {
                if(literals.find(*it) == literals.end())
                    literals.insert({*it, 1});
                else
                    literals[*it]++;
            }

        //Find a pure literal
        for(auto it = literals.begin(); it != literals.end(); ++it)
            if(literals.find(-(it->first)) == literals.end())
            {
                #if DPLL_DEBUG == true
                cout << "Pure literal: " << it->first << endl;
                #endif
                return DPLL(CreateNewMatrix(matrix, it->first));
            }

        //Find the literal with the highest frequency
        INT_TYPE literal;
        INT_TYPE max = 0;
        for(auto it = literals.begin(); it != literals.end(); ++it)
            if(it->second > max)
            {
                max = it->second;
                literal = it->first;
            }

        #if DPLL_DEBUG == true
        cout << "Highest frequency literal: " << literal << endl;
        #endif

        return DPLL(CreateNewMatrix(matrix, literal)) || DPLL(CreateNewMatrix(matrix, -literal));
    }
}

