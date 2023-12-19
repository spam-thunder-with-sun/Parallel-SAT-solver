#include "read_data.h"

using namespace std;

tuple<INT_TYPE, INT_TYPE> readDimacsFile_generic(string filename, void* matrix, FuncPtrResizeMatrix resizeMatrix, FuncPtrAddLiteral addLiteral)
{
    //Apro il file in lettura
    ifstream in (filename);
    if (!in.is_open()) 
    {
        cerr << "Error opening file \"" << filename << "\"" << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    istringstream line_buf;                
    char first_char;
    INT_TYPE row = 0;
    INT_TYPE tmp;

    //Dati del problema
    INT_TYPE literals, clauses;
    string problem_name;

    while(in)
    {
        //Pulisco la linea
        line.clear();

        //Leggo linea
        getline(in, line);

        //Se la linea non è vuota
        if(line.empty())
            continue;

        //Copio in un stringstream
        line_buf.clear();
        line_buf.str(line);

        //Leggo il primo carattere della linea
        line_buf >> first_char;

        switch(first_char)
        {
            case 'c':
                //Commento
                #if DEBUG == true
                    cout << line << endl;
                #endif
                break;
            case 'p':
                //Dati del problema
                line_buf >> problem_name >> literals >> clauses;

                #if DEBUG == true  
                    cout << "Problem name:" << problem_name << " literals:" << literals << " clauses:" << clauses << endl;
                #endif

                if(problem_name != "cnf" && problem_name != "CNF")
                {
                    cerr << "Wrong type of file" << endl;
                    exit(EXIT_FAILURE);
                }

                if(literals == 0 || clauses == 0)
                {
                    cerr << "Wrong number of literals or clauses" << endl;
                    exit(EXIT_FAILURE);
                }

                //Alloco la matrice
                resizeMatrix(clauses, matrix);

                //Leggo il resto del problema
                for(INT_TYPE row = 0; row < clauses; ++row)
                {
                    //Leggo un numero
                    in >> tmp;

                    while(tmp != 0)
                    {
                        //Aggiungo il numero alla matrice
                        addLiteral(row, tmp, matrix);

                        //Leggo il prossimo numero
                        in >> tmp;
                    }
                }

                break;
            default:
                cerr << "Error reading the file" << endl;
                exit(EXIT_FAILURE);
                break;
        }
    }

    return make_tuple(literals, clauses);
}

tuple<INT_TYPE, INT_TYPE, vector<vector<INT_TYPE>>> readDimacsFile2Vec(string filename)
{
    //Dichiaro la matrice 
    vector<vector<INT_TYPE>> matrix;

    auto resizeMatrix = [](INT_TYPE clauses, void* matrix)->void{
        vector<vector<INT_TYPE>>* m = (vector<vector<INT_TYPE>>*) matrix;
        m->resize(clauses);
    };

    auto addLiteral = [](INT_TYPE row, INT_TYPE literal, void* matrix)->void{
        vector<vector<INT_TYPE>>* m = (vector<vector<INT_TYPE>>*) matrix;
        (*m)[row].push_back(literal);
    };

    INT_TYPE literals, clauses;
    tie(literals, clauses) = readDimacsFile_generic(filename, &matrix, resizeMatrix, addLiteral);
    return make_tuple(literals, clauses, matrix);
}

tuple<INT_TYPE, INT_TYPE, vector<unordered_set<INT_TYPE>>> readDimacsFile2Hashset(string filename)
{
    //Dichiaro la matrice 
    vector<unordered_set<INT_TYPE>> matrix;

    auto resizeMatrix = [](INT_TYPE clauses, void* matrix)->void{
        vector<unordered_set<INT_TYPE>>* m = (vector<unordered_set<INT_TYPE>>*) matrix;
        m->resize(clauses);
    };

    auto addLiteral = [](INT_TYPE row, INT_TYPE literal, void* matrix)->void{
        vector<unordered_set<INT_TYPE>>* m = (vector<unordered_set<INT_TYPE>>*) matrix;
        (*m)[row].insert(literal);
    };

    INT_TYPE literals, clauses;
    tie(literals, clauses) = readDimacsFile_generic(filename, &matrix, resizeMatrix, addLiteral);
    return make_tuple(literals, clauses, matrix);
}

void print_matrix(INT_TYPE literals, INT_TYPE clauses, vector<vector<INT_TYPE>> matrix)
{
    cout << "Literals: " << literals << endl;
    cout << "Clauses: " << clauses << endl;
    cout << "Matrix:" << endl;
    for(INT_TYPE i = 0; i < matrix.size(); ++i)
    {
        vector<INT_TYPE> row = matrix[i];
        for (INT_TYPE j = 0; j < row.size(); ++j)
            cout << matrix[i][j] << " ";
        cout << endl;
    }
}

void print_matrix(INT_TYPE literals, INT_TYPE clauses, vector<unordered_set<INT_TYPE>> matrix)
{
    cout << "Literals: " << literals << endl;
    cout << "Clauses: " << clauses << endl;
    cout << "Matrix:" << endl;
    for(INT_TYPE i = 0; i < matrix.size(); ++i)
    {
        unordered_set<INT_TYPE> row = matrix[i];
        for(auto it = row.begin(); it != row.end(); ++it)
            cout << *it << " ";
        cout << endl;
    }
}
