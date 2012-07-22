from libcpp.string cimport string

cdef extern from "<fst/symbol-table.h>" namespace "fst":    
    cdef cppclass SymbolTable:
        SymbolTable(string &name)
        long AddSymbol(string &symbol, long key)
        long AddSymbol(string &symbol)
        string& Name()
        bint Write(string &filename)
        #WriteText (ostream &strm)
        string Find(long key)
        string Find(char* symbol)
        unsigned NumSymbols()
