from libcpp.string cimport string
from util cimport istream

cdef extern from "<fst/symbol-table.h>" namespace "fst":    
    cdef cppclass SymbolTable:
        SymbolTable(SymbolTable&)
        SymbolTable(string &name)
        long AddSymbol(string &symbol, long key)
        long AddSymbol(string &symbol)
        string& Name()
        bint Write(string &filename)
        #WriteText (ostream &strm)
        string Find(long key)
        string Find(char* symbol)
        unsigned NumSymbols()

    cdef SymbolTable* SymbolTableRead "fst::SymbolTable::Read" (istream &strm,
            string& source)
