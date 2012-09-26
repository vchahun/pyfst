from libcpp.string cimport string

cdef extern from "<iostream>" namespace "std":
    ctypedef string const_string "const string"
    cdef cppclass ostream:
        pass
    cdef cppclass istream:
        pass
    cdef cppclass ostringstream(ostream):
        ostringstream()
        string str()
    cdef cppclass ifstream(istream):
        ifstream(char* filename)

cimport fst
cimport sym

cdef extern from "<fst/script/draw-impl.h>":
    cdef cppclass FstDrawer "fst::FstDrawer<fst::StdArc>":
        FstDrawer(fst.StdVectorFst& fst, 
                  sym.SymbolTable *isyms,
                  sym.SymbolTable *osyms,
                  sym.SymbolTable *ssyms,
                  bint accep,
                  string title,
                  float width,
                  float height,
                  bint portrait,
                  bint vertical, 
                  float ranksep,
                  float nodesep,
                  int fontsize,
                  int precision,
                  bint show_weight_one)

        void Draw(ostream *strm, string &dest)
