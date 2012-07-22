from libcpp.string cimport string

cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        pass
    cdef cppclass ofstream(ostream):
        ofstream(char* filename)
        void close()

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

cdef extern from "<fst/script/print-impl.h>":
    cdef cppclass FstPrinter "fst::FstPrinter<fst::StdArc>":
        FstPrinter(fst.StdVectorFst& fst,
                   sym.SymbolTable *isyms,
                   sym.SymbolTable *osyms,
                   sym.SymbolTable *ssyms,
                   bint accep,
                   bint show_weight_one)

        void Print(ostream *strm, string &dest)
