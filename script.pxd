from libcpp.string cimport string
from util cimport ostream

cimport cfst
cimport sym

cdef extern from "<fst/script/draw-impl.h>":
    cdef cppclass FstDrawer "fst::FstDrawer<fst::StdArc>":
        FstDrawer(cfst.StdVectorFst& fst, 
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
