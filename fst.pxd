from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
cimport sym

cdef extern from "<fst/fstlib.h>" namespace "fst":
    cdef cppclass TropicalWeight:
        TropicalWeight(TropicalWeight)
        TropicalWeight(float value)
        float Value()
        bint operator==(TropicalWeight& other)

    cdef TropicalWeight Plus(TropicalWeight &w1, TropicalWeight& w2)
    cdef TropicalWeight Times(TropicalWeight &w1, TropicalWeight& w2)

    cdef TropicalWeight TropicalZero "fst::TropicalWeight::Zero" ()
    cdef TropicalWeight TropicalOne "fst::TropicalWeight::One" ()

    cdef cppclass StdArc "const fst::StdArc":
        StdArc(int ilabel, int olabel, TropicalWeight& weight, int nextstate)
        StdArc()
        int ilabel
        int olabel
        TropicalWeight weight
        int nextstate

    cdef cppclass ArcIterator "fst::ArcIterator<fst::StdVectorFst>":
        ArcIterator(StdVectorFst& fst, int state)
        bint Done()
        void Next()
        StdArc& Value()

    cdef cppclass StdVectorFst:
        StdVectorFst()
        int Start()
        TropicalWeight Final(int s)
        int NumStates()
        unsigned NumArcs(int s)
        void SetFinal(int s, TropicalWeight w)
        void SetStart(int s)
        int AddState()
        void AddArc(int s, StdArc &arc)
        bint Write(string& filename)
        StdVectorFst* Copy()
        sym.SymbolTable* MutableInputSymbols()
        sym.SymbolTable* MutableOutputSymbols()
        void SetInputSymbols(sym.SymbolTable* isyms)
        void SetOutputSymbols(sym.SymbolTable* osyms)

    cdef StdVectorFst* StdVectorFstRead "fst::StdVectorFst::Read" (string& filename)

    cdef cppclass ILabelCompare "fst::OLabelCompare<fst::StdArc>":
        pass

    cdef cppclass OLabelCompare "fst::OLabelCompare<fst::StdArc>":
        pass

    enum ProjectType:
        PROJECT_INPUT
        PROJECT_OUTPUT

    # const
    cdef void Determinize(StdVectorFst& ifst, StdVectorFst* ofst)
    cdef void Compose(StdVectorFst &ifst1, StdVectorFst &ifst2, StdVectorFst *ofst)
    # non const
    cdef void Minimize(StdVectorFst *ifst)
    cdef void ArcSort(StdVectorFst *fst, ILabelCompare& compare)
    cdef void ArcSort(StdVectorFst *fst, OLabelCompare& compare)
    cdef void Project(StdVectorFst *fst, ProjectType type)
    cdef void RmEpsilon(StdVectorFst* fst)
    cdef void Relabel(StdVectorFst* fst, 
            vector[pair[int, int]]& ipairs,
            vector[pair[int, int]]& opairs)
