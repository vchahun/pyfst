from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
cimport sym

cdef extern from "<fst/fstlib.h>" namespace "fst":

    cdef cppclass Weight:
        pass

    cdef cppclass Arc[W]:
        int ilabel
        int olabel
        int nextstate
        Arc(int ilabel, int olabel, W& weight, int nextstate)
        W weight

    ctypedef Arc[TropicalWeight] StdArc

    cdef cppclass ArcIterator[T]:
        ArcIterator(T& fst, int state)
        bint Done()
        void Next()
        Arc& Value()

    cdef cppclass Fst:
        int Start()
        TropicalWeight Final(int s)
        unsigned NumArcs(int s)
        Fst* Copy()
        bint Write(string& filename)

    cdef cppclass ExpandedFst(Fst):
        int NumStates()

    cdef cppclass MutableFst(ExpandedFst):
        int AddState()
        void SetFinal(int s, Weight w)
        void SetStart(int s)
        void SetInputSymbols(sym.SymbolTable* isyms)
        void SetOutputSymbols(sym.SymbolTable* osyms)
        sym.SymbolTable* MutableInputSymbols()
        sym.SymbolTable* MutableOutputSymbols()

    cdef cppclass TropicalWeight(Weight):
        float Value()
        TropicalWeight(float value)
        TropicalWeight(TropicalWeight weight)
        bint operator==(TropicalWeight& other)
        TropicalWeight& set_value "operator=" (TropicalWeight& other)

    cdef TropicalWeight Plus(TropicalWeight &w1, TropicalWeight& w2)
    cdef TropicalWeight Times(TropicalWeight &w1, TropicalWeight& w2)

    cdef TropicalWeight TropicalZero "fst::TropicalWeight::Zero" ()
    cdef TropicalWeight TropicalOne "fst::TropicalWeight::One" ()

    cdef cppclass StdVectorFst(MutableFst):
        void AddArc(int s, StdArc &arc)

    cdef StdVectorFst* StdVectorFstRead "fst::StdVectorFst::Read" (string& filename)

    cdef cppclass ILabelCompare[A]:
        pass

    cdef cppclass OLabelCompare[A]:
        pass

    enum ProjectType:
        PROJECT_INPUT
        PROJECT_OUTPUT

    enum ClosureType:
        CLOSURE_STAR
        CLOSURE_PLUS

    cdef bint Equivalent(Fst& fst1, Fst& fst2)

    # const
    cdef void Compose(Fst &ifst1, Fst &ifst2, MutableFst* ofst)
    cdef void Determinize(Fst& ifst, MutableFst* ofst)
    cdef void Difference(Fst &ifst1, Fst &ifst2, MutableFst* ofst)
    cdef void Intersect(Fst &ifst1, Fst &ifst2, MutableFst* ofst)
    cdef void Reverse(Fst &ifst, MutableFst* ofst)
    cdef void ShortestDistance(Fst &fst, vector[TropicalWeight]* distance, bint reverse)
    cdef void ShortestPath(Fst &ifst, MutableFst* ofst, unsigned n)
    # non const
    cdef void ArcSort(MutableFst* fst, ILabelCompare[StdArc]& compare)
    cdef void ArcSort(MutableFst* fst, OLabelCompare[StdArc]& compare)
    cdef void Closure(MutableFst* ifst, ClosureType type)
    cdef void Invert(MutableFst* ifst)
    cdef void Minimize(MutableFst* fst)
    cdef void Project(MutableFst* fst, ProjectType type)
    cdef void Prune(MutableFst* ifst, TropicalWeight threshold)
    cdef void Relabel(MutableFst* fst, 
            vector[pair[int, int]]& ipairs,
            vector[pair[int, int]]& opairs)
    cdef void RmEpsilon(MutableFst* fst)
    cdef void TopSort(MutableFst* fst)
    # other
    cdef void Union(MutableFst* ifst1, Fst &ifst2)
    cdef void Concat(MutableFst* ifst1, Fst &ifst2)
