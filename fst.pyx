cimport sym
cimport script

ZERO = Weight(float('inf'))
ONE = Weight(0)
EPSILON = 0

def read(char* filename):
    """read(filename): read a transducer from the binary filename"""
    return Fst().set_value(StdVectorFstRead(string(filename)))

def read_symbols(char* filename):
    """read_symbols(filename): read a symbol table"""
    cdef script.ifstream* fstream = new script.ifstream(filename)
    cdef SymbolTable table = SymbolTable()
    table.set_value(sym.SymbolTableRead(fstream[0], string(filename)))
    del fstream
    return table

def det(Fst fst):
    """det(Fst fst) -> determinized transducer"""
    return Fst.__det__(fst)

cdef class Weight:
    """A weight on the tropical semiring"""
    cdef TropicalWeight* weight

    def __cinit__(self, float value=0):
        self.weight = new TropicalWeight(value)

    cdef Weight set_value(self, TropicalWeight* weight):
        self.weight.set_value(weight[0])
        del weight
        return self

    def __dealloc__(self):
        del self.weight

    def __float__(self):
        return self.weight.Value()

    def __int__(self):
        return int(self.weight.Value())

    def __str__(self):
        return str(float(self))

    def __richcmp__(Weight x, Weight y, int op):
        if op == 2: # ==
            return x.weight[0] == y.weight[0]
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for Weight')

    def __add__(Weight x, Weight y):
        return Weight().set_value(new TropicalWeight(Plus(x.weight[0], y.weight[0])))

    def __mul__(Weight x, Weight y):
        return Weight().set_value(new TropicalWeight(Times(x.weight[0], y.weight[0])))

    def __iadd__(self, Weight other):
        self.set_value(new TropicalWeight(Plus(self.weight[0], other.weight[0])))

    def __imul__(self, Weight other):
        self.set_value(new TropicalWeight(Times(self.weight[0], other.weight[0])))

cdef class Arc:
    """A transducer arc"""
    cdef StdArc* arc

    property ilabel:
        def __get__(self):
            return self.arc.ilabel

    property olabel:
        def __get__(self):
            return self.arc.olabel

    property weight:
        def __get__(self):
            return Weight(self.arc.weight.Value())

    property nextstate:
        def __get__(self):
            return self.arc.nextstate

cdef class State:
    """A transducer state"""
    cdef public int stateid
    cdef StdVectorFst* fst

    def __len__(self):
        return self.fst.NumArcs(self.stateid)

    def __iter__(self):
        cdef ArcIterator* it = new ArcIterator(self.fst[0], self.stateid)
        cdef Arc arc
        try:
            while not it.Done():
                arc = Arc()
                arc.arc = &it.Value()
                yield arc
                it.Next()
        finally:
            del it

    property weight:
        def __get__(self):
            return Weight(self.fst.Final(self.stateid).Value())

    property final:
        def __get__(self):
            return self.weight != ZERO

cdef class Fst:
    """Fst() -> empty finite-state transducer"""
    cdef StdVectorFst* fst

    def __cinit__(self):
        self.fst = new StdVectorFst()

    def __dealloc__(self):
        del self.fst

    cdef Fst set_value(self, StdVectorFst* value):
        del self.fst
        self.fst = value
        return self

    def __len__(self):
        return self.fst.NumStates()

    def __str__(self):
        return '<Fst with %d states>' % len(self)

    def copy(self):
        """fst.copy() -> a copy of the transducer"""
        return Fst().set_value(self.fst.Copy())

    def __getitem__(self, int stateid):
        if not (0 <= stateid < len(self)):
            raise KeyError('state index out of range')
        cdef State state = State()
        state.stateid = stateid
        state.fst = self.fst
        return state

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    property start:
        def __get__(self):
            return self.fst.Start()
        
        def __set__(self, int start):
            self.fst.SetStart(start)

    def add_arc(self, int source, int dest, int ilabel, int olabel, float weight=0):
        """fst.add_arc(int source, int dest, int ilabel, int olabel, float weight=0):
        add an arc source->dest with labels ilabel:olabel weighted with weight"""
        cdef StdArc* arc = new StdArc(ilabel, olabel, TropicalWeight(weight), dest)
        self.fst.AddArc(source, arc[0])
        del arc

    def add_state(self):
        """fst.add_state() -> new state"""
        return self.fst.AddState()

    def set_final(self, int final, float weight=0):
        """fst.set_final(int final, float weight=0): select a final state"""
        self.fst.SetFinal(final, TropicalWeight(weight))

    property isyms:
        def __get__(self):
            if self.fst.MutableInputSymbols() == NULL: return None
            cdef SymbolTable isyms = SymbolTable()
            isyms.table = self.fst.MutableInputSymbols()
            return isyms

        def __set__(self, SymbolTable isyms):
            self.fst.SetInputSymbols(isyms.table)

    property osyms:
        def __get__(self):
            if self.fst.MutableOutputSymbols() == NULL: return None
            cdef SymbolTable osyms = SymbolTable()
            osyms.table = self.fst.MutableOutputSymbols()
            return osyms

        def __set__(self, SymbolTable osyms):
            self.fst.SetOutputSymbols(osyms.table)

    def write(self, char* filename):
        """fst.write(str filename): write the binary representation of the transducer in filename"""
        return self.fst.Write(string(filename))

    def __det__(Fst ifst):
        cdef Fst ofst = Fst()
        Determinize(ifst.fst[0], ofst.fst)
        return ofst

    def __rshift__(Fst x, Fst y):
        cdef Fst ofst = Fst()
        Compose(x.fst[0], y.fst[0], ofst.fst)
        return ofst

    def compose(self, Fst other):
        """fst.compose(Fst other) == fst >> other -> composed fst"""
        return (self >> other)

    def shortest_distance(self, bint reverse=False):
        """fst.shortest_distance(bool reverse=False) -> length of the shortest path"""
        cdef vector[TropicalWeight]* distances = new vector[TropicalWeight]()
        ShortestDistance(self.fst[0], distances, reverse)
        cdef list dist = []
        cdef unsigned i
        for i in range(distances.size()):
            dist.append(Weight(distances[0][i].Value()))
        del distances
        return dist

    def shortest_path(self, unsigned n=1):
        """fst.shortest_path(int n=1) -> transducer containing the n shortest paths"""
        cdef Fst ofst = Fst()
        ShortestPath(self.fst[0], ofst.fst, n)
        return ofst

    def minimize(self):
        """fst.minimize(): minimize the transducer"""
        Minimize(self.fst)

    def arc_sort_input(self):
        """fst.arc_sort_input(): sort the input arcs of the transducer"""
        cdef ILabelCompare* icomp = new ILabelCompare()
        ArcSort(self.fst, icomp[0])
        del icomp

    def arc_sort_output(self):
        """fst.arc_sort_output(): sort the output arcs of the transducer"""
        cdef OLabelCompare* ocomp = new OLabelCompare()
        ArcSort(self.fst, ocomp[0])
        del ocomp

    def top_sort(self):
        """fst.top_sort(): topologically sort the nodes of the transducer"""
        TopSort(self.fst)

    def project_input(self):
        """fst.project_input(): project the transducer on the input side"""
        Project(self.fst, PROJECT_INPUT)

    def project_output(self):
        """fst.project_output(): project the transducer on the output side"""
        Project(self.fst, PROJECT_OUTPUT)

    def remove_epsilon(self):
        """fst.remove_epsilon(): remove the epsilon transitions from the transducer"""
        RmEpsilon(self.fst)

    def relabel(self, ipairs=[], opairs=[]):
        """fst.relabel(ipairs=[], opairs=[]): relabel the symbols on the arcs of the transducer"""
        cdef vector[pair[int, int]]* ip = new vector[pair[int, int]]()
        cdef vector[pair[int, int]]* op = new vector[pair[int, int]]()
        for old, new in ipairs:
            ip.push_back(pair[int, int](old, new))
        for old, new in opairs:
            op.push_back(pair[int, int](old, new))
        Relabel(self.fst, ip[0], op[0])
        del ip, op

    def draw(self, SymbolTable isyms=None,
            SymbolTable osyms=None,
            SymbolTable ssyms=None):
        """fst.draw(SymbolTable isyms=None, SymbolTable osyms=None, SymbolTable ssyms=None)
        -> dot format representation of the transducer"""
        cdef script.ostringstream* out = new script.ostringstream()
        cdef script.FstDrawer* drawer = new script.FstDrawer(self.fst[0],
                (isyms.table if isyms else NULL),
                (osyms.table if osyms else NULL),
                (ssyms.table if ssyms else NULL),
                False, string(), 8.5, 11, True, False, 0.40, 0.25, 14, 5, False)
        drawer.Draw(out, 'fst')
        cdef bytes out_str = out.str().c_str()
        del drawer, out
        return out_str

cdef class SymbolTable:
    """SymbolTable() -> empty symbol table"""
    cdef sym.SymbolTable* table

    def __cinit__(self):
        cdef bytes name = bytes('SymbolTable<%d>' % id(self))
        self.table = new sym.SymbolTable(string(name))
        self.table.AddSymbol('<eps>')

    cdef SymbolTable set_value(self, sym.SymbolTable* table):
        del self.table
        self.table = table
        return self

    def __dealloc__(self):
        del self.table

    def __getitem__(self, char* sym):
        return self.table.AddSymbol(string(sym))

    def write(self, char* filename):
        """write(filename): save the symbol table to filename"""
        self.table.Write(string(filename))

    def find(self, long key):
        """find(int key) -> decoded symbol"""
        if not 0 <= key < len(self):
            raise KeyError('symbol table index out of range')
        return self.table.Find(key).c_str()

    def __len__(self):
        return self.table.NumSymbols()

    def __iter__(self):
        cdef unsigned i
        for i in range(len(self)):
            yield self.find(i)

    def __str__(self):
        return '<SymbolTable of size %d>' % len(self)
