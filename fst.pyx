cimport sym
cimport script

ZERO = Weight(float('inf'))
ONE = Weight(0)
EPSILON = 0

cdef bytes as_str(data):
    if isinstance(data, bytes):
        return data
    elif isinstance(data, unicode):
        return data.encode('utf8')
    raise TypeError('Cannot convert %s to string' % type(data))

def read(filename):
    """read(filename): read a transducer from the binary filename"""
    cdef Fst fst = Fst.__new__(Fst)
    fst.fst = StdVectorFstRead(as_str(filename))
    return fst

def read_symbols(filename):
    """read_symbols(filename): read a symbol table"""
    filename = as_str(filename)
    cdef script.ifstream* fstream = new script.ifstream(filename)
    cdef SymbolTable table = SymbolTable.__new__(SymbolTable)
    table.table = sym.SymbolTableRead(fstream[0], filename)
    del fstream
    return table

cdef class Weight:
    """A weight on the tropical semiring"""
    cdef TropicalWeight* weight

    def __init__(self, float value=0):
        self.weight = new TropicalWeight(value)

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
        cdef Weight result = Weight.__new__(Weight)
        result.weight = new TropicalWeight(Plus(x.weight[0], y.weight[0]))
        return result

    def __mul__(Weight x, Weight y):
        cdef Weight result = Weight.__new__(Weight)
        result.weight = new TropicalWeight(Times(x.weight[0], y.weight[0]))
        return result

    def __iadd__(self, Weight other):
        del self.weight
        self.weight = new TropicalWeight(Plus(self.weight[0], other.weight[0]))

    def __imul__(self, Weight other):
        del self.weight
        self.weight = new TropicalWeight(Times(self.weight[0], other.weight[0]))

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
            cdef Weight weight = Weight.__new__(Weight)
            weight.weight = new TropicalWeight(self.arc.weight)
            return weight

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
            cdef Weight weight = Weight.__new__(Weight)
            weight.weight = new TropicalWeight(self.fst.Final(self.stateid))
            return weight

    property final:
        def __get__(self):
            return self.weight != ZERO

cdef class Fst:
    """Fst() -> empty finite-state transducer"""
    cdef StdVectorFst* fst

    def __init__(self):
        self.fst = new StdVectorFst()

    def __dealloc__(self):
        del self.fst

    def __len__(self):
        return self.fst.NumStates()

    def __str__(self):
        return '<Fst with %d states>' % len(self)

    def copy(self):
        """fst.copy() -> a copy of the transducer"""
        cdef Fst fst = Fst.__new__(Fst)
        fst.fst = self.fst.Copy()
        return fst

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
        if source > self.fst.NumStates()-1:
            raise ValueError('invalid source state id ({0} > {0})'.format(source, self.fst.NumStates()-1))
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
            cdef RefSymbolTable isyms = RefSymbolTable.__new__(RefSymbolTable)
            isyms.table = self.fst.MutableInputSymbols()
            return isyms

        def __set__(self, RefSymbolTable isyms):
            self.fst.SetInputSymbols(isyms.table)

    property osyms:
        def __get__(self):
            if self.fst.MutableOutputSymbols() == NULL: return None
            cdef RefSymbolTable osyms = RefSymbolTable.__new__(RefSymbolTable)
            osyms.table = self.fst.MutableOutputSymbols()
            return osyms

        def __set__(self, RefSymbolTable osyms):
            self.fst.SetOutputSymbols(osyms.table)

    def __richcmp__(Fst x, Fst y, int op):
        if op == 2: # ==
            return Equivalent(x.fst[0], y.fst[0])
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for Fst')

    def write(self, filename):
        """fst.write(filename): write the binary representation of the transducer in filename"""
        return self.fst.Write(as_str(filename))

    def determinize(self):
        """fst.determinize() -> determinized transducer"""
        cdef Fst ofst = Fst()
        Determinize(self.fst[0], ofst.fst)
        return ofst

    def compose(self, Fst other):
        """fst.compose(Fst other) == fst >> other -> composed transducer"""
        cdef Fst result = Fst()
        Compose(self.fst[0], other.fst[0], result.fst)
        return result

    def __rshift__(Fst x, Fst y):
        return x.compose(y)

    def intersect(self, Fst other):
        """fst.intersect(Fst other) == fst & other -> intersection of the two transducers"""
        cdef Fst result = Fst()
        Intersect(self.fst[0], other.fst[0], result.fst)
        return result

    def __and__(Fst x, Fst y):
        return x.intersect(y)

    def set_union(self, Fst other):
        """fst.set_union(Fst other): modify to the union of the two transducers"""
        Union(self.fst, other.fst[0])

    def union(self, Fst other):
        """fst.union(Fst other) == fst | other -> union of the two transducers"""
        cdef Fst result = self.copy()
        result.set_union(other)
        return result

    def __or__(Fst x, Fst y):
        return x.union(y)

    def concatenate(self, Fst other):
        """fst.concatenate(Fst other): modify to the concatenation of the two transducers"""
        Concat(self.fst, other.fst[0])

    def concatenation(self, Fst other):
        """fst.concatenation(Fst other) == fst + other -> concatenation of the two transducers"""
        cdef Fst result = self.copy()
        result.concatenate(other)
        return result

    def __add__(Fst x, Fst y):
        return x.concatenation(y)

    def difference(self, Fst other):
        """fst.difference(Fst other) == fst - other -> difference of the two transducers"""
        cdef Fst result = Fst()
        Difference(self.fst[0], other.fst[0], result.fst)
        return result

    def __sub__(Fst x, Fst y):
        return x.difference(y)

    def set_closure(self):
        """fst.set_closure(): modify to the Kleene closure of the transducer"""
        Closure(self.fst, CLOSURE_STAR)

    def closure(self):
        """fst.closure() -> Kleene closure of the transducer"""
        cdef Fst result = self.copy()
        result.set_closure()
        return result

    def invert(self):
        """fst.invert(): modify to the inverse of the transducer"""
        Invert(self.fst)
    
    def inverse(self):
        """fst.inverse() -> inverse of the transducer"""
        cdef Fst result = self.copy()
        result.invert()
        return result

    def reverse(self):
        """fst.reverse() -> reversed transducer"""
        cdef Fst result = Fst()
        Reverse(self.fst[0], result.fst)
        return result

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

    def prune(self, float theshold):
        """fst.prune(float theshold): prune the transducer"""
        Prune(self.fst, TropicalWeight(theshold))

    def draw(self, RefSymbolTable isyms=None,
            RefSymbolTable osyms=None,
            RefSymbolTable ssyms=None):
        """fst.draw(SymbolTable isyms=None, SymbolTable osyms=None, SymbolTable ssyms=None)
        -> dot format representation of the transducer"""
        cdef script.ostringstream* out = new script.ostringstream()
        cdef sym.SymbolTable* isyms_table = self.fst.MutableInputSymbols()
        if isyms is not None:
            isyms_table = isyms.table
        cdef sym.SymbolTable* osyms_table = self.fst.MutableOutputSymbols()
        if osyms is not None:
            osyms_table = osyms.table
        cdef sym.SymbolTable* ssyms_table = NULL
        cdef script.FstDrawer* drawer = new script.FstDrawer(self.fst[0],
                isyms_table, osyms_table, ssyms_table,
                False, string(), 8.5, 11, True, False, 0.40, 0.25, 14, 5, False)
        drawer.Draw(out, 'fst')
        cdef bytes out_str = out.str()
        del drawer, out
        return out_str

cdef class RefSymbolTable:
    cdef sym.SymbolTable* table

    def __getitem__(self, sym):
        return self.table.AddSymbol(as_str(sym))

    def write(self, filename):
        """write(filename): save the symbol table to filename"""
        self.table.Write(as_str(filename))

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

cdef class SymbolTable(RefSymbolTable):
    """SymbolTable() -> empty symbol table"""
    def __init__(self):
        cdef bytes name = bytes('SymbolTable<%d>' % id(self))
        self.table = new sym.SymbolTable(name)
        self.table.AddSymbol('<eps>')

    def __dealloc__(self):
        del self.table
