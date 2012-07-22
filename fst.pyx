cimport sym
cimport script

def read(char* filename):
    cdef BaseFst fst = BaseFst()
    fst.fst = StdVectorFstRead(string(filename))
    return fst

cdef class Weight:
    cdef TropicalWeight* weight

    def __cinit__(self, float value=0):
        self.weight = new TropicalWeight(value)

    def __dealloc__(self):
        del self.weight

    def __float__(self):
        return self.weight.Value()

    def __richcmp__(Weight x, Weight y, int op):
        if op == 2: # ==
            return x.weight[0] == y.weight[0]
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for Weight')

    def __add__(Weight x, Weight y):
        cdef Weight result = Weight()
        result.weight = new TropicalWeight(Plus(x.weight[0], y.weight[0]))
        return result

    def __mul__(Weight x, Weight y):
        cdef Weight result = Weight()
        result.weight = new TropicalWeight(Times(x.weight[0], y.weight[0]))
        return result

    def __iadd__(self, Weight other):
        self.weight = new TropicalWeight(Plus(self.weight[0], other.weight[0]))

    def __imul__(self, Weight other):
        self.weight = new TropicalWeight(Times(self.weight[0], other.weight[0]))

    def __str__(self):
        return str(float(self))

ZERO = Weight(float('inf'))
ONE = Weight(0)

cdef class Arc:
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

cdef class BaseFst:
    cdef StdVectorFst* fst

    def __dealloc__(self):
        del self.fst

    def __len__(self):
        return self.fst.NumStates()

    def __str__(self):
        return '<Fst with %d states>' % len(self)

    def copy(self):
        cdef BaseFst fst = BaseFst()
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
        cdef StdArc* arc = new StdArc(ilabel, olabel, TropicalWeight(weight), dest)
        self.fst.AddArc(source, arc[0])
        del arc

    def add_state(self):
        return self.fst.AddState()

    def set_final(self, int final, float weight=0):
        self.fst.SetFinal(final, TropicalWeight(weight))

    property isyms:
        def __get__(self):
            cdef SymbolTable isyms = SymbolTable()
            isyms.table = self.fst.MutableInputSymbols()
            return isyms

        def __set__(self, SymbolTable isyms):
            self.fst.SetInputSymbols(isyms.table)

    property osyms:
        def __get__(self):
            cdef SymbolTable osyms = SymbolTable()
            osyms.table = self.fst.MutableOutputSymbols()
            return osyms

        def __set__(self, SymbolTable osyms):
            self.fst.SetOutputSymbols(osyms.table)

    def write(self, char* filename):
        return self.fst.Write(string(filename))

    def determinize(self):
        cdef Fst ofst = Fst()
        Determinize(self.fst[0], ofst.fst)
        return ofst

    def __or__(BaseFst x, BaseFst y):
        cdef Fst ofst = Fst()
        Compose(x.fst[0], y.fst[0], ofst.fst)
        return ofst

    def minimize(self):
        Minimize(self.fst)

    def arc_sort_input(self):
        cdef ILabelCompare* icomp = new ILabelCompare()
        ArcSort(self.fst, icomp[0])
        del icomp

    def arc_sort_output(self):
        cdef OLabelCompare* ocomp = new OLabelCompare()
        ArcSort(self.fst, ocomp[0])
        del ocomp

    def project_input(self):
        Project(self.fst, PROJECT_INPUT)

    def project_output(self):
        Project(self.fst, PROJECT_OUTPUT)

    def remove_epsilon(self):
        RmEpsilon(self.fst)

    def relabel(self, ipairs=[], opairs=[]):
        cdef vector[pair[int, int]]* ip = new vector[pair[int, int]]()
        cdef vector[pair[int, int]]* op = new vector[pair[int, int]]()
        for old, new in ipairs:
            ip.push_back(pair[int, int](old, new))
        for old, new in opairs:
            op.push_back(pair[int, int](old, new))
        Relabel(self.fst, ip[0], op[0])
        del ip, op

    def draw(self, char* filename='/dev/stdout', 
            SymbolTable isyms=None,
            SymbolTable osyms=None,
            SymbolTable ssyms=None):
        cdef script.ofstream* out = new script.ofstream(filename)
        cdef script.FstDrawer* drawer = new script.FstDrawer(self.fst[0],
                (isyms.table if isyms else NULL),
                (osyms.table if osyms else NULL),
                (ssyms.table if ssyms else NULL),
                False, string(), 8.5, 11, True, False, 0.40, 0.25, 14, 5, False)
        drawer.Draw(out, string(filename))
        del drawer, out

    def print_text(self, char* filename='/dev/stdout', 
            SymbolTable isyms=None,
            SymbolTable osyms=None,
            SymbolTable ssyms=None):
        cdef script.ofstream* out = new script.ofstream(filename)
        cdef script.FstPrinter* printer = new script.FstPrinter(self.fst[0],
                (isyms.table if isyms else NULL),
                (osyms.table if osyms else NULL),
                (ssyms.table if ssyms else NULL),
                False, False)
        printer.Print(out, string(filename))
        del printer, out

cdef class Fst(BaseFst):
    def __cinit__(self):
        self.fst = new StdVectorFst()

cdef class SymbolTable:
    cdef sym.SymbolTable* table

    def __cinit__(self):
        cdef bytes name = bytes('SymbolTable<%d>' % id(self))
        self.table = new sym.SymbolTable(string(name))
        self.table.AddSymbol(string('<eps>'))

    def __dealloc__(self):
        del self.table

    def __getitem__(self, char* sym):
        return self.table.AddSymbol(string(sym))

    def write(self, char* filename):
        self.table.Write(string(filename))

    def find(self, long key):
        if not 0 <= key < len(self):
            raise KeyError('symbol table index out of range')
        return self.table.Find(key).c_str()

    def __len__(self):
        return self.table.NumSymbols()

    def __iter__(self):
        cdef unsigned i
        for i in range(len(self)):
            yield self.find(i)
