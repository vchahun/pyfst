cimport cfst
cimport sym
import subprocess

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t
from util cimport ifstream, ostringstream

EPSILON = 0

cdef bytes as_str(data):
    if isinstance(data, bytes):
        return data
    elif isinstance(data, unicode):
        return data.encode('utf8')
    raise TypeError('Cannot convert %s to string' % type(data))

def read(filename):
    """read(filename): read a transducer from the binary filename & detect arc type"""
    filename = as_str(filename)
    cdef ifstream* stream = new ifstream(filename)
    cdef cfst.FstHeader* header = new cfst.FstHeader()
    header.Read(stream[0], filename)
    cdef bytes arc_type = header.ArcType()
    del stream, header
    if arc_type == 'standard':
        return read_std(filename)
    elif arc_type == 'log':
        return read_log(filename)
    raise TypeError('cannot read transducer with arcs of type {0}'.format(arc_type))

def read_std(filename):
    """read_std(filename): read a StdVectorFst from the binary filename"""
    cdef StdVectorFst fst = StdVectorFst.__new__(StdVectorFst)
    fst.fst = cfst.StdVectorFstRead(as_str(filename))
    fst._init_tables()
    return fst

def read_log(filename):
    """read_log(filename): read a LogVectorFst from the binary filename"""
    cdef LogVectorFst fst = LogVectorFst.__new__(LogVectorFst)
    fst.fst = cfst.LogVectorFstRead(as_str(filename))
    fst._init_tables()
    return fst

def read_symbols(filename):
    """read_symbols(filename): read a binaray symbol table"""
    filename = as_str(filename)
    cdef ifstream* fstream = new ifstream(filename)
    cdef SymbolTable table = SymbolTable.__new__(SymbolTable)
    cdef sym.SymbolTable* syms = sym.SymbolTableRead(fstream[0], filename)
    table.table = new sym.SymbolTable(syms[0])
    del syms, fstream
    return table

cdef class SymbolTable:
    """SymbolTable() -> empty symbol table"""
    cdef sym.SymbolTable* table

    def __init__(self):
        cdef bytes name = 'SymbolTable<{0}>'.format(id(self))
        self.table = new sym.SymbolTable(<string> name)
        self.table.AddSymbol('<eps>')

    def __dealloc__(self):
        del self.table

    def copy(self):
        cdef SymbolTable result = SymbolTable.__new__(SymbolTable)
        result.table = new sym.SymbolTable(self.table[0])
        return result

    def __getitem__(self, sym):
        return self.table.AddSymbol(as_str(sym))

    def __setitem__(self, sym, long key):
        self.table.AddSymbol(as_str(sym), key)

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

    def items(self):
        cdef sym.SymbolTableIterator* it = new sym.SymbolTableIterator(self.table[0])
        while not it.Done():
            yield (it.Symbol(), it.Value())
            it.Next()

    def __richcmp__(SymbolTable x, SymbolTable y, int op):
        if op == 2: # ==
            return x.table.CheckSum() == y.table.CheckSum()
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for SymbolTable')

    def __str__(self):
        return '<SymbolTable of size %d>' % len(self)

cdef class Fst:
    def __init__(self):
        raise NotImplemented('use StdVectorFst or LogVectorFst to create a transducer')

    def _repr_svg_(self):
        """IPython magic: show SVG reprensentation of the transducer"""
        try:
            process = subprocess.Popen(['dot', '-Tsvg'], 
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            raise Exception('cannot find the dot binary')
        out, err = process.communicate(self.draw())
        if err:
            raise Exception(err)
        return out


cdef class TropicalWeight:
    """A weight on the tropical semiring"""
    cdef cfst.TropicalWeight* weight

    ZERO = TropicalWeight(cfst.TropicalWeightZero().Value())
    ONE = TropicalWeight(cfst.TropicalWeightOne().Value())

    def __init__(self, value):
        self.weight = new cfst.TropicalWeight((cfst.TropicalWeightOne() if value is True or value is None
                        else cfst.TropicalWeightZero() if value is False
                        else cfst.TropicalWeight(float(value))))

    def __dealloc__(self):
        del self.weight

    def __float__(self):
        return self.weight.Value()

    def __int__(self):
        return int(self.weight.Value())

    def __bool__(self):
        return (self.weight[0] == cfst.TropicalWeightOne())

    def __str__(self):
        return 'TropicalWeight({0})'.format(float(self))

    def __richcmp__(TropicalWeight x, TropicalWeight y, int op):
        if op == 2: # ==
            return x.weight[0] == y.weight[0]
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for TropicalWeight')

    def __add__(TropicalWeight x, TropicalWeight y):
        cdef TropicalWeight result = TropicalWeight.__new__(TropicalWeight)
        result.weight = new cfst.TropicalWeight(cfst.Plus(x.weight[0], y.weight[0]))
        return result

    def __mul__(TropicalWeight x, TropicalWeight y):
        cdef TropicalWeight result = TropicalWeight.__new__(TropicalWeight)
        result.weight = new cfst.TropicalWeight(cfst.Times(x.weight[0], y.weight[0]))
        return result

    def __iadd__(self, TropicalWeight other):
        del self.weight
        self.weight = new cfst.TropicalWeight(cfst.Plus(self.weight[0], other.weight[0]))

    def __imul__(self, TropicalWeight other):
        del self.weight
        self.weight = new cfst.TropicalWeight(cfst.Times(self.weight[0], other.weight[0]))

cdef class StdArc:
    """A transducer arc"""
    cdef cfst.StdArc* arc

    def __init__(self):
        raise NotImplemented('cannot create independent arc')

    property ilabel:
        def __get__(self):
            return self.arc.ilabel

    property olabel:
        def __get__(self):
            return self.arc.olabel

    property nextstate:
        def __get__(self):
            return self.arc.nextstate

    property weight:
        def __get__(self):
            cdef TropicalWeight weight = TropicalWeight.__new__(TropicalWeight)
            weight.weight = new cfst.TropicalWeight(self.arc.weight)
            return weight

cdef class StdState:
    """A transducer state"""
    cdef public int stateid
    cdef cfst.StdVectorFst* fst

    def __init__(self):
        raise NotImplemented('cannot create independent state')

    def __len__(self):
        return self.fst.NumArcs(self.stateid)

    def __iter__(self):
        cdef cfst.ArcIterator[cfst.StdVectorFst]* it
        it = new cfst.ArcIterator[cfst.StdVectorFst](self.fst[0], self.stateid)
        cdef StdArc arc
        try:
            while not it.Done():
                arc = StdArc.__new__(StdArc)
                arc.arc = <cfst.StdArc*> &it.Value()
                yield arc
                it.Next()
        finally:
            del it

    property arcs:
        def __get__(self):
            return iter(self)

    property final:
        def __get__(self):
            cdef TropicalWeight weight = TropicalWeight.__new__(TropicalWeight)
            weight.weight = new cfst.TropicalWeight(self.fst.Final(self.stateid))
            return weight

        def __set__(self, weight):
            if not isinstance(weight, TropicalWeight):
                weight = TropicalWeight(weight)
            self.fst.SetFinal(self.stateid, (<TropicalWeight> weight).weight[0])

    property initial:
        def __get__(self):
            return self.stateid == self.fst.Start()

        def __set__(self, v):
            if v:
                self.fst.SetStart(self.stateid)
            elif self.stateid == self.fst.Start():
                self.fst.SetStart(-1)

cdef class StdVectorFst(Fst):
    """StdVectorFst() -> empty finite-state transducer"""
    cdef cfst.StdVectorFst* fst
    cdef public SymbolTable isyms, osyms

    def __init__(self, source=None, isyms=None, osyms=None):
        if isinstance(source, StdVectorFst):
            self.fst = <cfst.StdVectorFst*> self.fst.Copy()
        else:
            self.fst = new cfst.StdVectorFst()
            if isinstance(source, LogVectorFst):
                cfst.ArcMap((<LogVectorFst> source).fst[0], self.fst,
                    cfst.LogToStdWeightConvertMapper())
        if isyms is not None:
            self.isyms = isyms.copy()
        if osyms is not None:
            self.osyms = (self.isyms if (isyms is osyms) else osyms.copy())

    def __dealloc__(self):
        del self.fst

    def _init_tables(self):
        if self.fst.MutableInputSymbols() != NULL:
            self.isyms = SymbolTable.__new__(SymbolTable)
            self.isyms.table = new sym.SymbolTable(self.fst.MutableInputSymbols()[0])
            self.fst.SetInputSymbols(NULL)
        if self.fst.MutableOutputSymbols() != NULL:
            self.osyms = SymbolTable.__new__(SymbolTable)
            self.osyms.table = new sym.SymbolTable(self.fst.MutableOutputSymbols()[0])
            self.fst.SetOutputSymbols(NULL)

    def __len__(self):
        return self.fst.NumStates()

    def __str__(self):
        return '<StdVectorFst with %d states>' % len(self)

    def copy(self):
        """fst.copy() -> a copy of the transducer"""
        cdef StdVectorFst result = StdVectorFst.__new__(StdVectorFst)
        if self.isyms is not None:
            result.isyms = self.isyms.copy()
        if self.osyms is not None:
            result.osyms = (result.isyms if (self.isyms is self.osyms) else self.osyms.copy())
        result.fst = <cfst.StdVectorFst*> self.fst.Copy()
        return result

    def __getitem__(self, int stateid):
        if not (0 <= stateid < len(self)):
            raise KeyError('state index out of range')
        cdef StdState state = StdState.__new__(StdState)
        state.stateid = stateid
        state.fst = self.fst
        return state

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    property states:
        def __get__(self):
            return iter(self)

    property start:
        def __get__(self):
            return self.fst.Start()
        
        def __set__(self, int start):
            self.fst.SetStart(start)

    def add_arc(self, int source, int dest, int ilabel, int olabel, weight=None):
        """fst.add_arc(int source, int dest, int ilabel, int olabel, weight=None):
        add an arc source->dest labeled with labels ilabel:olabel and weighted with weight"""
        if source > self.fst.NumStates()-1:
            raise ValueError('invalid source state id ({0} > {0})'.format(source, self.fst.NumStates()-1))
        if not isinstance(weight, TropicalWeight):
            weight = TropicalWeight(weight)
        cdef cfst.StdArc* arc = new cfst.StdArc(ilabel, olabel, (<TropicalWeight> weight).weight[0], dest)
        self.fst.AddArc(source, arc[0])
        del arc

    def add_state(self):
        """fst.add_state() -> new state"""
        return self.fst.AddState()

    def __richcmp__(StdVectorFst x, StdVectorFst y, int op):
        if op == 2: # ==
            return cfst.Equivalent(x.fst[0], y.fst[0])
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for StdVectorFst')

    def write(self, filename):
        """fst.write(filename): write the binary representation of the transducer in filename"""
        if self.isyms:
            self.fst.SetInputSymbols(self.isyms.table)
        if self.osyms:
            self.fst.SetOutputSymbols(self.osyms.table)
        result = self.fst.Write(as_str(filename))
        self.fst.SetInputSymbols(NULL)
        self.fst.SetOutputSymbols(NULL)
        return bool(result)

    property deterministic:
        def __get__(self):
            return (self.fst.Properties(cfst.kIDeterministic, True) & cfst.kIDeterministic)

    def determinize(self):
        """fst.determinize() -> determinized transducer"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.Determinize(self.fst[0], result.fst)
        return result

    def compose(self, StdVectorFst other):
        """fst.compose(StdVectorFst other) == fst >> other -> composed transducer"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=other.osyms)
        cfst.Compose(self.fst[0], other.fst[0], result.fst)
        return result

    def __rshift__(StdVectorFst x, StdVectorFst y):
        return x.compose(y)

    def intersect(self, StdVectorFst other):
        """fst.intersect(StdVectorFst other) == fst & other -> intersection of the two transducers"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.Intersect(self.fst[0], other.fst[0], result.fst)
        return result

    def __and__(StdVectorFst x, StdVectorFst y):
        return x.intersect(y)

    def set_union(self, StdVectorFst other):
        """fst.set_union(StdVectorFst other): modify to the union of the two transducers"""
        cfst.Union(self.fst, other.fst[0])

    def union(self, StdVectorFst other):
        """fst.union(StdVectorFst other) == fst | other -> union of the two transducers"""
        cdef StdVectorFst result = self.copy()
        result.set_union(other)
        return result

    def __or__(StdVectorFst x, StdVectorFst y):
        return x.union(y)

    def concatenate(self, StdVectorFst other):
        """fst.concatenate(StdVectorFst other): modify to the concatenation of the two transducers"""
        cfst.Concat(self.fst, other.fst[0])

    def concatenation(self, StdVectorFst other):
        """fst.concatenation(StdVectorFst other) == fst + other -> concatenation of the two transducers"""
        cdef StdVectorFst result = self.copy()
        result.concatenate(other)
        return result

    def __add__(StdVectorFst x, StdVectorFst y):
        return x.concatenation(y)

    def difference(self, StdVectorFst other):
        """fst.difference(StdVectorFst other) == fst - other -> difference of the two transducers"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.Difference(self.fst[0], other.fst[0], result.fst)
        return result

    def __sub__(StdVectorFst x, StdVectorFst y):
        return x.difference(y)

    def set_closure(self):
        """fst.set_closure(): modify to the Kleene closure of the transducer"""
        cfst.Closure(self.fst, cfst.CLOSURE_STAR)

    def closure(self):
        """fst.closure() -> Kleene closure of the transducer"""
        cdef StdVectorFst result = self.copy()
        result.set_closure()
        return result

    def invert(self):
        """fst.invert(): modify to the inverse of the transducer"""
        cfst.Invert(self.fst)
    
    def inverse(self):
        """fst.inverse() -> inverse of the transducer"""
        cdef StdVectorFst result = self.copy()
        result.invert()
        return result

    def reverse(self):
        """fst.reverse() -> reversed transducer"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.osyms, osyms=self.isyms)
        cfst.Reverse(self.fst[0], result.fst)
        return result

    def shortest_distance(self, bint reverse=False):
        """fst.shortest_distance(bool reverse=False) -> length of the shortest path"""
        cdef vector[cfst.TropicalWeight]* distances = new vector[cfst.TropicalWeight]()
        cfst.ShortestDistance(self.fst[0], distances, reverse)
        cdef list dist = []
        cdef unsigned i
        for i in range(distances.size()):
            dist.append(TropicalWeight(distances[0][i].Value()))
        del distances
        return dist

    def shortest_path(self, unsigned n=1):
        """fst.shortest_path(int n=1) -> transducer containing the n shortest paths"""
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.ShortestPath(self.fst[0], result.fst, n)
        return result

    def minimize(self):
        """fst.minimize(): minimize the transducer"""
        if not self.deterministic:
            raise ValueError('transducer is not deterministic')
        cfst.Minimize(self.fst)

    def arc_sort_input(self):
        """fst.arc_sort_input(): sort the input arcs of the transducer"""
        cdef cfst.ILabelCompare[cfst.StdArc]* icomp = new cfst.ILabelCompare[cfst.StdArc]()
        cfst.ArcSort(self.fst, icomp[0])
        del icomp

    def arc_sort_output(self):
        """fst.arc_sort_output(): sort the output arcs of the transducer"""
        cdef cfst.OLabelCompare[cfst.StdArc]* ocomp = new cfst.OLabelCompare[cfst.StdArc]()
        cfst.ArcSort(self.fst, ocomp[0])
        del ocomp

    def top_sort(self):
        """fst.top_sort(): topologically sort the nodes of the transducer"""
        cfst.TopSort(self.fst)

    def project_input(self):
        """fst.project_input(): project the transducer on the input side"""
        cfst.Project(self.fst, cfst.PROJECT_INPUT)

    def project_output(self):
        """fst.project_output(): project the transducer on the output side"""
        cfst.Project(self.fst, cfst.PROJECT_OUTPUT)

    def remove_epsilon(self):
        """fst.remove_epsilon(): remove the epsilon transitions from the transducer"""
        cfst.RmEpsilon(self.fst)

    def relabel(self, ipairs=[], opairs=[]):
        """fst.relabel(ipairs=[], opairs=[]): relabel the symbols on the arcs of the transducer"""
        cdef vector[pair[int, int]]* ip = new vector[pair[int, int]]()
        cdef vector[pair[int, int]]* op = new vector[pair[int, int]]()
        for old, new in ipairs:
            ip.push_back(pair[int, int](old, new))
        for old, new in opairs:
            op.push_back(pair[int, int](old, new))
        cfst.Relabel(self.fst, ip[0], op[0])
        del ip, op

    def prune(self, threshold):
        """fst.prune(threshold): prune the transducer"""
        if not isinstance(threshold, TropicalWeight):
            threshold = TropicalWeight(threshold)
        cfst.Prune(self.fst, (<TropicalWeight> threshold).weight[0])

    def plus_map(self, value):
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        if not isinstance(value, TropicalWeight):
            value = TropicalWeight(value)
        cfst.ArcMap(self.fst[0], result.fst,
            cfst.PlusStdArcMapper((<TropicalWeight> value).weight[0]))
        return result

    def times_map(self, value):
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        if not isinstance(value, TropicalWeight):
            value = TropicalWeight(value)
        cfst.ArcMap(self.fst[0], result.fst,
            cfst.TimesStdArcMapper((<TropicalWeight> value).weight[0]))
        return result

    def remove_weights(self):
        cdef StdVectorFst result = StdVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.ArcMap(self.fst[0], result.fst, cfst.RmTropicalWeightMapper())
        return result

    def draw(self, SymbolTable isyms=None,
            SymbolTable osyms=None,
            SymbolTable ssyms=None):
        """fst.draw(SymbolTable isyms=None, SymbolTable osyms=None, SymbolTable ssyms=None)
        -> dot format representation of the transducer"""
        cdef ostringstream* out = new ostringstream()
        cdef sym.SymbolTable* isyms_table = (isyms.table if isyms 
                                             else self.isyms.table if self.isyms
                                             else NULL)
        cdef sym.SymbolTable* osyms_table = (osyms.table if osyms
                                             else self.osyms.table if self.osyms
                                             else NULL)
        cdef sym.SymbolTable* ssyms_table = (ssyms.table if ssyms else NULL)
        cdef cfst.FstDrawer[cfst.StdArc]* drawer =\
            new cfst.FstDrawer[cfst.StdArc](self.fst[0],
                isyms_table, osyms_table, ssyms_table,
                False, string(), 8.5, 11, True, False, 0.40, 0.25, 14, 5, False)
        drawer.Draw(out, 'fst')
        cdef bytes out_str = out.str()
        del drawer, out
        return out_str


cdef class LogWeight:
    """A weight on the tropical semiring"""
    cdef cfst.LogWeight* weight

    ZERO = LogWeight(cfst.LogWeightZero().Value())
    ONE = LogWeight(cfst.LogWeightOne().Value())

    def __init__(self, value):
        self.weight = new cfst.LogWeight((cfst.LogWeightOne() if value is True or value is None
                        else cfst.LogWeightZero() if value is False
                        else cfst.LogWeight(float(value))))

    def __dealloc__(self):
        del self.weight

    def __float__(self):
        return self.weight.Value()

    def __int__(self):
        return int(self.weight.Value())

    def __bool__(self):
        return (self.weight[0] == cfst.LogWeightOne())

    def __str__(self):
        return 'LogWeight({0})'.format(float(self))

    def __richcmp__(LogWeight x, LogWeight y, int op):
        if op == 2: # ==
            return x.weight[0] == y.weight[0]
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for LogWeight')

    def __add__(LogWeight x, LogWeight y):
        cdef LogWeight result = LogWeight.__new__(LogWeight)
        result.weight = new cfst.LogWeight(cfst.Plus(x.weight[0], y.weight[0]))
        return result

    def __mul__(LogWeight x, LogWeight y):
        cdef LogWeight result = LogWeight.__new__(LogWeight)
        result.weight = new cfst.LogWeight(cfst.Times(x.weight[0], y.weight[0]))
        return result

    def __iadd__(self, LogWeight other):
        del self.weight
        self.weight = new cfst.LogWeight(cfst.Plus(self.weight[0], other.weight[0]))

    def __imul__(self, LogWeight other):
        del self.weight
        self.weight = new cfst.LogWeight(cfst.Times(self.weight[0], other.weight[0]))

cdef class LogArc:
    """A transducer arc"""
    cdef cfst.LogArc* arc

    def __init__(self):
        raise NotImplemented('cannot create independent arc')

    property ilabel:
        def __get__(self):
            return self.arc.ilabel

    property olabel:
        def __get__(self):
            return self.arc.olabel

    property nextstate:
        def __get__(self):
            return self.arc.nextstate

    property weight:
        def __get__(self):
            cdef LogWeight weight = LogWeight.__new__(LogWeight)
            weight.weight = new cfst.LogWeight(self.arc.weight)
            return weight

cdef class LogState:
    """A transducer state"""
    cdef public int stateid
    cdef cfst.LogVectorFst* fst

    def __init__(self):
        raise NotImplemented('cannot create independent state')

    def __len__(self):
        return self.fst.NumArcs(self.stateid)

    def __iter__(self):
        cdef cfst.ArcIterator[cfst.LogVectorFst]* it
        it = new cfst.ArcIterator[cfst.LogVectorFst](self.fst[0], self.stateid)
        cdef LogArc arc
        try:
            while not it.Done():
                arc = LogArc.__new__(LogArc)
                arc.arc = <cfst.LogArc*> &it.Value()
                yield arc
                it.Next()
        finally:
            del it

    property arcs:
        def __get__(self):
            return iter(self)

    property final:
        def __get__(self):
            cdef LogWeight weight = LogWeight.__new__(LogWeight)
            weight.weight = new cfst.LogWeight(self.fst.Final(self.stateid))
            return weight

        def __set__(self, weight):
            if not isinstance(weight, LogWeight):
                weight = LogWeight(weight)
            self.fst.SetFinal(self.stateid, (<LogWeight> weight).weight[0])

    property initial:
        def __get__(self):
            return self.stateid == self.fst.Start()

        def __set__(self, v):
            if v:
                self.fst.SetStart(self.stateid)
            elif self.stateid == self.fst.Start():
                self.fst.SetStart(-1)

cdef class LogVectorFst(Fst):
    """LogVectorFst() -> empty finite-state transducer"""
    cdef cfst.LogVectorFst* fst
    cdef public SymbolTable isyms, osyms

    def __init__(self, source=None, isyms=None, osyms=None):
        if isinstance(source, LogVectorFst):
            self.fst = <cfst.LogVectorFst*> self.fst.Copy()
        else:
            self.fst = new cfst.LogVectorFst()
            if isinstance(source, StdVectorFst):
                cfst.ArcMap((<StdVectorFst> source).fst[0], self.fst,
                    cfst.StdToLogWeightConvertMapper())
        if isyms is not None:
            self.isyms = isyms.copy()
        if osyms is not None:
            self.osyms = (self.isyms if (isyms is osyms) else osyms.copy())

    def __dealloc__(self):
        del self.fst

    def _init_tables(self):
        if self.fst.MutableInputSymbols() != NULL:
            self.isyms = SymbolTable.__new__(SymbolTable)
            self.isyms.table = new sym.SymbolTable(self.fst.MutableInputSymbols()[0])
            self.fst.SetInputSymbols(NULL)
        if self.fst.MutableOutputSymbols() != NULL:
            self.osyms = SymbolTable.__new__(SymbolTable)
            self.osyms.table = new sym.SymbolTable(self.fst.MutableOutputSymbols()[0])
            self.fst.SetOutputSymbols(NULL)

    def __len__(self):
        return self.fst.NumStates()

    def __str__(self):
        return '<LogVectorFst with %d states>' % len(self)

    def copy(self):
        """fst.copy() -> a copy of the transducer"""
        cdef LogVectorFst result = LogVectorFst.__new__(LogVectorFst)
        if self.isyms is not None:
            result.isyms = self.isyms.copy()
        if self.osyms is not None:
            result.osyms = (result.isyms if (self.isyms is self.osyms) else self.osyms.copy())
        result.fst = <cfst.LogVectorFst*> self.fst.Copy()
        return result

    def __getitem__(self, int stateid):
        if not (0 <= stateid < len(self)):
            raise KeyError('state index out of range')
        cdef LogState state = LogState.__new__(LogState)
        state.stateid = stateid
        state.fst = self.fst
        return state

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    property states:
        def __get__(self):
            return iter(self)

    property start:
        def __get__(self):
            return self.fst.Start()
        
        def __set__(self, int start):
            self.fst.SetStart(start)

    def add_arc(self, int source, int dest, int ilabel, int olabel, weight=None):
        """fst.add_arc(int source, int dest, int ilabel, int olabel, weight=None):
        add an arc source->dest labeled with labels ilabel:olabel and weighted with weight"""
        if source > self.fst.NumStates()-1:
            raise ValueError('invalid source state id ({0} > {0})'.format(source, self.fst.NumStates()-1))
        if not isinstance(weight, LogWeight):
            weight = LogWeight(weight)
        cdef cfst.LogArc* arc = new cfst.LogArc(ilabel, olabel, (<LogWeight> weight).weight[0], dest)
        self.fst.AddArc(source, arc[0])
        del arc

    def add_state(self):
        """fst.add_state() -> new state"""
        return self.fst.AddState()

    def __richcmp__(LogVectorFst x, LogVectorFst y, int op):
        if op == 2: # ==
            return cfst.Equivalent(x.fst[0], y.fst[0])
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for LogVectorFst')

    def write(self, filename):
        """fst.write(filename): write the binary representation of the transducer in filename"""
        if self.isyms:
            self.fst.SetInputSymbols(self.isyms.table)
        if self.osyms:
            self.fst.SetOutputSymbols(self.osyms.table)
        result = self.fst.Write(as_str(filename))
        self.fst.SetInputSymbols(NULL)
        self.fst.SetOutputSymbols(NULL)
        return bool(result)

    property deterministic:
        def __get__(self):
            return (self.fst.Properties(cfst.kIDeterministic, True) & cfst.kIDeterministic)

    def determinize(self):
        """fst.determinize() -> determinized transducer"""
        cdef LogVectorFst result = LogVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.Determinize(self.fst[0], result.fst)
        return result

    def compose(self, LogVectorFst other):
        """fst.compose(LogVectorFst other) == fst >> other -> composed transducer"""
        cdef LogVectorFst result = LogVectorFst(isyms=self.isyms, osyms=other.osyms)
        cfst.Compose(self.fst[0], other.fst[0], result.fst)
        return result

    def __rshift__(LogVectorFst x, LogVectorFst y):
        return x.compose(y)

    def intersect(self, LogVectorFst other):
        """fst.intersect(LogVectorFst other) == fst & other -> intersection of the two transducers"""
        cdef LogVectorFst result = LogVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.Intersect(self.fst[0], other.fst[0], result.fst)
        return result

    def __and__(LogVectorFst x, LogVectorFst y):
        return x.intersect(y)

    def set_union(self, LogVectorFst other):
        """fst.set_union(LogVectorFst other): modify to the union of the two transducers"""
        cfst.Union(self.fst, other.fst[0])

    def union(self, LogVectorFst other):
        """fst.union(LogVectorFst other) == fst | other -> union of the two transducers"""
        cdef LogVectorFst result = self.copy()
        result.set_union(other)
        return result

    def __or__(LogVectorFst x, LogVectorFst y):
        return x.union(y)

    def concatenate(self, LogVectorFst other):
        """fst.concatenate(LogVectorFst other): modify to the concatenation of the two transducers"""
        cfst.Concat(self.fst, other.fst[0])

    def concatenation(self, LogVectorFst other):
        """fst.concatenation(LogVectorFst other) == fst + other -> concatenation of the two transducers"""
        cdef LogVectorFst result = self.copy()
        result.concatenate(other)
        return result

    def __add__(LogVectorFst x, LogVectorFst y):
        return x.concatenation(y)

    def difference(self, LogVectorFst other):
        """fst.difference(LogVectorFst other) == fst - other -> difference of the two transducers"""
        cdef LogVectorFst result = LogVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.Difference(self.fst[0], other.fst[0], result.fst)
        return result

    def __sub__(LogVectorFst x, LogVectorFst y):
        return x.difference(y)

    def set_closure(self):
        """fst.set_closure(): modify to the Kleene closure of the transducer"""
        cfst.Closure(self.fst, cfst.CLOSURE_STAR)

    def closure(self):
        """fst.closure() -> Kleene closure of the transducer"""
        cdef LogVectorFst result = self.copy()
        result.set_closure()
        return result

    def invert(self):
        """fst.invert(): modify to the inverse of the transducer"""
        cfst.Invert(self.fst)
    
    def inverse(self):
        """fst.inverse() -> inverse of the transducer"""
        cdef LogVectorFst result = self.copy()
        result.invert()
        return result

    def reverse(self):
        """fst.reverse() -> reversed transducer"""
        cdef LogVectorFst result = LogVectorFst(isyms=self.osyms, osyms=self.isyms)
        cfst.Reverse(self.fst[0], result.fst)
        return result

    def shortest_distance(self, bint reverse=False):
        """fst.shortest_distance(bool reverse=False) -> length of the shortest path"""
        cdef vector[cfst.LogWeight]* distances = new vector[cfst.LogWeight]()
        cfst.ShortestDistance(self.fst[0], distances, reverse)
        cdef list dist = []
        cdef unsigned i
        for i in range(distances.size()):
            dist.append(LogWeight(distances[0][i].Value()))
        del distances
        return dist

    def shortest_path(self, unsigned n=1):
        """fst.shortest_path(int n=1) -> transducer containing the n shortest paths"""
        cdef LogVectorFst result = LogVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.ShortestPath(self.fst[0], result.fst, n)
        return result

    def minimize(self):
        """fst.minimize(): minimize the transducer"""
        if not self.deterministic:
            raise ValueError('transducer is not deterministic')
        cfst.Minimize(self.fst)

    def arc_sort_input(self):
        """fst.arc_sort_input(): sort the input arcs of the transducer"""
        cdef cfst.ILabelCompare[cfst.LogArc]* icomp = new cfst.ILabelCompare[cfst.LogArc]()
        cfst.ArcSort(self.fst, icomp[0])
        del icomp

    def arc_sort_output(self):
        """fst.arc_sort_output(): sort the output arcs of the transducer"""
        cdef cfst.OLabelCompare[cfst.LogArc]* ocomp = new cfst.OLabelCompare[cfst.LogArc]()
        cfst.ArcSort(self.fst, ocomp[0])
        del ocomp

    def top_sort(self):
        """fst.top_sort(): topologically sort the nodes of the transducer"""
        cfst.TopSort(self.fst)

    def project_input(self):
        """fst.project_input(): project the transducer on the input side"""
        cfst.Project(self.fst, cfst.PROJECT_INPUT)

    def project_output(self):
        """fst.project_output(): project the transducer on the output side"""
        cfst.Project(self.fst, cfst.PROJECT_OUTPUT)

    def remove_epsilon(self):
        """fst.remove_epsilon(): remove the epsilon transitions from the transducer"""
        cfst.RmEpsilon(self.fst)

    def relabel(self, ipairs=[], opairs=[]):
        """fst.relabel(ipairs=[], opairs=[]): relabel the symbols on the arcs of the transducer"""
        cdef vector[pair[int, int]]* ip = new vector[pair[int, int]]()
        cdef vector[pair[int, int]]* op = new vector[pair[int, int]]()
        for old, new in ipairs:
            ip.push_back(pair[int, int](old, new))
        for old, new in opairs:
            op.push_back(pair[int, int](old, new))
        cfst.Relabel(self.fst, ip[0], op[0])
        del ip, op

    def prune(self, threshold):
        """fst.prune(threshold): prune the transducer"""
        if not isinstance(threshold, LogWeight):
            threshold = LogWeight(threshold)
        cfst.Prune(self.fst, (<LogWeight> threshold).weight[0])

    def plus_map(self, value):
        cdef LogVectorFst result = LogVectorFst(isyms=self.isyms, osyms=self.osyms)
        if not isinstance(value, LogWeight):
            value = LogWeight(value)
        cfst.ArcMap(self.fst[0], result.fst,
            cfst.PlusLogArcMapper((<LogWeight> value).weight[0]))
        return result

    def times_map(self, value):
        cdef LogVectorFst result = LogVectorFst(isyms=self.isyms, osyms=self.osyms)
        if not isinstance(value, LogWeight):
            value = LogWeight(value)
        cfst.ArcMap(self.fst[0], result.fst,
            cfst.TimesLogArcMapper((<LogWeight> value).weight[0]))
        return result

    def remove_weights(self):
        cdef LogVectorFst result = LogVectorFst(isyms=self.isyms, osyms=self.osyms)
        cfst.ArcMap(self.fst[0], result.fst, cfst.RmLogWeightMapper())
        return result

    def draw(self, SymbolTable isyms=None,
            SymbolTable osyms=None,
            SymbolTable ssyms=None):
        """fst.draw(SymbolTable isyms=None, SymbolTable osyms=None, SymbolTable ssyms=None)
        -> dot format representation of the transducer"""
        cdef ostringstream* out = new ostringstream()
        cdef sym.SymbolTable* isyms_table = (isyms.table if isyms 
                                             else self.isyms.table if self.isyms
                                             else NULL)
        cdef sym.SymbolTable* osyms_table = (osyms.table if osyms
                                             else self.osyms.table if self.osyms
                                             else NULL)
        cdef sym.SymbolTable* ssyms_table = (ssyms.table if ssyms else NULL)
        cdef cfst.FstDrawer[cfst.LogArc]* drawer =\
            new cfst.FstDrawer[cfst.LogArc](self.fst[0],
                isyms_table, osyms_table, ssyms_table,
                False, string(), 8.5, 11, True, False, 0.40, 0.25, 14, 5, False)
        drawer.Draw(out, 'fst')
        cdef bytes out_str = out.str()
        del drawer, out
        return out_str


cdef class SimpleFst(StdVectorFst):
    """SimpleFst() -> transducer with input/output symbol tables"""
    def __init__(self):
        StdVectorFst.__init__(self)
        self.start = self.add_state()
        self.isyms = SymbolTable()
        self.osyms = SymbolTable()

    def add_arc(self, src, tgt, ilabel, olabel, weight=None):
        """fst.add_arc(int source, int dest, ilabel, olabel, weight=None):
        add an arc source->dest labeled with labels ilabel:olabel and weighted with weight"""
        while max(src, tgt) > len(self)-1:
            self.add_state()
        StdVectorFst.add_arc(self, src, tgt, self.isyms[ilabel], self.osyms[olabel], weight)

cdef class Acceptor(SimpleFst):
    """Acceptor() -> acceptor transducer with a symbol table"""
    def __init__(self):
        StdVectorFst.__init__(self)
        self.start = self.add_state()
        self.isyms = self.osyms = SymbolTable()

    def add_arc(self, src, tgt, label, weight=None):
        """fst.add_arc(int source, int dest, label, weight=None):
        add an arc source->dest labeled with label and weighted with weight"""
        SimpleFst.add_arc(self, src, tgt, label, label, weight)
