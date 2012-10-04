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
    """read_symbols(filename): read a symbol table"""
    filename = as_str(filename)
    cdef ifstream* fstream = new ifstream(filename)
    cdef SymbolTable table = SymbolTable.__new__(SymbolTable)
    cdef sym.SymbolTable* syms = sym.SymbolTableRead(fstream[0], filename)
    table.table = new sym.SymbolTable(syms[0])
    del syms
    del fstream
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
        # TODO: use SymbolTableIterator?
        cdef unsigned i
        for i in range(len(self)):
            yield self.find(i)

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

{{#types}}

cdef class {{weight}}:
    """A weight on the tropical semiring"""
    cdef cfst.{{weight}}* weight

    ZERO = {{weight}}(cfst.{{weight}}Zero().Value())
    ONE = {{weight}}(cfst.{{weight}}One().Value())

    def __init__(self, value):
        self.weight = new cfst.{{weight}}((cfst.{{weight}}One() if value is True or value is None
                        else cfst.{{weight}}Zero() if value is False
                        else cfst.{{weight}}(float(value))))

    def __dealloc__(self):
        del self.weight

    def __float__(self):
        return self.weight.Value()

    def __int__(self):
        return int(self.weight.Value())

    def __bool__(self):
        return (self.weight[0] == cfst.{{weight}}One())

    def __str__(self):
        return '{{weight}}({0})'.format(float(self))

    def __richcmp__({{weight}} x, {{weight}} y, int op):
        if op == 2: # ==
            return x.weight[0] == y.weight[0]
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for {{weight}}')

    def __add__({{weight}} x, {{weight}} y):
        cdef {{weight}} result = {{weight}}.__new__({{weight}})
        result.weight = new cfst.{{weight}}(cfst.Plus(x.weight[0], y.weight[0]))
        return result

    def __mul__({{weight}} x, {{weight}} y):
        cdef {{weight}} result = {{weight}}.__new__({{weight}})
        result.weight = new cfst.{{weight}}(cfst.Times(x.weight[0], y.weight[0]))
        return result

    def __iadd__(self, {{weight}} other):
        del self.weight
        self.weight = new cfst.{{weight}}(cfst.Plus(self.weight[0], other.weight[0]))

    def __imul__(self, {{weight}} other):
        del self.weight
        self.weight = new cfst.{{weight}}(cfst.Times(self.weight[0], other.weight[0]))

cdef class {{arc}}:
    """A transducer arc"""
    cdef cfst.{{arc}}* arc

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
            cdef {{weight}} weight = {{weight}}.__new__({{weight}})
            weight.weight = new cfst.{{weight}}(self.arc.weight)
            return weight

cdef class {{state}}:
    """A transducer state"""
    cdef public int stateid
    cdef cfst.{{fst}}* fst

    def __init__(self):
        raise NotImplemented('cannot create independent state')

    def __len__(self):
        return self.fst.NumArcs(self.stateid)

    def __iter__(self):
        cdef cfst.ArcIterator[cfst.{{fst}}]* it
        it = new cfst.ArcIterator[cfst.{{fst}}](self.fst[0], self.stateid)
        cdef {{arc}} arc
        try:
            while not it.Done():
                arc = {{arc}}.__new__({{arc}})
                arc.arc = <cfst.{{arc}}*> &it.Value()
                yield arc
                it.Next()
        finally:
            del it

    property final:
        def __get__(self):
            cdef {{weight}} weight = {{weight}}.__new__({{weight}})
            weight.weight = new cfst.{{weight}}(self.fst.Final(self.stateid))
            return weight

        def __set__(self, weight):
            if not isinstance(weight, {{weight}}):
                weight = {{weight}}(weight)
            self.fst.SetFinal(self.stateid, (<{{weight}}> weight).weight[0])

    property initial:
        def __get__(self):
            return self.stateid == self.fst.Start()

        def __set__(self, v):
            if v:
                self.fst.SetStart(self.stateid)
            elif self.stateid == self.fst.Start():
                self.fst.SetStart(-1)

cdef class {{fst}}(Fst):
    """{{fst}}() -> empty finite-state transducer"""
    cdef cfst.{{fst}}* fst
    cdef public SymbolTable isyms, osyms

    def __init__(self, source=None, isyms=None, osyms=None):
        if isinstance(source, {{fst}}):
            self.fst = <cfst.{{fst}}*> self.fst.Copy()
        else:
            self.fst = new cfst.{{fst}}()
            if isinstance(source, {{other}}VectorFst):
                cfst.ArcMap((<{{other}}VectorFst> source).fst[0], self.fst,
                    cfst.{{convert}}WeightConvertMapper())
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
        return '<{{fst}} with %d states>' % len(self)

    def copy(self):
        """fst.copy() -> a copy of the transducer"""
        cdef {{fst}} result = {{fst}}.__new__({{fst}})
        if self.isyms is not None:
            result.isyms = self.isyms.copy()
        if self.osyms is not None:
            result.osyms = (result.isyms if (self.isyms is self.osyms) else self.osyms.copy())
        result.fst = <cfst.{{fst}}*> self.fst.Copy()
        return result

    def __getitem__(self, int stateid):
        if not (0 <= stateid < len(self)):
            raise KeyError('state index out of range')
        cdef {{state}} state = {{state}}.__new__({{state}})
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

    def add_arc(self, int source, int dest, int ilabel, int olabel, weight=None):
        """fst.add_arc(int source, int dest, int ilabel, int olabel, weight=None):
        add an arc source->dest labeled with labels ilabel:olabel and weighted with weight"""
        if source > self.fst.NumStates()-1:
            raise ValueError('invalid source state id ({0} > {0})'.format(source, self.fst.NumStates()-1))
        if not isinstance(weight, {{weight}}):
            weight = {{weight}}(weight)
        cdef cfst.{{arc}}* arc = new cfst.{{arc}}(ilabel, olabel, (<{{weight}}> weight).weight[0], dest)
        self.fst.AddArc(source, arc[0])
        del arc

    def add_state(self):
        """fst.add_state() -> new state"""
        return self.fst.AddState()

    def __richcmp__({{fst}} x, {{fst}} y, int op):
        if op == 2: # ==
            return cfst.Equivalent(x.fst[0], y.fst[0])
        elif op == 3: # !=
            return not (x == y)
        raise NotImplemented('comparison not implemented for {{fst}}')

    def write(self, filename):
        """fst.write(filename): write the binary representation of the transducer in filename"""
        if self.isyms:
            self.fst.SetInputSymbols(self.isyms.table)
        if self.osyms:
            self.fst.SetOutputSymbols(self.osyms.table)
        return self.fst.Write(as_str(filename))

    property deterministic:
        def __get__(self):
            return (self.fst.Properties(cfst.kIDeterministic, True) & cfst.kIDeterministic)

    def determinize(self):
        """fst.determinize() -> determinized transducer"""
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        cfst.Determinize(self.fst[0], result.fst)
        return result

    def compose(self, {{fst}} other):
        """fst.compose({{fst}} other) == fst >> other -> composed transducer"""
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        cfst.Compose(self.fst[0], other.fst[0], result.fst)
        return result

    def __rshift__({{fst}} x, {{fst}} y):
        return x.compose(y)

    def intersect(self, {{fst}} other):
        """fst.intersect({{fst}} other) == fst & other -> intersection of the two transducers"""
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        cfst.Intersect(self.fst[0], other.fst[0], result.fst)
        return result

    def __and__({{fst}} x, {{fst}} y):
        return x.intersect(y)

    def set_union(self, {{fst}} other):
        """fst.set_union({{fst}} other): modify to the union of the two transducers"""
        cfst.Union(self.fst, other.fst[0])

    def union(self, {{fst}} other):
        """fst.union({{fst}} other) == fst | other -> union of the two transducers"""
        cdef {{fst}} result = self.copy()
        result.set_union(other)
        return result

    def __or__({{fst}} x, {{fst}} y):
        return x.union(y)

    def concatenate(self, {{fst}} other):
        """fst.concatenate({{fst}} other): modify to the concatenation of the two transducers"""
        cfst.Concat(self.fst, other.fst[0])

    def concatenation(self, {{fst}} other):
        """fst.concatenation({{fst}} other) == fst + other -> concatenation of the two transducers"""
        cdef {{fst}} result = self.copy()
        result.concatenate(other)
        return result

    def __add__({{fst}} x, {{fst}} y):
        return x.concatenation(y)

    def difference(self, {{fst}} other):
        """fst.difference({{fst}} other) == fst - other -> difference of the two transducers"""
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        cfst.Difference(self.fst[0], other.fst[0], result.fst)
        return result

    def __sub__({{fst}} x, {{fst}} y):
        return x.difference(y)

    def set_closure(self):
        """fst.set_closure(): modify to the Kleene closure of the transducer"""
        cfst.Closure(self.fst, cfst.CLOSURE_STAR)

    def closure(self):
        """fst.closure() -> Kleene closure of the transducer"""
        cdef {{fst}} result = self.copy()
        result.set_closure()
        return result

    def invert(self):
        """fst.invert(): modify to the inverse of the transducer"""
        cfst.Invert(self.fst)
    
    def inverse(self):
        """fst.inverse() -> inverse of the transducer"""
        cdef {{fst}} result = self.copy()
        result.invert()
        return result

    def reverse(self):
        """fst.reverse() -> reversed transducer"""
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        cfst.Reverse(self.fst[0], result.fst)
        return result

    def shortest_distance(self, bint reverse=False):
        """fst.shortest_distance(bool reverse=False) -> length of the shortest path"""
        cdef vector[cfst.{{weight}}]* distances = new vector[cfst.{{weight}}]()
        cfst.ShortestDistance(self.fst[0], distances, reverse)
        cdef list dist = []
        cdef unsigned i
        for i in range(distances.size()):
            dist.append({{weight}}(distances[0][i].Value()))
        del distances
        return dist

    def shortest_path(self, unsigned n=1):
        """fst.shortest_path(int n=1) -> transducer containing the n shortest paths"""
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        cfst.ShortestPath(self.fst[0], result.fst, n)
        return result

    def minimize(self):
        """fst.minimize(): minimize the transducer"""
        if not self.deterministic:
            raise ValueError('transducer is not deterministic')
        cfst.Minimize(self.fst)

    def arc_sort_input(self):
        """fst.arc_sort_input(): sort the input arcs of the transducer"""
        cdef cfst.ILabelCompare[cfst.{{arc}}]* icomp = new cfst.ILabelCompare[cfst.{{arc}}]()
        cfst.ArcSort(self.fst, icomp[0])
        del icomp

    def arc_sort_output(self):
        """fst.arc_sort_output(): sort the output arcs of the transducer"""
        cdef cfst.OLabelCompare[cfst.{{arc}}]* ocomp = new cfst.OLabelCompare[cfst.{{arc}}]()
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
        if not isinstance(threshold, {{weight}}):
            threshold = {{weight}}(threshold)
        cfst.Prune(self.fst, (<{{weight}}> threshold).weight[0])

    def plus_map(self, value):
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        if not isinstance(value, {{weight}}):
            value = {{weight}}(value)
        cfst.ArcMap(self.fst[0], result.fst,
            cfst.Plus{{arc}}Mapper((<{{weight}}> value).weight[0]))
        return result

    def times_map(self, value):
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        if not isinstance(value, {{weight}}):
            value = {{weight}}(value)
        cfst.ArcMap(self.fst[0], result.fst,
            cfst.Times{{arc}}Mapper((<{{weight}}> value).weight[0]))
        return result

    def remove_weights(self):
        cdef {{fst}} result = {{fst}}(isyms=self.isyms, osyms=self.osyms)
        cfst.ArcMap(self.fst[0], result.fst, cfst.Rm{{weight}}Mapper())
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
        cdef cfst.FstDrawer[cfst.{{arc}}]* drawer =\
            new cfst.FstDrawer[cfst.{{arc}}](self.fst[0],
                isyms_table, osyms_table, ssyms_table,
                False, string(), 8.5, 11, True, False, 0.40, 0.25, 14, 5, False)
        drawer.Draw(out, 'fst')
        cdef bytes out_str = out.str()
        del drawer, out
        return out_str

{{/types}}

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
