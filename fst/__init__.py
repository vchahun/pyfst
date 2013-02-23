from fst._fst import EPSILON, EPSILON_ID, SymbolTable,\
        read, read_log, read_std, read_symbols,\
        LogWeight, LogArc, LogState, LogVectorFst,\
        TropicalWeight, StdArc, StdState, StdVectorFst

class SimpleFst(StdVectorFst):
    def __init__(self, isyms=None, osyms=None):
        """SimpleFst(isyms=None, osyms=None) -> transducer with input/output symbol tables"""
        StdVectorFst.__init__(self, isyms, osyms)
        self.start = self.add_state()
        self.isyms = (isyms if isyms is not None else SymbolTable())
        self.osyms = (osyms if osyms is not None else SymbolTable())

    def add_arc(self, src, tgt, ilabel, olabel, weight=None):
        """fst.add_arc(int source, int dest, ilabel, olabel, weight=None):
        add an arc source->dest labeled with labels ilabel:olabel and weighted with weight"""
        while src > len(self) - 1:
            self.add_state()
        StdVectorFst.add_arc(self, src, tgt, self.isyms[ilabel], self.osyms[olabel],
                weight=weight)

    def __getitem__(self, stateid):
        while stateid > len(self) - 1:
            self.add_state()
        return StdVectorFst.__getitem__(self, stateid)

class Acceptor(SimpleFst):
    def __init__(self, syms=None):
        """Acceptor(syms=None) -> acceptor transducer with an input/output symbol table"""
        StdVectorFst.__init__(self)
        self.start = self.add_state()
        self.isyms = self.osyms = (syms if syms is not None else SymbolTable())

    def add_arc(self, src, tgt, label, weight=None):
        """fst.add_arc(int source, int dest, label, weight=None):
        add an arc source->dest labeled with label and weighted with weight"""
        SimpleFst.add_arc(self, src, tgt, label, label, weight)

def linear_chain(text, syms=None):
    """linear_chain(text, syms=None) -> linear chain acceptor for the given input text"""
    chain = Acceptor(syms)
    for i, c in enumerate(text):
        chain.add_arc(i, i+1, c)
    chain[i+1].final = True
    return chain
