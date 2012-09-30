#!/usr/bin/env python
from itertools import chain
import sys
import fst

def make_input(word, syms):
    """
    Make a charcter input transducer:
    [0] =w:w=> 1 =o:o=> 2 =r:r=> 3 =d:d=> (4) 
    """
    inp = fst.StdVectorFst()
    inp.start = inp.add_state()
    source = inp.start
    for c in word:
        dest = inp.add_state()
        inp.add_arc(source, dest, syms[c], syms[c])
        source = dest
    inp[source].final = True
    return inp

def make_edit(sigma):
    """ Make an edit distance transducer """
    # Create transducer
    syms = fst.SymbolTable()
    sigma.add('<eps>')
    edit = fst.StdVectorFst()
    edit.start = edit.add_state()
    edit[0].final = True
    for x in sigma:
        for y in sigma:
            if x == y == '<eps>': continue
            edit.add_arc(0, 0, syms[x], syms[y], (0 if x == y else 1))

    # Define edit distance
    def distance(a, b):
        # Compose a o edit transducer o b
        comp = make_input(a, syms) >> edit >> make_input(b, syms)
        # Compute distance
        distances = comp.shortest_distance(reverse=True)
        dist = int(distances[0])
        # Find best alignment
        alignment = comp.shortest_path()
        # Re-order states
        alignment.top_sort()
        # Replace "<eps>" -> "-"
        dash = syms['-']
        eps = syms['<eps>']
        alignment.relabel(ipairs=[(eps, dash)], opairs=[(eps, dash)])
        arcs = (next(iter(state)) for state in alignment)
        labels = ((arc.ilabel, arc.olabel) for arc in arcs)
        align = [(syms.find(x), syms.find(y)) for x, y in labels]
        return dist, align

    return distance

def main(a, b):
    """
    python edit.py atctagctagctagtgctagctgatgctgatcga acgtgtgctagtcgtgatggcatgctg
    Distance: 14
    atctagctagctagtgctagctgat-gc-tgatcga
    a-cgtg-t-gctagt-c--g-tgatggcatgct-g-
    """
    sigma = set(chain(a, b))
    edit_distance = make_edit(sigma)
    dist, align = edit_distance(a, b)
    print('Distance: {0}'.format(dist))
    x, y = zip(*align)
    print(''.join(x))
    print(''.join(y))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: {0} a b\n'.format(sys.argv[0]))
        sys.exit(1)
    main(*sys.argv[1:])
