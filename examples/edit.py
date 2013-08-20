#!/usr/bin/env python
import sys
import fst

def make_edit(sigma):
    """
    Make an edit distance transducer with operations:
    - deletion:     x:<epsilon>/1
    - insertion:    <epsilon>:x/1
    - substitution: x:x/0 and x/y:1
    """
    # Create common symbol table
    syms = fst.SymbolTable()

    # Create transducer
    edit = fst.Transducer(syms, syms)
    edit[0].final = True
    for x in sigma:
        edit.add_arc(0, 0, x, fst.EPSILON, 1)
        edit.add_arc(0, 0, fst.EPSILON, x, 1)
        for y in sigma:
            edit.add_arc(0, 0, x, y, (0 if x == y else 1))

    # Define edit distance
    def distance(a, b):
        # Compose a o edit transducer o b
        composed = fst.linear_chain(a, syms) >> edit >> fst.linear_chain(b, syms)
        # Compute distance
        distances = composed.shortest_distance(reverse=True)
        dist = int(distances[0])
        # Find best alignment
        alignment = composed.shortest_path()
        # Re-order states
        alignment.top_sort()
        # Replace <epsilon> -> "-"
        alignment.relabel({fst.EPSILON: '-'}, {fst.EPSILON: '-'})
        # Read alignment on the arcs of the transducer
        arcs = (next(state.arcs) for state in alignment)
        labels = ((arc.ilabel, arc.olabel) for arc in arcs)
        align = [(alignment.isyms.find(x), alignment.osyms.find(y)) for x, y in labels]
        return dist, align

    return distance

def main(a, b):
    """
    python edit.py atctagctagctagtgctagctgatgctgatcga acgtgtgctagtcgtgatggcatgctg
    Distance: 14
    atctagctagctagtgctagctgat-gc-tgatcga
    a-cgtg-t-gctagt-c--g-tgatggcatgct-g-
    """
    edit_distance = make_edit(set(a+b))
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
