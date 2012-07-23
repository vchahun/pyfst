from fst import Fst

fst = Fst()

fst.start = fst.add_state()

fst.add_arc(0, 1, 1, 1, 0.5)
fst.add_arc(0, 1, 2, 2, 1.5)

fst.add_state()
fst.add_arc(1, 2, 3, 3, 2.5)

fst.add_state()
fst.set_final(2, 3.5)

fst.write('binary.fst')
