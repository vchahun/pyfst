import fst
from nose.tools import eq_, ok_

def test_simple():
    t = fst.SimpleFst()
    for i, (ic, oc) in enumerate(zip('hello', 'olleh')):
        t.add_arc(i, i+1, ic, oc)
    t[i+1].final = True
    eq_(len(t), 6)
    ok_(t[5].final)

if __name__ == '__main__':
    test_simple()
