# pyfst

Python interface to [OpenFst](http://openfst.org)

## Installation

1. Install the latest version of OpenFst (1.3.2)
2. `pip install -e git+https://github.com/vchahun/pyfst.git#egg=pyfst`

## Usage

The [basic example](http://www.openfst.org/twiki/bin/view/FST/FstQuickTour#CreatingFsts) from the documentation translates to:

```python
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
```
