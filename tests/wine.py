# application of H-LDA on the Wine data set
# Usage: python -m tests.wine
#
# T.Makinen terhi.makinen@fmi.fi

from sklearn.datasets import load_wine
from .testutils import ClassData, HLDATest

# set up and execute the test

data = load_wine()
test = HLDATest('Wine', ClassData(data.data, data.target, data.target_names),
    outdir='tests/wine/')

test.run()
