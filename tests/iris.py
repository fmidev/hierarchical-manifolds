# application of H-LDA on the Iris data set
# Usage: python -m tests.iris
#
# T.Makinen terhi.makinen@fmi.fi

from sklearn.datasets import load_iris
from .testutils import ClassData, HLDATest

# set up and execute the test

data = load_iris()
test = HLDATest('Iris', ClassData(data.data, data.target, data.target_names),
    outdir='tests/iris/')

test.run()
