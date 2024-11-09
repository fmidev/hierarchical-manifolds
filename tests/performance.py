# comparative performance test
# Usage: python -m tests.performance
#
# T.Makinen terhi.makinen@fmi.fi

from .testutils import PerformanceTest

# set up and execute the test

test = PerformanceTest('Multi')
test.r_train = [ 1, 2, 4, 8, 16, 32, 64, 128 ]
test.data.n_data = 1024
test.data.n_classes = 10
test.data.distance = (0.5, 3.0)
test.data.random_state = 20240821

test.run()

