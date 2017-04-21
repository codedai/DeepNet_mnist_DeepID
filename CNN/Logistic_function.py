import numpy
import theano.tensor as T
from theano import function
from theano import shared
import theano

x = T.dmatrix('x')
out = 1 / (1 + T.exp(-x))
out_2 = (1 + T.tanh(x / 2)) / 2

logistic = function([x], out)
logistic_2 = function([x], out_2)

a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f = function([a, b], [diff, diff_squared, abs_diff])

# setting a default value for an argument
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, theano.In(y, value = 1), theano.In(w, value = 2, name = 'w_by_name')], z)
# print(f(33))
# print(f(33, w_by_name = 10, y = 2))

# Using shared Variables
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates = [(state, state + inc)])
decrementor = function([inc], state, updates = [(state, state - inc)])

fn_of_state = state * 2 + inc
foo = T.scalar(dtype=state.dtype)

skip_shared = function([inc, foo], fn_of_state, givens = [(state, foo)])
skip_shared(1, 3)
print(state.get_value())
