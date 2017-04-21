# To get start with Theano and get a feel of what we're working with, let us make a simple function: add two numbers
# together.
import numpy
import theano.tensor as T
from theano import function

# add two 0-dimensional
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)


# print(f(2, 3))
# print(numpy.allclose(f(16.3, 12.1), 28.4))
# print(numpy.allclose(z.eval({x : 2.3, y: 3.1}), 5.4))

# 5.0
# True
# True

# add two Matrices
x = T.dmatrix('x')
y = T.dmatrix('y')

z = x + y

f = function([x, y], z)
# print( f([[1, 2], [3, 4]], [[10, 20], [30, 40]]) )
# print(f(numpy.array([[1, 2], [3, 4]]), numpy.array([[10, 20], [30, 40]])))

# [[ 11.  22.]
#  [ 33.  44.]]
# [[ 11.  22.]
#  [ 33.  44.]]

a = T.vector('a')
b = T.vector('b')

out = a ** 2 + b ** 2 + 2 * a * b

f = function([a, b], out)

# print(f([1, 2], [4, 5]))
# [ 25.  49.]
