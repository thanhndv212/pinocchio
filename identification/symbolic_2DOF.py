import numpy as np 
from sympy import *
x, y, z = symbols('x y z')
init_printing()
I = MatrixSymbol('I', 3, 3)
I = Matrix([[x, y, y], [x, y, x],[y, x, z]])
print(I)