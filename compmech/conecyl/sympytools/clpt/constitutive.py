import sympy
from sympy import Matrix

sympy.var('A11, A12, A16, A22, A26, A66', commutative=False)
sympy.var('B11, B12, B16, B22, B26, B66', commutative=False)
sympy.var('D11, D12, D16, D22, D26, D66', commutative=False)
# laminated constitutive relations
LC = Matrix([[A11, A12, A16, B11, B12, B16],
             [A12, A22, A26, B12, B22, B26],
             [A16, A26, A66, B16, B26, B66],
             [B11, B12, B16, D11, D12, D16],
             [B12, B22, B26, D12, D22, D26],
             [B16, B26, B66, D16, D26, D66]])
