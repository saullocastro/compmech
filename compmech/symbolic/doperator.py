import subprocess

import sympy
from sympy.core.decorators import call_highest_priority
from sympy import Expr, Matrix, Mul, Add, diff
from sympy.core.numbers import Zero

class D(Expr):
    _op_priority = 11.
    is_commutative = False
    def __init__(self, *variables, **assumptions):
        super(D, self).__init__()
        self.evaluate = False
        self.variables = variables

    def __repr__(self):
        return 'D%s' % str(self.variables)

    def __str__(self):
        return self.__repr__()

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return Mul(other, self)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if isinstance(other, D):
            variables = self.variables + other.variables
            return D(*variables)
        if isinstance(other, Matrix):
            other_copy = other.copy()
            for i, elem in enumerate(other):
                other_copy[i] = self * elem
            return other_copy

        if self.evaluate:
            return diff(other, *self.variables)
        else:
            return Mul(self, other)

    def __pow__(self, other):
        variables = self.variables
        for i in range(other-1):
            variables += self.variables
        return D(*variables)

def mydiff(expr, *variables):
    if isinstance(expr, D):
        expr.variables += variables
        return D(*expr.variables)
    if isinstance(expr, Matrix):
        expr_copy = expr.copy()
        for i, elem in enumerate(expr):
            expr_copy[i] = diff(elem, *variables)
        return expr_copy
    return diff(expr, *variables)

def evaluateMul(expr):
    end = 0
    if expr.args <> ():
        if isinstance(expr.args[-1], D):
            return Zero()
    for i in range(len(expr.args)-1+end, -1, -1):
        arg = expr.args[i]
        if isinstance(arg, Add):
            arg = evaluateAdd(arg)
        elif isinstance(arg, Mul):
            arg = evaluateMul(arg)
        elif isinstance(arg, D):
            left = Mul(*expr.args[:i])
            right = Mul(*expr.args[i+1:])
            right = mydiff(right, *arg.variables)
            ans = left * right
            return evaluateMul(ans)
        else:
            pass
    return expr

def evaluateAdd(expr):
    newargs = []
    for arg in expr.args:
        if isinstance(arg, Mul):
            arg = evaluateMul(arg)
        elif isinstance(arg, Add):
            arg = evaluateAdd(arg)
        elif isinstance(arg, D):
            arg = Zero()
        else:
            pass
        newargs.append(arg)
    return Add(*newargs)

def evaluateExpr(expr):
    if isinstance(expr, Matrix):
        for i, elem in enumerate(expr):
            elem = elem.expand()
            expr[i] = evaluateExpr(elem)
        return expr
    expr = expr.expand()
    if isinstance(expr, Mul):
        expr = evaluateMul(expr)
    elif isinstance(expr, Add):
        expr = evaluateAdd(expr)
    elif isinstance(expr, D):
        expr = Zero()
    return expr

def symList(string, imin, imax):
    outlist = []
    for i in range(imin, imax+1):
        string2 = (string+('%d' % i))
        sympy.var(string2)
        outlist.append(eval(string2))
    return outlist

def print_to_latex(expr):
    import matplotlib.pyplot as pyplot
    text = sympy.latex(expr, mode='inline')
    pyplot.plot ; pyplot.text(0.5, 0.5, '%s' % text) ; pyplot.show()
    return text

def print_to_file(guy, append=False,
               software=r"C:\Program Files (x86)\Notepad++\notepad++.exe"):
    flag = 'w'
    if append: flag = 'a'
    outfile = open(r'print.txt', flag)
    outfile.write('\n')
    outfile.write(sympy.pretty(guy, wrap_line=False))
    outfile.write('\n')
    outfile.close()
    subprocess.Popen(software + ' print.txt')

