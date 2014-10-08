from sympy.printing.printer import Printer
from sympy.printing.precedence import precedence

class MathematicaPrinter(Printer):
    """Print SymPy's expressions using Mathematica syntax. """
    printmethod = "_mathematica"

    _default_settings = {}

    _function_translation_table = {
        'asin': 'ArcSin',
        'sin': 'Sin',
        'cos': 'Cos',
    }

    def parenthesize(self, item, level):
        printed = self._print(item)

        if precedence(item) <= level:
            return "(%s)" % printed
        else:
            return printed

    def emptyPrinter(self, expr):
        return str(expr)

    def _print_Pow(self, expr):
        prec = precedence(expr)

        if expr.exp == -1:
            return '1/%s' % (self.parenthesize(expr.base, prec))
        else:
            return '%s^%s' % (self.parenthesize(expr.base, prec),
                              self.parenthesize(expr.exp, prec))

    def _print_Matrix(self, expr):
        return '{' + ','.join([\
               '{' + ','.join([self._print(expr[j,i]) \
                               for i in range(expr.shape[1])]) + '}'\
                               for j in range(expr.shape[0])]) + '}'

    def _print_ImmutableMatrix(self, expr):
        return self._print_Matrix(expr)

    def _print_ImmutableDenseMatrix(self, expr):
        return self._print_Matrix(expr)

    def _print_MutableMatrix(self, expr):
        return self._print_Matrix(expr)

    def _print_MutableDenseMatrix(self, expr):
        return self._print_Matrix(expr)

    def _print_Mul(self, expr):
        prec = precedence(expr)
        return '*'.join([ self.parenthesize(arg, prec) for arg in expr.args])

    def _print_Add(self, expr):
        prec = precedence(expr)
        return '+'.join([ self.parenthesize(arg, prec) for arg in expr.args ])

    def _print_Pi(self, expr):
        return 'Pi'

    def _print_Function(self, expr):
        name = expr.func.__name__
        args = ", ".join([ self._print(arg) for arg in expr.args ])

        if expr.func.nargs is not None:
            try:
                name = self._function_translation_table[name]
            except KeyError:
                name = name.capitalize()

        return "%s[%s]" % (name, args)

def print_mathematica(expr, tofile=None, **settings):
    """Transform an expression to a string with Mathematica syntax. """
    p = MathematicaPrinter(settings)
    s = p.doprint(expr)
    if tofile:
        with open(tofile, 'w') as f:
            f.write(s + '\n')
    return s
