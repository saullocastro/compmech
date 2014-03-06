import numpy as np
import sympy
from sympy import collect

def mprint_as_sparse(m, mname, sufix, numeric=False, use_cse=False,
        header=None, print_file=True, collect_for=None):
    left, right = sufix
    if use_cse:
        subs, m_list = sympy.cse(m)
        for i, v in enumerate(m_list):
            m[i] = v
    filename = 'print_{mname}_{sufix}.txt'.format(mname=mname, sufix=sufix)
    ls = []
    if header:
        ls.append(header)
    if use_cse:
        ls.append('cdefs')
        num = 10
        for i, sub in enumerate(subs[::num]):
            ls.append('cdef double ' + ', '.join(
                        map(str, [j[0] for j in subs[num*i:num*(i+1)]])))
        ls.append('subs')
        for sub in subs:
            ls.append('{0} = {1}'.format(*sub))
    if not numeric:
        ls.append('# {mname}_{sufix}'.format(mname=mname, sufix=sufix))
        num = len([i for i in list(m) if i])
        ls.append('# {mname}_{sufix}_num={num}'.format(
            mname=mname, sufix=sufix, num=num))
        for (i, j), v in np.ndenumerate(m):
            if v:
                ls.append('c += 1')

                if left=='0':
                    ls.append('{mname}r[c] = {i}'.format(mname=mname, i=i))
                else:
                    ls.append('{mname}r[c] = row+{i}'.format(mname=mname, i=i))

                if right=='0':
                    ls.append('{mname}c[c] = {j}'.format(mname=mname, j=j))
                else:
                    ls.append('{mname}c[c] = col+{j}'.format(mname=mname, j=j))

                if collect_for!=None:
                    v = collect(v, collect_for, evaluate=False)
                    ls.append('{mname}v[c] +='.format(mname=mname))
                    for k, expr in v.items():
                        ls.append('#   collected for {k}'.format(k=k))
                        ls.append('    {expr}'.format(expr=k*expr))
                else:
                    ls.append('{mname}v[c] += {v}'.format(mname=mname, v=v))
    else:
        ls.append('# {mname}_{sufix}'.format(mname=mname, sufix=sufix))
        num = len([i for i in list(m) if i])
        ls.append('# {mname}_{sufix}_num={num}'.format(
            mname=mname, sufix=sufix, num=num))
        ls.append('#')
        ls.append('# values')
        ls.append('#')
        for (i, j), v in np.ndenumerate(m):
            if v:
                ls.append('c += 1')
                ls.append('fval[c+fdim*pti] = {v}'.format(mname=mname, v=v))
        ls.append('#')
        ls.append('# rows and columns')
        ls.append('#')
        for (i, j), v in np.ndenumerate(m):
            if v:
                ls.append('c += 1')
                ls.append('csub += 1')

                if left=='0':
                    ls.append('rows[c] = {i}'.format(i=i))
                else:
                    ls.append('rows[c] = row+{i}'.format(i=i))

                if right=='0':
                    ls.append('cols[c] = {j}'.format(j=j))
                else:
                    ls.append('cols[c] = col+{j}'.format(j=j))

                ls.append('k0Lv[c] = subv[csub]')
    string = '\n'.join(ls)
    if print_file:
        with open(filename, 'w') as f:
            f.write(sting)
    return string

