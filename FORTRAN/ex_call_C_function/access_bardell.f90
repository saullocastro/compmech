! Exercise to attempt calling a C function from FORTRAN

PROGRAM access_bardell
    USE ISO_C_BINDING
    INTERFACE
        FUNCTION integral_ff(i, j, x1t, x1r, x2t, x2r,&
                             y1t, y1r, y2t, y2r) BIND(C, name='integral_ff')
            USE ISO_C_BINDING
            INTEGER(KIND=C_INT), INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8 integral_ff
        END FUNCTION
    END INTERFACE

    INTEGER i, j
    REAL*8 x11, x1r, x2t, x2r, y1t, y1r, y2t, y2r
    i = 1
    j = 2
    x11 = 1.
    x1r = 1.
    x2t = 1.
    x2r = 1.
    y1t = 1.
    y1r = 1.
    y2t = 1.
    y2r = 1.
    WRITE(*, *) integral_ff(i, j, x1t, x1r, x2t, x2r,&
                            y1t, y1r, y2t, y2r)

END PROGRAM
