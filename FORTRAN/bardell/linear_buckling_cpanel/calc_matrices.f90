SUBROUTINE CALC_K0(M, N, K0, a, b, r, &
                   A11, A12, A16, A22, A26, A66, B11, B12, B16, B22, B26, B66, D11, D12, D16, D22, D26, D66, &
                   u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                   v1tx, v1rx, v2tx, v2rx, v1ty, v1ry, v2ty, v2ry, &
                   w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    IMPLICIT NONE
    REAL*8, INTENT(IN) :: A11, A12, A16, A22, A26, A66
    REAL*8, INTENT(IN) :: B11, B12, B16, B22, B26, B66
    REAL*8, INTENT(IN) :: D11, D12, D16, D22, D26, D66
    INTEGER, INTENT(IN) :: M, N
    REAL*8, INTENT(IN) :: a, b, r
    REAL*8, INTENT(IN) :: u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry
    REAL*8, INTENT(IN) :: v1tx, v1rx, v2tx, v2rx, v1ty, v1ry, v2ty, v2ry
    REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    REAL*8, INTENT(OUT) :: K0(3*M*N, 3*M*N)

    INTEGER row, col, i, j, k, l

    REAL*8 fAufBu, fAufBuxi, fAuxifBu, fAuxifBuxi, fAufBv, fAufBvxi, fAuxifBv, fAuxifBvxi, fAuxifBwxixi, fAuxifBw
    REAL*8 fAufBwxixi, fAuxifBwxi, fAufBw, fAufBwxi, fAvfBuxi, fAvxifBuxi, fAvfBu, fAvxifBu, fAvfBv, fAvfBvxi
    REAL*8 fAvxifBv, fAvxifBvxi, fAvfBwxixi, fAvxifBwxixi, fAvfBw, fAvfBwxi, fAvxifBw, fAvxifBwxi, fAwxixifBuxi
    REAL*8 fAwfBuxi, fAwxifBuxi, fAwxixifBu, fAwfBu, fAwxifBu, fAwxixifBv, fAwxixifBvxi, fAwfBv, fAwfBvxi
    REAL*8 fAwxifBv, fAwxifBvxi, fAwxixifBwxixi, fAwfBwxixi, fAwxixifBw, fAwxifBwxixi, fAwxixifBwxi, fAwfBw
    REAL*8 fAwfBwxi, fAwxifBw, fAwxifBwxi

    REAL*8 gAugBu, gAugBueta, gAuetagBu, gAuetagBueta, gAugBv, gAugBveta, gAuetagBv, gAuetagBveta, gAuetagBwetaeta
    REAL*8 gAuetagBw, gAugBwetaeta, gAuetagBweta, gAugBw, gAugBweta, gAvgBueta, gAvetagBueta, gAvgBu, gAvetagBu
    REAL*8 gAvgBv, gAvgBveta, gAvetagBv, gAvetagBveta, gAvgBwetaeta, gAvetagBwetaeta, gAvgBw, gAvgBweta, gAvetagBw
    REAL*8 gAvetagBweta, gAwetaetagBueta, gAwgBueta, gAwetagBueta, gAwetaetagBu, gAwgBu, gAwetagBu, gAwetaetagBv
    REAL*8 gAwetaetagBveta, gAwgBv, gAwgBveta, gAwetagBv, gAwetagBveta, gAwetaetagBwetaeta, gAwgBwetaeta
    REAL*8 gAwetaetagBw, gAwetagBwetaeta, gAwetaetagBweta, gAwgBw, gAwgBweta, gAwetagBw, gAwetagBweta

    DO j=1, N
        DO l=1, N

            CALL integral_ff(j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry, gAugBu)
            CALL integral_ffxi(j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry, gAugBueta)
            CALL integral_ffxi(l, j, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry, gAuetagBu)
            CALL integral_fxifxi(j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry, gAuetagBueta)
            CALL integral_ff(j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry, gAugBv)
            CALL integral_ffxi(j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry, gAugBveta)
            CALL integral_ffxi(l, j, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry, gAuetagBv)
            CALL integral_fxifxi(j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry, gAuetagBveta)
            CALL integral_fxifxixi(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAuetagBwetaeta)
            CALL integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry, gAuetagBw)
            CALL integral_ffxixi(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAugBwetaeta)
            CALL integral_fxifxi(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAuetagBweta)
            CALL integral_ff(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAugBw)
            CALL integral_ffxi(j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAugBweta)
            CALL integral_ffxi(j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry, gAvgBueta)
            CALL integral_fxifxi(j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry, gAvetagBueta)
            CALL integral_ff(j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry, gAvgBu)
            CALL integral_ffxi(l, j, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry, gAvetagBu)
            CALL integral_ff(j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry, gAvgBv)
            CALL integral_ffxi(j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry, gAvgBveta)
            CALL integral_ffxi(l, j, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry, gAvetagBv)
            CALL integral_fxifxi(j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry, gAvetagBveta)
            CALL integral_ffxixi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvgBwetaeta)
            CALL integral_fxifxixi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvetagBwetaeta)
            CALL integral_ff(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvgBw)
            CALL integral_ffxi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvgBweta)
            CALL integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry, gAvetagBw)
            CALL integral_fxifxi(j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvetagBweta)
            CALL integral_fxifxixi(l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBueta)
            CALL integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry, gAwgBueta)
            CALL integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry, gAwetagBueta)
            CALL integral_ffxixi(l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBu)
            CALL integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry, gAwgBu)
            CALL integral_ffxi(l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBu)
            CALL integral_ffxixi(l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBv)
            CALL integral_fxifxixi(l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBveta)
            CALL integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry, gAwgBv)
            CALL integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry, gAwgBveta)
            CALL integral_ffxi(l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBv)
            CALL integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry, gAwetagBveta)
            CALL integral_fxixifxixi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBwetaeta)
            CALL integral_ffxixi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBwetaeta)
            CALL integral_ffxixi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBw)
            CALL integral_fxifxixi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBwetaeta)
            CALL integral_fxifxixi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBweta)
            CALL integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBw)
            CALL integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBweta)
            CALL integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBw)
            CALL integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBweta)

            DO i=1, M
                DO k=1, M

                    row = 3*((j-1)*M + (i-1)) + 1
                    col = 3*((l-1)*M + (k-1)) + 1

                    IF (row > col) THEN
                        CYCLE
                    END IF

                    CALL integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx, fAufBu)
                    CALL integral_ffxi(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx, fAufBuxi)
                    CALL integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx, fAuxifBu)
                    CALL integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx, fAuxifBuxi)
                    CALL integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx, fAufBv)
                    CALL integral_ffxi(i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx, fAufBvxi)
                    CALL integral_ffxi(k, i, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx, fAuxifBv)
                    CALL integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx, fAuxifBvxi)
                    CALL integral_fxifxixi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAuxifBwxixi)
                    CALL integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx, fAuxifBw)
                    CALL integral_ffxixi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAufBwxixi)
                    CALL integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAuxifBwxi)
                    CALL integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAufBw)
                    CALL integral_ffxi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAufBwxi)
                    CALL integral_ffxi(i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx, fAvfBuxi)
                    CALL integral_fxifxi(i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx, fAvxifBuxi)
                    CALL integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, u1tx, u1rx, u2tx, u2rx, fAvfBu)
                    CALL integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, v1tx, v1rx, v2tx, v2rx, fAvxifBu)
                    CALL integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx, fAvfBv)
                    CALL integral_ffxi(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx, fAvfBvxi)
                    CALL integral_ffxi(k, i, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx, fAvxifBv)
                    CALL integral_fxifxi(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx, fAvxifBvxi)
                    CALL integral_ffxixi(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx, fAvfBwxixi)
                    CALL integral_fxifxixi(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx, fAvxifBwxixi)
                    CALL integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx, fAvfBw)
                    CALL integral_ffxi(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx, fAvfBwxi)
                    CALL integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx, fAvxifBw)
                    CALL integral_fxifxi(i, k, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx, fAvxifBwxi)
                    CALL integral_fxifxixi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBuxi)
                    CALL integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx, fAwfBuxi)
                    CALL integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx, fAwxifBuxi)
                    CALL integral_ffxixi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBu)
                    CALL integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx, fAwfBu)
                    CALL integral_ffxi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBu)
                    CALL integral_ffxixi(k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBv)
                    CALL integral_fxifxixi(k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBvxi)
                    CALL integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx, fAwfBv)
                    CALL integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx, fAwfBvxi)
                    CALL integral_ffxi(k, i, v1tx, v1rx, v2tx, v2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBv)
                    CALL integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, v1tx, v1rx, v2tx, v2rx, fAwxifBvxi)
                    CALL integral_fxixifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBwxixi)
                    CALL integral_ffxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBwxixi)
                    CALL integral_ffxixi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBw)
                    CALL integral_fxifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxixi)
                    CALL integral_fxifxixi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBwxi)
                    CALL integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBw)
                    CALL integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBwxi)
                    CALL integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBw)
                    CALL integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxi)

                    K0(row+0, col+0) = A11*b*fAuxifBuxi*gAugBu/a + A16*(fAufBuxi*gAuetagBu + fAuxifBu*gAugBueta) + A66*a*fAufBu*gAuetagBueta/b
                    K0(row+0, col+1) = A12*fAuxifBv*gAugBveta + A16*b*fAuxifBvxi*gAugBv/a + A26*a*fAufBv*gAuetagBveta/b + A66*fAufBvxi*gAuetagBv
                    K0(row+0, col+2) = 0.5*A12*b*fAuxifBw*gAugBw/r + 0.5*A26*a*fAufBw*gAuetagBw/r - 2*B11*b*fAuxifBwxixi*gAugBw/(a*a) - 2*B12*fAuxifBw*gAugBwetaeta/b - 2*B16*(fAufBwxixi*gAuetagBw + 2*fAuxifBwxi*gAugBweta)/a - 2*B26*a*fAufBw*gAuetagBwetaeta/(b*b) - 4*B66*fAufBwxi*gAuetagBweta/b
                    K0(row+1, col+0) = A12*fAvfBuxi*gAvetagBu + A16*b*fAvxifBuxi*gAvgBu/a + A26*a*fAvfBu*gAvetagBueta/b + A66*fAvxifBu*gAvgBueta
                    K0(row+1, col+1) = A22*a*fAvfBv*gAvetagBveta/b + A26*(fAvfBvxi*gAvetagBv + fAvxifBv*gAvgBveta) + A66*b*fAvxifBvxi*gAvgBv/a
                    K0(row+1, col+2) = 0.5*A22*a*fAvfBw*gAvetagBw/r + 0.5*A26*b*fAvxifBw*gAvgBw/r - 2*B12*fAvfBwxixi*gAvetagBw/a - 2*B16*b*fAvxifBwxixi*gAvgBw/(a*a) - 2*B22*a*fAvfBw*gAvetagBwetaeta/(b*b) - 2*B26*(2*fAvfBwxi*gAvetagBweta + fAvxifBw*gAvgBwetaeta)/b - 4*B66*fAvxifBwxi*gAvgBweta/a
                    K0(row+2, col+0) = 0.5*A12*b*fAwfBuxi*gAwgBu/r + 0.5*A26*a*fAwfBu*gAwgBueta/r - 2*B11*b*fAwxixifBuxi*gAwgBu/(a*a) - 2*B12*fAwfBuxi*gAwetaetagBu/b - 2*B16*(2*fAwxifBuxi*gAwetagBu + fAwxixifBu*gAwgBueta)/a - 2*B26*a*fAwfBu*gAwetaetagBueta/(b*b) - 4*B66*fAwxifBu*gAwetagBueta/b
                    K0(row+2, col+1) = 0.5*A22*a*fAwfBv*gAwgBveta/r + 0.5*A26*b*fAwfBvxi*gAwgBv/r - 2*B12*fAwxixifBv*gAwgBveta/a - 2*B16*b*fAwxixifBvxi*gAwgBv/(a*a) - 2*B22*a*fAwfBv*gAwetaetagBveta/(b*b) - 2*B26*(fAwfBvxi*gAwetaetagBv + 2*fAwxifBv*gAwetagBveta)/b - 4*B66*fAwxifBvxi*gAwetagBv/a
                    K0(row+2, col+2) = 0.25*A22*a*b*fAwfBw*gAwgBw/(r*r) - B12*b*gAwgBw*(fAwfBwxixi + fAwxixifBw)/(a*r) - B22*a*fAwfBw*(gAwgBwetaeta + gAwetaetagBw)/(b*r) - 2*B26*(fAwfBwxi*gAwgBweta + fAwxifBw*gAwetagBw)/r + 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*D12*(fAwfBwxixi*gAwetaetagBw + fAwxixifBw*gAwgBwetaeta)/(a*b) + 8*D16*(fAwxifBwxixi*gAwetagBw + fAwxixifBwxi*gAwgBweta)/(a*a) + 4*D22*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*D26*(fAwfBwxi*gAwetaetagBweta + fAwxifBw*gAwetagBwetaeta)/(b*b) + 16*D66*fAwxifBwxi*gAwetagBweta/(a*b)

                END DO
            END DO
        END DO
    END DO

END SUBROUTINE


SUBROUTINE CALC_KG0(M, N, KG0, a, b, Nxx, Nyy, Nxy, w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: M, N
    REAL*8, INTENT(IN) :: a, b, Nxx, Nyy, Nxy
    REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    REAL*8, INTENT(OUT) :: KG0(3*M*N, 3*M*N)

    INTEGER row, col, i, j, k, l
    REAL*8 fAwfBw, fAwfBwxi, fAwxifBw, fAwxifBwxi
    REAL*8 gAwgBw, gAwgBweta, gAwetagBw, gAwetagBweta

    DO j=1, N
        DO l=1, N

            CALL integral_ff(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBw)
            CALL integral_ffxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBweta)
            CALL integral_ffxi(l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBw)
            CALL integral_fxifxi(j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBweta)

            DO i=1, M
                DO k=1, M

                    row = 3*((j-1)*M + (i-1)) + 1
                    col = 3*((l-1)*M + (k-1)) + 1

                    IF (row > col) THEN
                        CYCLE
                    END IF

                    CALL integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBw)
                    CALL integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBwxi)
                    CALL integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBw)
                    CALL integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxi)

                    KG0(row+2, col+2) = Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

                END DO
            END DO
        END DO
    END DO

END SUBROUTINE
