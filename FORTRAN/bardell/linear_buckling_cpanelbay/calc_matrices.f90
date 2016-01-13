SUBROUTINE CALC_K0Y1Y2(M, N, K0, y1, y2, a, b, r, ABD, &
                       u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                       v1tx, v1rx, v2tx, v2rx, v1ty, v1ry, v2ty, v2ry, &
                       w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    INTEGER, INTENT(IN) :: M, N
    REAL*8, INTENT(IN) :: y1, y2, a, b, r
    REAL*8, INTENT(IN) :: ABD(6, 6)
    REAL*8, INTENT(IN) :: u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry
    REAL*8, INTENT(IN) :: v1tx, v1rx, v2tx, v2rx, v1ty, v1ry, v2ty, v2ry
    REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    REAL*8, INTENT(OUT) :: K0(3*M*N, 3*M*N)

    INTEGER row, col, i, j, k, l
    REAL*8 A11, A12, A16, A22, A26, A66
    REAL*8 B11, B12, B16, B22, B26, B66
    REAL*8 D11, D12, D16, D22, D26, D66
    REAL*8 eta1, eta2

    REAL*8 fAufBu, fAufBuxi, fAuxifBu, fAuxifBuxi, fAufBv, fAufBvxi
    REAL*8 fAuxifBv, fAuxifBvxi, fAuxifBwxixi, fAuxifBw, fAufBwxixi
    REAL*8 fAuxifBwxi, fAufBw, fAufBwxi, fAvfBuxi, fAvxifBuxi, fAvfBu
    REAL*8 fAvxifBu, fAvfBv, fAvfBvxi, fAvxifBv, fAvxifBvxi, fAvfBwxixi
    REAL*8 fAvxifBwxixi, fAvfBw, fAvfBwxi, fAvxifBw, fAvxifBwxi
    REAL*8 fAwxixifBuxi, fAwfBuxi, fAwxifBuxi, fAwxixifBu, fAwfBu
    REAL*8 fAwxifBu, fAwxixifBv, fAwxixifBvxi, fAwfBv, fAwfBvxi, fAwxifBv
    REAL*8 fAwxifBvxi, fAwxixifBwxixi, fAwfBwxixi, fAwxixifBw
    REAL*8 fAwxifBwxixi, fAwxixifBwxi, fAwfBw, fAwfBwxi, fAwxifBw
    REAL*8 fAwxifBwxi
    REAL*8 gAugBu, gAugBueta, gAuetagBu, gAuetagBueta, gAugBv, gAugBveta
    REAL*8 gAuetagBv, gAuetagBveta, gAuetagBwetaeta, gAuetagBw
    REAL*8 gAugBwetaeta, gAuetagBweta, gAugBw, gAugBweta, gAvgBueta
    REAL*8 gAvetagBueta, gAvgBu, gAvetagBu, gAvgBv, gAvgBveta, gAvetagBv
    REAL*8 gAvetagBveta, gAvgBwetaeta, gAvetagBwetaeta, gAvgBw, gAvgBweta
    REAL*8 gAvetagBw, gAvetagBweta, gAwetaetagBueta, gAwgBueta
    REAL*8 gAwetagBueta, gAwetaetagBu, gAwgBu, gAwetagBu, gAwetaetagBv
    REAL*8 gAwetaetagBveta, gAwgBv, gAwgBveta, gAwetagBv, gAwetagBveta
    REAL*8 gAwetaetagBwetaeta, gAwgBwetaeta, gAwetaetagBw
    REAL*8 gAwetagBwetaeta, gAwetaetagBweta, gAwgBw, gAwgBweta, gAwetagBw
    REAL*8 gAwetagBweta

    A11 = ABD(1, 1)
    A12 = ABD(1, 2)
    A16 = ABD(1, 3)
    A22 = ABD(2, 2)
    A26 = ABD(2, 3)
    A66 = ABD(3, 3)

    B11 = ABD(1, 4)
    B12 = ABD(1, 5)
    B16 = ABD(1, 6)
    B22 = ABD(2, 5)
    B26 = ABD(2, 6)
    B66 = ABD(3, 6)

    D11 = ABD(4, 4)
    D12 = ABD(4, 5)
    D16 = ABD(4, 6)
    D22 = ABD(5, 5)
    D26 = ABD(5, 6)
    D66 = ABD(6, 6)

    eta1 = 2*y1/b - 1.
    eta2 = 2*y2/b - 1.

    DO j=1, N
        DO l=1, N

            CALL integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry, gAugBu)
            CALL integral_ffxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry, gAugBueta)
            CALL integral_ffxi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry, gAuetagBu)
            CALL integral_fxifxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, u1ty, u1ry, u2ty, u2ry, gAuetagBueta)
            CALL integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry, gAugBv)
            CALL integral_ffxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry, gAugBveta)
            CALL integral_ffxi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry, gAuetagBv)
            CALL integral_fxifxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry, gAuetagBveta)
            CALL integral_fxifxixi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAuetagBwetaeta)
            CALL integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry, gAuetagBw)
            CALL integral_ffxixi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAugBwetaeta)
            CALL integral_fxifxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAuetagBweta)
            CALL integral_ff_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAugBw)
            CALL integral_ffxi_12(eta1, eta2, j, l, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAugBweta)
            CALL integral_ffxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry, gAvgBueta)
            CALL integral_fxifxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry, gAvetagBueta)
            CALL integral_ff_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, u1ty, u1ry, u2ty, u2ry, gAvgBu)
            CALL integral_ffxi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, v1ty, v1ry, v2ty, v2ry, gAvetagBu)
            CALL integral_ff_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry, gAvgBv)
            CALL integral_ffxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry, gAvgBveta)
            CALL integral_ffxi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry, gAvetagBv)
            CALL integral_fxifxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, v1ty, v1ry, v2ty, v2ry, gAvetagBveta)
            CALL integral_ffxixi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvgBwetaeta)
            CALL integral_fxifxixi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvetagBwetaeta)
            CALL integral_ff_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvgBw)
            CALL integral_ffxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvgBweta)
            CALL integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry, gAvetagBw)
            CALL integral_fxifxi_12(eta1, eta2, j, l, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAvetagBweta)
            CALL integral_fxifxixi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBueta)
            CALL integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry, gAwgBueta)
            CALL integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry, gAwetagBueta)
            CALL integral_ffxixi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBu)
            CALL integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, u1ty, u1ry, u2ty, u2ry, gAwgBu)
            CALL integral_ffxi_12(eta1, eta2, l, j, u1ty, u1ry, u2ty, u2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBu)
            CALL integral_ffxixi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBv)
            CALL integral_fxifxixi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBveta)
            CALL integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry, gAwgBv)
            CALL integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry, gAwgBveta)
            CALL integral_ffxi_12(eta1, eta2, l, j, v1ty, v1ry, v2ty, v2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBv)
            CALL integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, v1ty, v1ry, v2ty, v2ry, gAwetagBveta)
            CALL integral_fxixifxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBwetaeta)
            CALL integral_ffxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBwetaeta)
            CALL integral_ffxixi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBw)
            CALL integral_fxifxixi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBwetaeta)
            CALL integral_fxifxixi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBweta)
            CALL integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBw)
            CALL integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBweta)
            CALL integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBw)
            CALL integral_fxifxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBweta)

            DO i=1, M
                DO k=1, M

                    row = 3*((j-1)*M + (i-1)) + 1
                    col = 3*((l-1)*M + (k-1)) + 1

                    !IF (row > col) THEN
                        !continue
                    !END IF

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


SUBROUTINE CALC_KG0Y1Y2(M, N, KG0, y1, y2, a, b, Nxx, Nyy, Nxy, w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    INTEGER, INTENT(IN) :: M, N
    REAL*8, INTENT(IN) :: y1, y2, a, b, Nxx, Nyy, Nxy
    REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    REAL*8, INTENT(OUT) :: KG0(3*M*N, 3*M*N)

    INTEGER row, col, i, j, k, l
    REAL*8 eta1, eta2

    REAL*8 fAwfBw, fAwfBwxi, fAwxifBw, fAwxifBwxi
    REAL*8 gAwgBw, gAwgBweta, gAwetagBw, gAwetagBweta

    eta1 = 2*y1/b - 1.
    eta2 = 2*y2/b - 1.

    DO j=1, N
        DO l=1, N

            CALL integral_ff_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBw)
            CALL integral_ffxi_12(eta1, eta2, j, l, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBweta)
            CALL integral_ffxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBw)
            CALL integral_fxifxi_12(eta1, eta2, l, j, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBweta)

            DO i=1, M
                DO k=1, M

                    row = 3*((j-1)*M + (i-1)) + 1
                    col = 3*((l-1)*M + (k-1)) + 1

                    !IF (row > col) THEN
                        !continue
                    !END IF

                    CALL integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBw)
                    CALL integral_ffxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBwxi)
                    CALL integral_ffxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBw)
                    CALL integral_fxifxi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxi)

                    KG0(row+2, col+2) = Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

                END DO
            END DO
        END DO
    END DO

END SUBROUTINE


SUBROUTINE CALC_K0F(M, N, K0F, ys, a, b, bf, df, &
                    E1, F1, S1, Jxx, &
                    u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                    w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    INTEGER, INTENT(IN) :: M, N
    REAL*8, INTENT(IN) :: ys, a, b, bf, df
    REAL*8, INTENT(IN) :: E1, F1, S1, Jxx
    REAL*8, INTENT(IN) :: u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry
    REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    REAL*8, INTENT(OUT) :: K0F(3*M*N, 3*M*N)

    INTEGER row, col, i, j, k, l

    REAL*8 fAuxifBuxi, fAuxifBwxixi, fAuxifBwxi, fAwxixifBuxi
    REAL*8 fAwxifBuxi, fAwxifBwxi, fAwxifBwxixi, fAwxixifBwxi
    REAL*8 fAwxixifBwxixi
    REAL*8 gAu, gBu, gAw, gBw, gAweta, gBweta
    REAL*8 eta

    eta = 2*ys/b -1.

    DO i=1, M
        DO k=1, M

            CALL integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx, fAuxifBuxi)
            CALL integral_fxifxixi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAuxifBwxixi)
            CALL integral_fxifxi(i, k, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAuxifBwxi)
            CALL integral_fxifxixi(k, i, u1tx, u1rx, u2tx, u2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBuxi)
            CALL integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, u1tx, u1rx, u2tx, u2rx, fAwxifBuxi)
            CALL integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxi)
            CALL integral_fxifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxixi)
            CALL integral_fxifxixi(k, i, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBwxi)
            CALL integral_fxixifxixi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBwxixi)

            DO j=1, N

                CALL calc_f(j, eta, u1ty, u1ry, u2ty, u2ry, gAu)
                CALL calc_f(j, eta, w1ty, w1ry, w2ty, w2ry, gAw)
                CALL calc_fxi(j, eta, w1ty, w1ry, w2ty, w2ry, gAweta)

                DO l=1, N

                    row = 3*((j-1)*M + (i-1)) + 1
                    col = 3*((l-1)*M + (k-1)) + 1

                    !IF (row > col) THEN
                        !continue
                    !END IF

                    CALL calc_f(l, eta, u1ty, u1ry, u2ty, u2ry, gBu)
                    CALL calc_f(l, eta, w1ty, w1ry, w2ty, w2ry, gBw)
                    CALL calc_fxi(l, eta, w1ty, w1ry, w2ty, w2ry, gBweta)

                    K0F(row+0, col+0) = 2*E1*bf*fAuxifBuxi*gAu*gBu/a
                    K0F(row+0, col+2) = 0.5*a*bf*(8*E1*df*fAuxifBwxixi*gAu*gBw/(a*a*a) - 8*S1*fAuxifBwxi*gAu*gBweta/((a*a)*b))
                    K0F(row+2, col+0) = bf*gBu*(4*E1*df*fAwxixifBuxi*gAw/(a*a) - 4*S1*fAwxifBuxi*gAweta/(a*b))
                    K0F(row+2, col+2) = 0.5*a*bf*(-4*gBweta*(-4*Jxx*fAwxifBwxi*gAweta/(a*b) + 4*S1*df*fAwxixifBwxi*gAw/(a*a))/(a*b) - 4*gBw*(4*S1*df*fAwxifBwxixi*gAweta/(a*b) + fAwxixifBwxixi*gAw*(-4*E1*(df*df) - 4*F1)/(a*a))/(a*a))

                END DO
            END DO
        END DO
    END DO

END SUBROUTINE


SUBROUTINE CALC_KG0F(M, N, KG0F, ys, Fx, a, b, bf, df, &
                     u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                     w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    INTEGER, INTENT(IN) :: M, N
    REAL*8, INTENT(IN) :: ys, Fx, a, b, bf, df
    REAL*8, INTENT(IN) :: u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry
    REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    REAL*8, INTENT(OUT) :: KG0F(3*M*N, 3*M*N)

    INTEGER row, col, i, j, k, l

    REAL*8 fAwxifBwxi, gAw, gBw
    REAL*8 eta

    eta = 2*ys/b -1.

    DO i=1, M
        DO k=1, M

            CALL integral_fxifxi(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxi)

            DO j=1, N

                CALL calc_f(j, eta, w1ty, w1ry, w2ty, w2ry, gAw)

                DO l=1, N

                    row = 3*((j-1)*M + (i-1)) + 1
                    col = 3*((l-1)*M + (k-1)) + 1

                    !IF (row > col) THEN
                        !continue
                    !END IF

                    CALL calc_f(l, eta, w1ty, w1ry, w2ty, w2ry, gBw)

                    KG0F(row+2, col+2) = 2*Fx*fAwxifBwxi*gAw*gBw/a

                END DO
            END DO
        END DO
    END DO

END SUBROUTINE
