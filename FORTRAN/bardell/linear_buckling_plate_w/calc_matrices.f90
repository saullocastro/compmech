SUBROUTINE CALC_K0(M, N, K, a, b, D11, D12, D16, D22, D26, D66, w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    REAL*8, INTENT(IN) :: D11, D12, D16, D22, D26, D66
    INTEGER, INTENT(IN) :: M, N
    REAL*8, INTENT(IN) :: a, b
    REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    REAL*8, INTENT(OUT) :: K(M*N, M*N)

    INTEGER row, col, i1, j1, k1, l1
    REAL*8 fAwxixifBwxixi, fAwfBwxixi, fAwxixifBw, fAwxifBwxixi, fAwxixifBwxi, fAwfBw, fAwfBwxi, fAwxifBw, fAwxifBwxi
    REAL*8 gAwetaetagBwetaeta, gAwgBwetaeta, gAwetaetagBw, gAwetagBwetaeta, gAwetaetagBweta, gAwgBw, gAwgBweta, gAwetagBw, gAwetagBweta

    DO j1=1, N
        DO l1=1, N

            CALL integral_fxixifxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBwetaeta)
            CALL integral_ffxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBwetaeta)
            CALL integral_ffxixi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBw)
            CALL integral_fxifxixi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBwetaeta)
            CALL integral_fxifxixi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetaetagBweta)
            CALL integral_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBw)
            CALL integral_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBweta)
            CALL integral_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBw)
            CALL integral_fxifxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBweta)

            DO i1=1, M
                DO k1=1, M

                    row = 1*((j1-1)*M + (i1-1)) + 1
                    col = 1*((l1-1)*M + (k1-1)) + 1

                    IF (row > col) THEN
                        CYCLE
                    END IF

                    CALL integral_fxixifxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBwxixi)
                    CALL integral_ffxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBwxixi)
                    CALL integral_ffxixi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBw)
                    CALL integral_fxifxixi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxixi)
                    CALL integral_fxifxixi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxixifBwxi)
                    CALL integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBw)
                    CALL integral_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBwxi)
                    CALL integral_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBw)
                    CALL integral_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxi)

                    K(row, col) = 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*D12*(fAwfBwxixi*gAwetaetagBw + fAwxixifBw*gAwgBwetaeta)/(a*b) + 8*D16*(fAwxifBwxixi*gAwetagBw + fAwxixifBwxi*gAwgBweta)/(a*a) + 4*D22*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*D26*(fAwfBwxi*gAwetaetagBweta + fAwxifBw*gAwetagBwetaeta)/(b*b) + 16*D66*fAwxifBwxi*gAwetagBweta/(a*b)

                END DO
            END DO
        END DO
    END DO

END SUBROUTINE


SUBROUTINE CALC_KG0(M, N, K, a, b, Nxx, Nyy, Nxy, w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    INTEGER, INTENT(IN) :: M, N
    REAL*8, INTENT(IN) :: a, b, Nxx, Nyy, Nxy
    REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    REAL*8, INTENT(OUT) :: K(M*N, M*N)

    INTEGER row, col, i1, j1, k1, l1
    REAL*8 fAwfBw, fAwfBwxi, fAwxifBw, fAwxifBwxi
    REAL*8 gAwgBw, gAwgBweta, gAwetagBw, gAwetagBweta

    DO j1=1, N
        DO l1=1, N

            CALL integral_ff(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBw)
            CALL integral_ffxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwgBweta)
            CALL integral_ffxi(l1, j1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBw)
            CALL integral_fxifxi(j1, l1, w1ty, w1ry, w2ty, w2ry, w1ty, w1ry, w2ty, w2ry, gAwetagBweta)

            DO i1=1, M
                DO k1=1, M

                    row = 1*((j1-1)*M + (i1-1)) + 1
                    col = 1*((l1-1)*M + (k1-1)) + 1

                    IF (row > col) THEN
                        CYCLE
                    END IF

                    CALL integral_ff(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBw)
                    CALL integral_ffxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwfBwxi)
                    CALL integral_ffxi(k1, i1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBw)
                    CALL integral_fxifxi(i1, k1, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx, fAwxifBwxi)

                    K(row, col) = Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*(fAwfBwxi*gAwetagBw + fAwxifBw*gAwgBweta) + Nyy*a*fAwfBw*gAwetagBweta/b

                END DO
            END DO
        END DO
    END DO

END SUBROUTINE
