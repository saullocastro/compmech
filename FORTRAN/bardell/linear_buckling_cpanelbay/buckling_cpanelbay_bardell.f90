! BUCKLING_CPANELBAY_BARDELL program
!
! The required inputs that should be given are described below
! 
! Run control
! -----------
! NUM : integer
!   Number of returned eigenvalues
!
! M : integer
!   Number of terms along x
!
! N : integer
!   Number of terms along x
!
! Geometry
! --------
! a : float
!   The panel length (dimension along x)
! b : float
!   The panel circumferential width (dimension along y)
! r : float
!   The panel radius.
!
! Applied Loads
! -------------
! Nxx : float
!   Nxx stress
! Nyy : float
!   Nyy stress
! Nxy : float
!   Nxy stress
!
! Laminate Constitutive Varibles (matrix ABD)
! -------------------------------------------
! A11 : float
!   Membrane stiffness along x
! A12 : float
!   Membrane stiffness
! A16 : float
!   Shear-extension coupling
! A22 : float
!   Membrane stiffness along y
! A26 : float
!   Shear-extension coupling
! A66 : float
!   Membrane Shear stiffness
! B11 : float
!   Bending-extension coupling
! B12 : float
!   Bending-extension coupling
! B16 : float
!   Bending-extension coupling
! B22 : float
!   Bending-extension coupling
! B26 : float
!   Bending-extension coupling
! B66 : float
!   Bending-extension coupling
! D11 : float
!   Bending stiffness
! D12 : float
!   Bending stiffness
! D16 : float
!   Bending-twist stiffness
! D22 : float
!   Bending stiffness
! D26 : float
!   Bending-twist stiffness
! D66 : float
!   Twist (torsion) stiffness
!
! Boundary conditions
! -------------------
! u1tx : float
!   If 1. the edge at x=0 can translate along u   
!   If 0. the edge at x=0 cannot translate along u   
! u1rx : float
!   If 1. the end at x=0 can rotate
!   If 0. the end at x=0 cannot translate along u   
! u2tx : float
!   If 1. the edge at x=a can translate along u   
!   If 0. the edge at x=a cannot translate along u   
! u2rx : float
!   If 1. the end at x=a can rotate
!   If 0. the end at x=a cannot translate along u   
! u1ty : float
!   If 1. the edge at y=0 can translate along u   
!   If 0. the edge at y=0 cannot translate along u   
! u1ry : float
!   If 1. the end at y=0 can rotate
!   If 0. the end at y=0 cannot translate along u   
! u2ty : float
!   If 1. the edge at y=b can translate along u   
!   If 0. the edge at y=b cannot translate along u   
! u2ry : float
!   If 1. the end at y=b can rotate
!   If 0. the end at y=b cannot translate along u   
! v1tx : float
!   If 1. the edge at x=0 can translate along v   
!   If 0. the edge at x=0 cannot translate along v   
! v1rx : float
!   If 1. the end at x=0 can rotate
!   If 0. the end at x=0 cannot translate along v   
! v2tx : float
!   If 1. the edge at x=a can translate along v   
!   If 0. the edge at x=a cannot translate along v   
! v2rx : float
!   If 1. the end at x=a can rotate
!   If 0. the end at x=a cannot translate along v   
! v1ty : float
!   If 1. the edge at y=0 can translate along v   
!   If 0. the edge at y=0 cannot translate along v   
! v1ry : float
!   If 1. the end at y=0 can rotate
!   If 0. the end at y=0 cannot translate along v   
! v2ty : float
!   If 1. the edge at y=b can translate along v   
!   If 0. the edge at y=b cannot translate along v   
! v2ry : float
!   If 1. the end at y=b can rotate
!   If 0. the end at y=b cannot translate along v   
! w1tx : float
!   If 1. the edge at x=0 can translate along w   
!   If 0. the edge at x=0 cannot translate along w   
! w1rx : float
!   If 1. the end at x=0 can rotate
!   If 0. the end at x=0 cannot translate along w   
! w2tx : float
!   If 1. the edge at x=a can translate along w   
!   If 0. the edge at x=a cannot translate along w   
! w2rx : float
!   If 1. the end at x=a can rotate
!   If 0. the end at x=a cannot translate along w   
! w1ty : float
!   If 1. the edge at y=0 can translate along w   
!   If 0. the edge at y=0 cannot translate along w   
! w1ry : float
!   If 1. the end at y=0 can rotate
!   If 0. the end at y=0 cannot translate along w   
! w2ty : float
!   If 1. the edge at y=b can translate along w   
!   If 0. the edge at y=b cannot translate along w   
! w2ry : float
!   If 1. the end at y=b can rotate
!   If 0. the end at y=b cannot translate along w   

INCLUDE 'calc_matrices.f90'
INCLUDE '../bardell.f90'
INCLUDE '../bardell_12.f90'
INCLUDE '../bardell_functions.f90'

PROGRAM BUCKLING_CPANELBAY_BARDELL
    IMPLICIT NONE
    INTERFACE
        SUBROUTINE integral_ff(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_ffxi(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_ffxixi(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_fxifxi(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_fxifxixi(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_fxixifxixi(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_ff_12(xi1, xi2, i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: xi1, xi2, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_ffxi_12(xi1, xi2, i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: xi1, xi2, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_ffxixi_12(xi1, xi2, i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: xi1, xi2, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_fxifxi_12(xi1, xi2, i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: xi1, xi2, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_fxifxixi_12(xi1, xi2, i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: xi1, xi2, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE integral_fxixifxixi_12(xi1, xi2, i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i, j
            REAL*8, INTENT(IN) :: xi1, xi2, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE calc_f(i, xi, x1t, x1r, x2t, x2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i
            REAL*8, INTENT(IN) :: xi, x1t, x1r, x2t, x2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE calc_fxi(i, xi, x1t, x1r, x2t, x2r, out)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: i
            REAL*8, INTENT(IN) :: xi, x1t, x1r, x2t, x2r
            REAL*8, INTENT(OUT) :: out
        END SUBROUTINE
        SUBROUTINE CALC_K0Y1Y2(M, N, K0, y1, y2, a, b, r, ABD, &
                           u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                           v1tx, v1rx, v2tx, v2rx, v1ty, v1ry, v2ty, v2ry, &
                           w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: M, N
            REAL*8, INTENT(IN) :: y1, y2, a, b, r
            REAL*8, INTENT(IN) :: ABD(6, 6)
            REAL*8, INTENT(IN) :: u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry
            REAL*8, INTENT(IN) :: v1tx, v1rx, v2tx, v2rx, v1ty, v1ry, v2ty, v2ry
            REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
            REAL*8, INTENT(OUT) :: K0(3*M*N, 3*M*N)
        END SUBROUTINE
        SUBROUTINE CALC_KG0Y1Y2(M, N, KG0, y1, y2, a, b, Nxx, Nyy, Nxy, w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: M, N
            REAL*8, INTENT(IN) :: y1, y2, a, b, Nxx, Nyy, Nxy
            REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
            REAL*8, INTENT(OUT) :: KG0(3*M*N, 3*M*N)
        END SUBROUTINE
        SUBROUTINE CALC_K0F(M, N, K0F, ys, a, b, bf, df, &
                            E1, F1, S1, Jxx, &
                            u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                            w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: M, N
            REAL*8, INTENT(IN) :: ys, a, b, bf, df
            REAL*8, INTENT(IN) :: E1, F1, S1, Jxx
            REAL*8, INTENT(IN) :: u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry
            REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
            REAL*8, INTENT(OUT) :: K0F(3*M*N, 3*M*N)
        END SUBROUTINE
        SUBROUTINE CALC_KG0F(M, N, KG0F, ys, Fx, a, b, bf, df, &
                            u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                            w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: M, N
            REAL*8, INTENT(IN) :: ys, Fx, a, b, bf, df
            REAL*8, INTENT(IN) :: u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry
            REAL*8, INTENT(IN) :: w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
            REAL*8, INTENT(OUT) :: KG0F(3*M*N, 3*M*N)
        END SUBROUTINE

    END INTERFACE

    ! inputs
    CHARACTER BALANC, JOBVL, JOBVR, SENSE
    INTEGER NT, nulls, NUM, M, N
    REAL*8 a, b, r
    REAL*8, ALLOCATABLE :: K0(:, :), KG0(:, :), K02(:, :), KG02(:, :)
    REAL*8, ALLOCATABLE :: ABDs(:, :, :)
    REAL*8, ALLOCATABLE :: y1s(:), y2s(:), Nxxs(:), Nyys(:), Nxys(:)
    REAL*8, ALLOCATABLE :: NxxsCTE(:), NyysCTE(:), NxysCTE(:)
    REAL*8, ALLOCATABLE :: Fxs(:), FxsCTE(:), E1s(:), F1s(:), S1s(:), Jxxs(:)
    REAL*8, ALLOCATABLE :: ys(:), bfs(:), dfs(:)
    REAL*8 u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry
    REAL*8 v1tx, v1rx, v2tx, v2rx, v1ty, v1ry, v2ty, v2ry
    REAL*8 w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry
    INTEGER LDA, LDB, LDZ

    ! workspace
    CHARACTER (LEN=100) :: line
    CHARACTER (LEN=400) :: input_file, output_file
    INTEGER stat, ipanel, istiff, i, j, id, jd, NPANELS, NSTIFF
    INTEGER LWORK
    INTEGER, ALLOCATABLE :: IWORK(:)
    REAL*8, ALLOCATABLE :: WORK(:)
    INTEGER, ALLOCATABLE :: TMP(:)

    ! outputs
    INTEGER Mout, INFO
    INTEGER, ALLOCATABLE :: IFAIL(:)
    REAL*8, ALLOCATABLE :: EIGVALS(:), EIGVECS(:, :)


    BALANC = 'N'
    JOBVL = 'N'
    JOBVR = 'N'
    SENSE = 'N'    

    NUM = 10
    M = 15
    N = 15

    ! Default boundary conditions (simply supported)
    u1tx = 0.
    u1rx = 1.D0
    u2tx = 0.D0
    u2rx = 1.D0
    u1ty = 0.D0
    u1ry = 1.D0
    u2ty = 0.D0
    u2ry = 1.D0
    v1tx = 0.D0
    v1rx = 1.D0
    v2tx = 0.D0
    v2rx = 1.D0
    v1ty = 0.D0
    v1ry = 1.D0
    v2ty = 0.D0
    v2ry = 1.D0
    w1tx = 0.D0
    w1rx = 1.D0
    w2tx = 0.D0
    w2rx = 1.D0
    w1ty = 0.D0
    w1ry = 1.D0
    w2ty = 0.D0
    w2ry = 1.D0

    IF (COMMAND_ARGUMENT_COUNT() .NE. 2) THEN
        STOP "This program should be called as: 'buckling_cpanel_bardell input output'"
    ELSE
        CALL GET_COMMAND_ARGUMENT(1, VALUE=input_file, STATUS=stat)
        CALL GET_COMMAND_ARGUMENT(2, VALUE=output_file, STATUS=stat)
    END IF

    ipanel = 0
    istiff = 0
    OPEN(10, FILE=input_file)
    DO 
        READ(10, *) line
        IF (TRIM(line) == "NUM") READ(10, *) NUM

        IF (TRIM(line) == "NPANELS") THEN
            READ(10, *) NPANELS
            ALLOCATE(ABDs(NPANELS, 6, 6))
            ALLOCATE(y1s(NPANELS))
            ALLOCATE(y2s(NPANELS))
            ALLOCATE(Nxxs(NPANELS))
            ALLOCATE(Nyys(NPANELS))
            ALLOCATE(Nxys(NPANELS))
            ALLOCATE(NxxsCTE(NPANELS))
            ALLOCATE(NyysCTE(NPANELS))
            ALLOCATE(NxysCTE(NPANELS))
        ENDIF

        IF (TRIM(line) == "NSTIFF") THEN
            READ(10, *) NSTIFF
            ALLOCATE(ys(NSTIFF))
            ALLOCATE(bfs(NSTIFF))
            ALLOCATE(dfs(NSTIFF))
            ALLOCATE(E1s(NSTIFF))
            ALLOCATE(F1s(NSTIFF))
            ALLOCATE(S1s(NSTIFF))
            ALLOCATE(Jxxs(NSTIFF))
            ALLOCATE(Fxs(NSTIFF))
            ALLOCATE(FxsCTE(NSTIFF))
        ENDIF

        IF (TRIM(line) == "M") READ(10, *) M
        IF (TRIM(line) == "N") READ(10, *) N
        IF (TRIM(line) == "a") READ(10, *) a
        IF (TRIM(line) == "b") READ(10, *) b
        IF (TRIM(line) == "r") READ(10, *) r

        IF (TRIM(line) == "Panel") THEN
            READ(10, *) ipanel
            ABDs(ipanel, 6, 6) = 0.D0
            y1s(ipanel) = 0.D0
            y2s(ipanel) = 0.D0
            Nxxs(ipanel) = 0.D0
            Nyys(ipanel) = 0.D0
            Nxys(ipanel) = 0.D0
            NxxsCTE(ipanel) = 0.D0
            NyysCTE(ipanel) = 0.D0
            NxysCTE(ipanel) = 0.D0
        ENDIF

        IF (TRIM(line) == "y1") READ(10, *) y1s(ipanel)
        IF (TRIM(line) == "y2") READ(10, *) y2s(ipanel)

        IF (TRIM(line) == "Nxx") READ(10, *) Nxxs(ipanel)
        IF (TRIM(line) == "Nyy") READ(10, *) Nyys(ipanel)
        IF (TRIM(line) == "Nxy") READ(10, *) Nxys(ipanel)

        IF (TRIM(line) == "NxxCTE") READ(10, *) NxxsCTE(ipanel)
        IF (TRIM(line) == "NyyCTE") READ(10, *) NyysCTE(ipanel)
        IF (TRIM(line) == "NxyCTE") READ(10, *) NxysCTE(ipanel)

        IF (TRIM(line) == "A11") READ(10, *) ABDs(ipanel, 1, 1)
        IF (TRIM(line) == "A12") READ(10, *) ABDs(ipanel, 1, 2)
        IF (TRIM(line) == "A16") READ(10, *) ABDs(ipanel, 1, 3)
        IF (TRIM(line) == "A22") READ(10, *) ABDs(ipanel, 2, 2)
        IF (TRIM(line) == "A26") READ(10, *) ABDs(ipanel, 2, 3)
        IF (TRIM(line) == "A66") READ(10, *) ABDs(ipanel, 3, 3)

        IF (TRIM(line) == "B11") READ(10, *) ABDs(ipanel, 1, 4)
        IF (TRIM(line) == "B12") READ(10, *) ABDs(ipanel, 1, 5)
        IF (TRIM(line) == "B16") READ(10, *) ABDs(ipanel, 1, 6)
        IF (TRIM(line) == "B22") READ(10, *) ABDs(ipanel, 2, 5)
        IF (TRIM(line) == "B26") READ(10, *) ABDs(ipanel, 2, 6)
        IF (TRIM(line) == "B66") READ(10, *) ABDs(ipanel, 3, 6)

        IF (TRIM(line) == "D11") READ(10, *) ABDs(ipanel, 4, 4)
        IF (TRIM(line) == "D12") READ(10, *) ABDs(ipanel, 4, 5)
        IF (TRIM(line) == "D16") READ(10, *) ABDs(ipanel, 4, 6)
        IF (TRIM(line) == "D22") READ(10, *) ABDs(ipanel, 5, 5)
        IF (TRIM(line) == "D26") READ(10, *) ABDs(ipanel, 5, 6)
        IF (TRIM(line) == "D66") READ(10, *) ABDs(ipanel, 6, 6)

        IF (TRIM(line) == "Stiffener") THEN
            READ(10, *) istiff
            ys(istiff) = 0.D0
            bfs(istiff) = 0.D0
            dfs(istiff) = 0.D0
            E1s(istiff) = 0.D0
            F1s(istiff) = 0.D0
            S1s(istiff) = 0.D0
            Jxxs(istiff) = 0.D0
            Fxs(istiff) = 0.D0
            FxsCTE(istiff) = 0.D0
        END IF

        IF (TRIM(LINE) == "ys") READ(10, *) ys(istiff)
        IF (TRIM(LINE) == "bf") READ(10, *) bfs(istiff)
        IF (TRIM(LINE) == "df") READ(10, *) dfs(istiff)
        IF (TRIM(LINE) == "E1") READ(10, *) E1s(istiff)
        IF (TRIM(LINE) == "F1") READ(10, *) F1s(istiff)
        IF (TRIM(LINE) == "S1") READ(10, *) S1s(istiff)
        IF (TRIM(LINE) == "Jxx") READ(10, *) Jxxs(istiff)
        IF (TRIM(LINE) == "Fx") READ(10, *) Fxs(istiff)

        IF (TRIM(line) == "u1tx") READ(10, *) u1tx
        IF (TRIM(line) == "u1rx") READ(10, *) u1rx
        IF (TRIM(line) == "u2tx") READ(10, *) u2tx
        IF (TRIM(line) == "u2rx") READ(10, *) u2rx
        IF (TRIM(line) == "u1ty") READ(10, *) u1ty
        IF (TRIM(line) == "u1ry") READ(10, *) u1ry
        IF (TRIM(line) == "u2ty") READ(10, *) u2ty
        IF (TRIM(line) == "u2ry") READ(10, *) u2ry

        IF (TRIM(line) == "v1tx") READ(10, *) v1tx
        IF (TRIM(line) == "v1rx") READ(10, *) v1rx
        IF (TRIM(line) == "v2tx") READ(10, *) v2tx
        IF (TRIM(line) == "v2rx") READ(10, *) v2rx
        IF (TRIM(line) == "v1ty") READ(10, *) v1ty
        IF (TRIM(line) == "v1ry") READ(10, *) v1ry
        IF (TRIM(line) == "v2ty") READ(10, *) v2ty
        IF (TRIM(line) == "v2ry") READ(10, *) v2ry

        IF (TRIM(line) == "w1tx") READ(10, *) w1tx
        IF (TRIM(line) == "w1rx") READ(10, *) w1rx
        IF (TRIM(line) == "w2tx") READ(10, *) w2tx
        IF (TRIM(line) == "w2rx") READ(10, *) w2rx
        IF (TRIM(line) == "w1ty") READ(10, *) w1ty
        IF (TRIM(line) == "w1ry") READ(10, *) w1ry
        IF (TRIM(line) == "w2ty") READ(10, *) w2ty
        IF (TRIM(line) == "w2ry") READ(10, *) w2ry

        IF (TRIM(line) == "END") EXIT       
    END DO

    NT = 3*M*N

    ! allocating arrays
    ALLOCATE(K0(NT, NT))
    ALLOCATE(KG0(NT, NT))

    K0 = 0.D0
    KG0 = 0.D0

    WRITE(*, *) "Calculating matrices for each panel..."
    DO i=1, NPANELS
        WRITE(*, *) "    Calculating panel ", i
        ! constitutive stiffness matrix K0
        CALL CALC_K0Y1Y2(M, N, K0, y1s(i), y2s(i), a, b, r, ABDs(i, :, :), &
                         u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                         v1tx, v1rx, v2tx, v2rx, v1ty, v1ry, v2ty, v2ry, &
                         w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
        ! constante stress state that contributes to K0
        CALL CALC_KG0Y1Y2(M, N, K0, y1s(i), y2s(i), a, b, &
                          NxxsCTE(i), NyysCTE(i), NxysCTE(i), &
                          w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
        ! geometric stiffness matrix
        CALL CALC_KG0Y1Y2(M, N, KG0, y1s(i), y2s(i), a, b, &
                          Nxxs(i), Nyys(i), Nxys(i), &
                          w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    END DO
    WRITE(*, *) "finished!"

    WRITE(*, *) "Calculating matrices for each stiffener..."
    DO i=1, NSTIFF
        WRITE(*, *) "    Calculating stiffener ", i
        ! constitutive stiffness matrix K0
        CALL CALC_K0F(M, N, K0, ys(i), a, b, bfs(i), dfs(i), &
                      E1s(i), F1s(i), S1s(i), Jxxs(i), &
                      u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                      w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
        ! constante stress state that contributes to K0
        CALL CALC_KG0F(M, N, K0, ys(i), FxsCTE(i), a, b, bfs(i), dfs(i),&
                       u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                       w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
        ! geometric stiffness matrix
        CALL CALC_KG0F(M, N, KG0, ys(i), Fxs(i), a, b, bfs(i), dfs(i),&
                       u1tx, u1rx, u2tx, u2rx, u1ty, u1ry, u2ty, u2ry, &
                       w1tx, w1rx, w2tx, w2rx, w1ty, w1ry, w2ty, w2ry)
    END DO
    WRITE(*, *) "finished!"

    ! removing null rows and columns
    ALLOCATE(TMP(NT)) 
    TMP = 0
    WHERE (ABS(SUM(K0, DIM=1)) <= 0.0000001) TMP = 1
    nulls = SUM(TMP)
    WRITE(*, *) "Number of removed cols:", nulls

    ALLOCATE(K02(NT-nulls, NT-nulls))
    ALLOCATE(KG02(NT-nulls, NT-nulls))

    jd = 0
    DO j=1, NT
        IF (TMP(j) == 1) THEN
            jd = jd+1
            CYCLE
        END IF
        id = 0
        DO i=1, NT
            IF (TMP(i) == 1) THEN
                id = id+1
                CYCLE
            END IF
            K02(i-id, j-jd) = K0(i, j)
            KG02(i-id, j-jd) = KG0(i, j)
        END DO
    END DO
    DEALLOCATE(TMP)

    ! allocating arrays
    ALLOCATE(IWORK((NT-nulls)**2))

    ! allocating output arrays
    ALLOCATE(EIGVALS(NT-nulls))
    ALLOCATE(EIGVECS(NT-nulls, NT-nulls))
    ALLOCATE(IFAIL(NT-nulls))

    LDA = NT-nulls
    LDB = NT-nulls
    LDZ = NT-nulls

    WRITE(*, *) "Eigenvalue analysis started..."

    ! signature of eigenvalue solver used:
    !
    ! CALL DSYGVX(ITYPE, JOBZ, RANGE, UPLO, N, A, LDA, B, LDB, &
    !             VL, VU, IL, IU, ABSTOL, Mout, W, Z, LDZ, &
    !             WORK, LWORK, IWORK, IFAIL, INFO)

    EIGVALS = EIGVALS*0

    ! Workspace query
    LWORK = -1
    ALLOCATE(WORK(1))    
    CALL DSYGVX(1, "N", "I", "U", (NT-nulls), KG02, LDB, K02, LDA, &
                -1.D10, 0, 1, NUM, 0., Mout, EIGVALS, EIGVECS, LDZ, &
                WORK, LWORK, IWORK, IFAIL, INFO)
    LWORK = WORK(1)
    DEALLOCATE(WORK)
    ! Eigensolver query
    ALLOCATE(WORK(LWORK))
    CALL DSYGVX(1, "N", "I", "U", (NT-nulls), KG02, LDB, K02, LDA, &
                -1.D10, 0, 1, NUM, 0., Mout, EIGVALS, EIGVECS, LDZ, &
                WORK, LWORK, IWORK, IFAIL, INFO)
    DEALLOCATE(WORK)

    WHERE(EIGVALS /= 0) EIGVALS = -1/EIGVALS

    WRITE(*, *) "Eigenvalue analysis completed!"

    ! Writing eigenvalues
    OPEN(11, FILE=output_file, ACTION="WRITE", STATUS="REPLACE")
    DO i=1, NUM
        WRITE(11, *) EIGVALS(i)
    END DO

    IF (INFO /= 0) THEN
        WRITE(*, *) 'EIGVALS', EIGVALS(1), EIGVALS(2)
        WRITE(*, *) 'Mout', Mout
        WRITE(*, *) 'INFO', NT-nulls, INFO
        !WRITE(*, *) 'IFAIL', IFAIL
        WRITE(*, *) 'MIN(K02), MAX(K02)', MINVAL(K02), MAXVAL(K02)
        WRITE(*, *) 'SUM(K02), SUM(KG02)', SUM(K02), SUM(KG02)
    END IF
    WRITE(*, *) 'EIGVALS', EIGVALS(1), EIGVALS(2)
    WRITE(*, *) 'Mout', Mout
    WRITE(*, *) 'INFO', NT-nulls, INFO
    WRITE(*, *) 'MIN(K02), MAX(K02)', MINVAL(K02), MAXVAL(K02)
    WRITE(*, *) 'SUM(K02), SUM(KG02)', SUM(K02), SUM(KG02)

    DEALLOCATE(K0)
    DEALLOCATE(KG0)
    DEALLOCATE(K02)
    DEALLOCATE(KG02)
    DEALLOCATE(IWORK)
    DEALLOCATE(EIGVALS)
    DEALLOCATE(EIGVECS)
    DEALLOCATE(IFAIL)

END PROGRAM BUCKLING_CPANELBAY_BARDELL
