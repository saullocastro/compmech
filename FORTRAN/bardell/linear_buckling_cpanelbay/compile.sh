source /nfs/cae/Ferramentas/EXEC/INTEL/Intel_Fortran_13.1/composer_xe_2013.5.192/bin/compilervars.sh intel64
#ifort bardell_12_integral_ff.obj bardell_12_integral_ffxi.obj bardell_12_integral_ffxixi.obj bardell_12_integral_fxifxi.obj bardell_12_integral_fxifxixi.obj bardell_12_integral_fxixifxixi.obj buckling_cpanelbay_bardell.f90 -mkl=parallel -static-intel -O3 -xhost -o buckling_cpanelbay_bardell
ifort buckling_cpanelbay_bardell.f90 -mkl=parallel -static-intel -O3 -xhost -o buckling_cpanelbay_bardell
