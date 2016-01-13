source /nfs/cae/Ferramentas/EXEC/INTEL/Intel_Fortran_13.1/composer_xe_2013.5.192/bin/compilervars.sh intel64
icc -c ../../../C/src/bardell_12_integral_ff.c -o ./bardell_12_integral_ff.obj
icc -c ../../../C/src/bardell_12_integral_ffxi.c -o ./bardell_12_integral_ffxi.obj
icc -c ../../../C/src/bardell_12_integral_ffxixi.c -o ./bardell_12_integral_ffxixi.obj
icc -c ../../../C/src/bardell_12_integral_fxifxi.c -o ./bardell_12_integral_fxifxi.obj
icc -c ../../../C/src/bardell_12_integral_fxifxixi.c -o ./bardell_12_integral_fxifxixi.obj
icc -c ../../../C/src/bardell_12_integral_fxixifxixi.c -o ./bardell_12_integral_fxixifxixi.obj
