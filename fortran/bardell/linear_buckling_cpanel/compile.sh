source /nfs/cae/Ferramentas/EXEC/INTEL/Intel_Fortran_13.1/composer_xe_2013.5.192/bin/compilervars.sh intel64
ifort "$@" -mkl=parallel -static-intel -O3 -xhost
