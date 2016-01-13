compilervars.sh intel64
ifort buckling_cpanelbay_bardell.f90 /Qmkl=parallel -O3 /Qxhost -o buckling_cpanelbay_bardell.exe
