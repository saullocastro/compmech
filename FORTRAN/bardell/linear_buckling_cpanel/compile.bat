compilervars.sh intel64
del *.obj
del *.exe
ifort buckling_cpanel_bardell.f90 /check /Qmkl=parallel -O3 /Qxhost -o buckling_cpanel_bardell.exe
