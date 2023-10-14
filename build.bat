@echo off

set NVCC=nvcc
set FLAGS=-I ..\src -o dora -lUser32 -arch sm_75 --optimize 3 -std c++17 --threads 0 --use_fast_math -x cu -Xcompiler /GL -Xcompiler /Gm- -Xcompiler /MP -Xcompiler /Ox -Xcompiler /Z7
set SRC=..\src\activations.cpp ^
        ..\src\arrays.cpp ^
        ..\src\datasets.cpp ^
        ..\src\derivatives.cpp ^
        ..\src\initializers.cpp ^
        ..\src\kernels.cpp ^
        ..\src\linalg.cpp ^
        ..\src\losses.cpp ^
        ..\src\main.cpp ^
        ..\src\mathematics.cpp ^
        ..\src\metrics.cpp ^
        ..\src\nn.cpp ^
        ..\src\preprocessing.cpp ^
        ..\src\q_learning.cpp ^
        ..\src\random.cpp ^
        ..\src\regularizations.cpp ^
        ..\src\tensor.cpp ^
        ..\src\window.cpp

if not exist bin mkdir bin
pushd bin
if not defined DevEnvDir (call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
%NVCC% %FLAGS% %SRC%
popd