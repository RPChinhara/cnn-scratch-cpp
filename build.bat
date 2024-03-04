@echo off

set NVCC=nvcc
set FLAGS=-I ..\src -o dora -arch sm_75 --optimize 3 -std c++20 --threads 0 --use_fast_math -x cu -Xcompiler /GL -Xcompiler /Gm- -Xcompiler /MP -Xcompiler /Z7 -Xcompiler /O2
set SRC=..\src\activation.cpp ^
        ..\src\array.cpp ^
        ..\src\cnn.cpp ^
        ..\src\dataset.cpp ^
        ..\src\derivative.cpp ^
        ..\src\environment.cpp ^
        ..\src\kernel.cpp ^
        ..\src\linalg.cpp ^
        ..\src\loss.cpp ^
        ..\src\main.cpp ^
        ..\src\mathematics.cpp ^
        ..\src\metric.cpp ^
        ..\src\nn.cpp ^
        ..\src\physics.cpp ^
        ..\src\preprocessing.cpp ^
        ..\src\q_learning.cpp ^
        ..\src\random.cpp ^
        ..\src\tensor.cpp ^

if not exist bin mkdir bin

pushd bin
if not defined DevEnvDir (call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
%NVCC% %FLAGS% %SRC%
popd