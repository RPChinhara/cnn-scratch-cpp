@echo off

set NVCC=nvcc
set FLAGS=-I ..\src -o dora -arch sm_75 --optimize 3 -std c++20 --threads 0 --use_fast_math -x cu -Xcompiler /GL -Xcompiler /Gm- -Xcompiler /MP -Xcompiler /Z7 -Xcompiler /O2
set SRC=..\src\act.cpp ^
        ..\src\arrs.cpp ^
        ..\src\datasets\enes.cpp ^
        ..\src\datasets\imdb.cpp ^
        ..\src\datasets\iris.cpp ^
        ..\src\datasets\mnist.cpp ^
        ..\src\datasets\ta.cpp ^
        ..\src\diffs.cpp ^
        ..\src\knls.cpp ^
        ..\src\linalg.cpp ^
        ..\src\losses.cpp ^
        ..\src\main.cpp ^
        ..\src\math.cpp ^
        ..\src\metrics.cpp ^
        ..\src\mdls\cnn2d.cpp ^
        ..\src\mdls\nn.cpp ^
        ..\src\mdls\ql.cpp ^
        ..\src\mdls\trans.cpp ^
        ..\src\preproc.cpp ^
        ..\src\rand.cpp ^
        ..\src\ten.cpp

if not exist bin mkdir bin

pushd bin
if not defined DevEnvDir (call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
%NVCC% %FLAGS% %SRC%
popd