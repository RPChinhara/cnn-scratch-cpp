@echo off

set NVCC=nvcc
set FLAGS=-I ..\src -o dora -arch sm_75 --optimize 3 -std c++20 --threads 0 --use_fast_math -Xcompiler /GL -Xcompiler /Gm- -Xcompiler /MP -Xcompiler /Z7 -Xcompiler /O2 -Xcompiler /utf-8
set SRC=..\src\act.cu ^
        ..\src\arrs.cpp ^
        ..\src\datas\enes.cpp ^
        ..\src\datas\imdb.cpp ^
        ..\src\datas\iris.cpp ^
        ..\src\datas\mnist.cpp ^
        ..\src\datas\ta.cpp ^
        ..\src\diffs.cpp ^
        ..\src\knls.cu ^
        ..\src\linalg.cu ^
        ..\src\losses.cpp ^
        ..\src\main.cpp ^
        ..\src\math.cu ^
        ..\src\mdls\cnn2d.cpp ^
        ..\src\mdls\nn.cpp ^
        ..\src\mdls\ql.cpp ^
        ..\src\mdls\trans.cpp ^
        ..\src\metrics.cpp ^
        ..\src\preproc.cpp ^
        ..\src\rand.cpp ^
        ..\src\ten.cpp

if not exist bin mkdir bin

pushd bin
if not defined DevEnvDir (call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
%NVCC% %FLAGS% %SRC%
popd