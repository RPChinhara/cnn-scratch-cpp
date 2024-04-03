@echo off

set NVCC=nvcc
set FLAGS=-I ..\src -o dora -arch sm_75 --optimize 3 -std c++20 --threads 0 --use_fast_math -x cu -Xcompiler /GL -Xcompiler /Gm- -Xcompiler /MP -Xcompiler /Z7 -Xcompiler /O2
set SRC=..\src\act.cpp ^
        ..\src\arrs.cpp ^
        ..\src\datasets\engspa.cpp ^
        ..\src\datasets\imdb.cpp ^
        ..\src\datasets\iris.cpp ^
        ..\src\datasets\mnist.cpp ^
        ..\src\datasets\tripadvisor.cpp ^
        ..\src\diffs.cpp ^
        ..\src\knls.cpp ^
        ..\src\linalg.cpp ^
        ..\src\losses.cpp ^
        ..\src\main.cpp ^
        ..\src\math.cpp ^
        ..\src\metrics.cpp ^
        ..\src\models\cnn2d.cpp ^
        ..\src\models\nn.cpp ^
        ..\src\models\ql.cpp ^
        ..\src\models\trans.cpp ^
        ..\src\preproc.cpp ^
        ..\src\rand.cpp ^
        ..\src\ten.cpp

if not exist bin mkdir bin

pushd bin
if not defined DevEnvDir (call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
%NVCC% %FLAGS% %SRC%
popd