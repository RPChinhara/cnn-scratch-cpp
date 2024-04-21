@echo off

set NVCC=nvcc
set FLAGS=-arch sm_75 -I ..\src -o dora --optimize 3 -std c++20 --threads 0 --use_fast_math -Xcompiler /GL -Xcompiler /MP -Xcompiler /O2 -Xcompiler /utf-8
set SRC=..\src\arrs.cpp ..\src\datas\enes.cpp ..\src\knls.cu ..\src\main.cpp ..\src\math.cu ..\src\preproc.cpp ..\src\rd.cpp ..\src\ten.cpp ..\src\mdls\nn.cpp ..\src\losses.cpp ..\src\diffs.cpp ..\src\act.cu ..\src\datas\iris.cpp ..\src\linalg.cu ..\src\mets.cpp

if not exist bin mkdir bin

pushd bin
if not defined DevEnvDir (call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
%NVCC% %FLAGS% %SRC%
popd

if %ERRORLEVEL% == 0 (
    echo Compilation successful, running the program...
    bin\dora.exe
) else (
    echo Compilation failed with error %ERRORLEVEL%.
)