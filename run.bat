@echo off

set NVCC=nvcc
set FLAGS=-arch sm_75 -o dora -O3 -std c++20 --threads 0 --use_fast_math -Xcompiler "/O2 /MT /MP"
set SRC=..\src\acts.cpp ..\src\arrs.cpp ..\src\datas.cpp ..\src\linalg.cu ..\src\losses.cpp ..\src\lyrs.cpp ..\src\main.cpp ..\src\math.cu ..\src\rand.cpp ..\src\strings.cpp ..\src\tensor.cpp

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