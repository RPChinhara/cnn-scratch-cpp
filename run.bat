@echo off

set NVCC=nvcc
set FLAGS=-arch sm_75 -O3 -std c++20 --use_fast_math --threads 0 -Xcompiler "/MP /MT /O2" -o dora
set SRC=..\src\acts.cpp ..\src\arrs.cpp ..\src\datasets.cpp ..\src\input_handler.cpp ..\src\linalg.cu ..\src\losses.cpp ..\src\lyrs.cpp ..\src\main.cpp ..\src\math.cu ..\src\mesh.cpp ..\src\rand.cpp ..\src\renderer.cpp ..\src\scene.cpp ..\src\strings.cpp ..\src\tensor.cpp ..\src\window.cpp

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