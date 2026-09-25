@echo off
rem Build stems.cpp for one backend into its own build dir (backends coexist for A/B).
rem
rem Usage: build.cmd [cpu^|cuda^|vulkan^|all]   (default: cpu)
setlocal EnableDelayedExpansion

set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cpu"

rem locate cmake (PATH, else the copy bundled with Visual Studio 2022)
where cmake >nul 2>nul
if errorlevel 1 (
    set "VSCMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    if exist "!VSCMAKE!\cmake.exe" set "PATH=!VSCMAKE!;%PATH%"
)

set "GEN=-G "Visual Studio 17 2022" -A x64"

if /i "%BACKEND%"=="cpu" (
    set "DIR=build" & set "FLAGS=-DGGML_CUDA=OFF"
) else if /i "%BACKEND%"=="cuda" (
    set "DIR=build-cuda" & set "FLAGS=-DSTEMS_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
) else if /i "%BACKEND%"=="vulkan" (
    set "DIR=build-vulkan" & set "FLAGS=-DSTEMS_VULKAN=ON"
    if "%VULKAN_SDK%"=="" echo [stems] WARNING: VULKAN_SDK not set - install the Vulkan SDK and open a fresh shell.
) else if /i "%BACKEND%"=="all" (
    set "DIR=build-all" & set "FLAGS=-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DSTEMS_CUDA=ON -DSTEMS_VULKAN=ON"
) else (
    echo unknown backend: %BACKEND%  ^(cpu^|cuda^|vulkan^|all^)& exit /b 1
)

echo [stems] configuring %BACKEND% -^> %DIR%\
cmake -S . -B %DIR% %GEN% %FLAGS% || exit /b 1
if "%JOBS%"=="" set JOBS=4
cmake --build %DIR% --config Release --parallel %JOBS% || exit /b 1
echo [stems] built %BACKEND% -^> %DIR%\bin\Release\
