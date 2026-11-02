@echo off
rem SMAA 1x compute passes (smaa.cs.hlsl, thirdparty\smaa\SMAA.hlsl - MIT).
set FXC="C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe"
cd /d "%~dp0"
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_SMAA_EDGES /Vn smaa_edges_cs /Fh bytecode\d3d12_5_1\smaa_edges_cs.h smaa.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_SMAA_WEIGHTS /Vn smaa_weights_cs /Fh bytecode\d3d12_5_1\smaa_weights_cs.h smaa.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_SMAA_BLEND /Vn smaa_blend_cs /Fh bytecode\d3d12_5_1\smaa_blend_cs.h smaa.cs.hlsl || exit /b 1
echo FXC OK
