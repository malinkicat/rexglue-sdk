@echo off
set FXC="C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe"
cd /d "%~dp0"
%FXC% /nologo /T cs_5_1 /E main /O3  /Vn resolve_rt_to_texture_8888_cs /Fh bytecode\d3d12_5_1\resolve_rt_to_texture_8888_cs.h resolve_rt_to_texture_8888.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_DIRECT_MSAA /Vn resolve_rt_to_texture_8888_ms_cs /Fh bytecode\d3d12_5_1\resolve_rt_to_texture_8888_ms_cs.h resolve_rt_to_texture_8888.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3  /Vn resolve_depth_to_texture_cs /Fh bytecode\d3d12_5_1\resolve_depth_to_texture_cs.h resolve_depth_to_texture.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_DIRECT_MSAA /Vn resolve_depth_to_texture_ms_cs /Fh bytecode\d3d12_5_1\resolve_depth_to_texture_ms_cs.h resolve_depth_to_texture.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_DIRECT_FLOAT24 /Vn resolve_depth_to_texture_f24_cs /Fh bytecode\d3d12_5_1\resolve_depth_to_texture_f24_cs.h resolve_depth_to_texture.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_DIRECT_FLOAT24 /D XE_DIRECT_MSAA /Vn resolve_depth_to_texture_f24_ms_cs /Fh bytecode\d3d12_5_1\resolve_depth_to_texture_f24_ms_cs.h resolve_depth_to_texture.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3  /Vn texture_store_32bpb_tiled_cs /Fh bytecode\d3d12_5_1\texture_store_32bpb_tiled_cs.h texture_store_32bpb_tiled.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_STORE_DEPTH24 /Vn texture_store_depth24_tiled_cs /Fh bytecode\d3d12_5_1\texture_store_depth24_tiled_cs.h texture_store_32bpb_tiled.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_STORE_FLOAT32 /Vn texture_store_f32_tiled_cs /Fh bytecode\d3d12_5_1\texture_store_f32_tiled_cs.h texture_store_32bpb_tiled.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3  /Vn texture_store_32bpb_tiled_scaled_cs /Fh bytecode\d3d12_5_1\texture_store_32bpb_tiled_scaled_cs.h texture_store_32bpb_tiled_scaled.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_STORE_DEPTH24 /Vn texture_store_depth24_tiled_scaled_cs /Fh bytecode\d3d12_5_1\texture_store_depth24_tiled_scaled_cs.h texture_store_32bpb_tiled_scaled.cs.hlsl || exit /b 1
%FXC% /nologo /T cs_5_1 /E main /O3 /D XE_STORE_FLOAT32 /Vn texture_store_f32_tiled_scaled_cs /Fh bytecode\d3d12_5_1\texture_store_f32_tiled_scaled_cs.h texture_store_32bpb_tiled_scaled.cs.hlsl || exit /b 1
echo FXC OK
