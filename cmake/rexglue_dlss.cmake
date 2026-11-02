# rexglue_dlss.cmake — Optional NVIDIA DLSS (NGX) integration
#
# Expects REXGLUE_ENABLE_DLSS to be set before inclusion. Uses the DLSS SDK in
# REXGLUE_DLSS_SDK_DIR (include/, lib/x64/nvsdk_ngx_d.lib, bin/nvngx_dlss.dll),
# or downloads exactly those files of NVIDIA's DLSS SDK (pinned commit, checked
# hashes). The SDK is under the NVIDIA RTX SDKs license (LICENSE.txt next to
# it), not this project's: it is only fetched, never committed.
#
# On success, sets REXGLUE_DLSS_INCLUDE_DIR, REXGLUE_DLSS_LIBRARY and
# REXGLUE_DLSS_RUNTIME_DLL (copied next to executables by
# rexglue_configure_target).

unset(REXGLUE_DLSS_INCLUDE_DIR CACHE)
unset(REXGLUE_DLSS_LIBRARY CACHE)
unset(REXGLUE_DLSS_RUNTIME_DLL CACHE)

if(NOT REXGLUE_ENABLE_DLSS)
    return()
endif()

if(NOT WIN32 OR NOT REXGLUE_USE_D3D12)
    message(WARNING "REXGLUE_ENABLE_DLSS needs Windows and the D3D12 backend - disabling DLSS.")
    set(REXGLUE_ENABLE_DLSS OFF CACHE BOOL "" FORCE)
    return()
endif()

set(REXGLUE_DLSS_SDK_DIR "" CACHE PATH
    "NVIDIA DLSS SDK folder (include, lib/x64, bin); empty = download the needed files")

if(NOT REXGLUE_DLSS_SDK_DIR)
    # NVIDIA/DLSS v310.9.1.
    set(_rexglue_dlss_commit 374959484e79a640feaba44c93ac8cfb0a03f5b5)
    set(_rexglue_dlss_base "https://raw.githubusercontent.com/NVIDIA/DLSS/${_rexglue_dlss_commit}")
    set(_rexglue_dlss_dir "${CMAKE_BINARY_DIR}/_deps/dlss_sdk")
    set(_rexglue_dlss_files
        "include/nvsdk_ngx.h|include/nvsdk_ngx.h|dc38e7467cf415379c9d12ae1b6e4a494c453ed92720fb53e92aecb523e7b848"
        "include/nvsdk_ngx_defs.h|include/nvsdk_ngx_defs.h|ea23f33497cd274860d1c25a97644fce807dcb0037c594547203343103fad03e"
        "include/nvsdk_ngx_params.h|include/nvsdk_ngx_params.h|943bc8cc5cdae03b6303016fbad3183636f2335ae27a2d18776798c3b4efabbc"
        "include/nvsdk_ngx_helpers.h|include/nvsdk_ngx_helpers.h|5bcbadfe7478b802cf6d3aca4dc5ddd7d0889b99726e69c63f9e9bd555f44471"
        "include/nvsdk_ngx_helpers_d3d.h|include/nvsdk_ngx_helpers_d3d.h|ec75224f36ed6580baaf250fa83620405ad81a1d98c2c436654a0c9cf6a6b8ba"
        "include/nvsdk_ngx_helpers_cuda.h|include/nvsdk_ngx_helpers_cuda.h|f2851a76107bdf4fbcb7b261f3c35573c312d8ecb4bc18fb02d9e2cec7705042"
        "include/nvsdk_ngx_loader.h|include/nvsdk_ngx_loader.h|abdf5f66a692620538357d29fdf78e3e2a677ee8140b7a638962ecb766585e75"
        "lib/Windows_x86_64/x64/nvsdk_ngx_d.lib|lib/x64/nvsdk_ngx_d.lib|4b6cecad7f1906571c94010241f650e4a5457e64fad49ddaccb82de79f6c2999"
        "lib/Windows_x86_64/rel/nvngx_dlss.dll|bin/nvngx_dlss.dll|3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983"
        "LICENSE.txt|LICENSE.txt|d4216e39ebef5f9b50a6712ebb37beeb5379862a67733a9999c651f21592aaf0"
    )
    foreach(_rexglue_dlss_entry ${_rexglue_dlss_files})
        string(REPLACE "|" ";" _rexglue_dlss_parts "${_rexglue_dlss_entry}")
        list(GET _rexglue_dlss_parts 0 _rexglue_dlss_source)
        list(GET _rexglue_dlss_parts 1 _rexglue_dlss_dest)
        list(GET _rexglue_dlss_parts 2 _rexglue_dlss_hash)
        set(_rexglue_dlss_path "${_rexglue_dlss_dir}/${_rexglue_dlss_dest}")
        if(EXISTS "${_rexglue_dlss_path}")
            file(SHA256 "${_rexglue_dlss_path}" _rexglue_dlss_existing_hash)
            if(_rexglue_dlss_existing_hash STREQUAL _rexglue_dlss_hash)
                continue()
            endif()
        endif()
        message(STATUS "Downloading DLSS SDK file ${_rexglue_dlss_source}")
        file(DOWNLOAD "${_rexglue_dlss_base}/${_rexglue_dlss_source}" "${_rexglue_dlss_path}"
             EXPECTED_HASH SHA256=${_rexglue_dlss_hash} STATUS _rexglue_dlss_status)
        list(GET _rexglue_dlss_status 0 _rexglue_dlss_status_code)
        if(NOT _rexglue_dlss_status_code EQUAL 0)
            message(WARNING "Failed to download ${_rexglue_dlss_source} (${_rexglue_dlss_status}) - disabling DLSS.")
            set(REXGLUE_ENABLE_DLSS OFF CACHE BOOL "" FORCE)
            return()
        endif()
    endforeach()
    set(_rexglue_dlss_sdk "${_rexglue_dlss_dir}")
else()
    set(_rexglue_dlss_sdk "${REXGLUE_DLSS_SDK_DIR}")
endif()

foreach(_rexglue_dlss_required include/nvsdk_ngx_helpers.h lib/x64/nvsdk_ngx_d.lib bin/nvngx_dlss.dll)
    if(NOT EXISTS "${_rexglue_dlss_sdk}/${_rexglue_dlss_required}")
        message(WARNING "DLSS SDK file ${_rexglue_dlss_sdk}/${_rexglue_dlss_required} is missing - disabling DLSS.")
        set(REXGLUE_ENABLE_DLSS OFF CACHE BOOL "" FORCE)
        return()
    endif()
endforeach()

set(REXGLUE_DLSS_INCLUDE_DIR "${_rexglue_dlss_sdk}/include" CACHE INTERNAL "DLSS SDK headers")
set(REXGLUE_DLSS_LIBRARY "${_rexglue_dlss_sdk}/lib/x64/nvsdk_ngx_d.lib" CACHE INTERNAL "NGX static library")
set(REXGLUE_DLSS_RUNTIME_DLL "${_rexglue_dlss_sdk}/bin/nvngx_dlss.dll" CACHE INTERNAL "DLSS runtime")
message(STATUS "DLSS SDK: ${_rexglue_dlss_sdk}")
