# ReAmped 3 changes to ReXGlue

This fork of [rexglue-sdk](https://github.com/rexglue/rexglue-sdk) is the runtime for ReAmped 3, the Amped 3 recompilation. Same licence as ReXGlue (BSD 3-Clause, see LICENSE). Most changes are general; the Amped 3 specific ones are marked as such in the code and are off unless the game turns them on.

## GPU (Direct3D 12)
- Single-pass "tile collapse" (`gpu_tile_collapse`): games that draw each frame in strips to fit the 360's 10 MB EDRAM are drawn once. About a third of the draw work.
- MSAA, D24FS8, k_24_8_FLOAT and strip resolves done directly, clears in place (ported from iExplosiveRage's fork, commits 5281f5d and 6aaf31a, then extended).
- Command processor: optional second thread (`gpu_cp_split`, on under Wine/Proton), cheaper register writes, bulk side-effect-free register runs, constant gathering, texture/sampler caching, pipeline reuse, hot-page tracking, presenting from the submission thread.
- SMAA 1x post-process (thirdparty/smaa, MIT), AMD FSR/CAS, ultrawide and 16:10 squeeze, UI scale.
- Texture replacement: XXH3-keyed lookups, background decoding, alpha handling (based on iExplosiveRage's burstlimit branch).
- NVIDIA DLAA and FSR 3 frame generation: in the code, off by default (they need per-object motion vectors).

## Audio
- Your own music player (FLAC/MP3/WAV via dr_libs, Ogg via stb_vorbis) that holds while the title's soundtrack is silent or event music plays.
- XMA loop points read correctly (looping sounds now loop).

## System
- Data overlay for the virtual file system (patched game files without touching the game folder).
- Controller on-screen keyboard for text entry.
- Blocking timer-queue wait under Wine (idea from Project8Recomp).
- Mouse/keyboard and XMP (console music player) fixes.

## Building
Configure with the `win-amd64` preset and `-DREXGLUE_ENABLE_FIDELITYFX=ON` (optionally `-DREXGLUE_ENABLE_DLSS=ON`, which downloads NVIDIA's SDK and never commits it). ReAmped 3's `tools/build_sdk_ffx.cmd` shows the exact flags.
