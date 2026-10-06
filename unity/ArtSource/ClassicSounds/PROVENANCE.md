# Classic speaker effects

These are the three unmodified sound-program files from the Crystal Caves
Episode 1 shareware data, mirrored by
[OpenCrystalCaves](https://github.com/OpenCrystalCaves/OpenCrystalCaves/tree/master/media/CC1).
They remain original Crystal Caves game material, copyright 1991 Apogee / Peder
Jungck; the fan engine's MIT license does not relicense the game assets.

Each file contains twelve 610-byte records: frequency commands, terminator,
priority and rapid speaker gating. `src/unity_bridge/classic_audio.py` renders
them into mono PCM at export time using the DOS sound-service timer divisor.
[Format research](https://moddingwiki.shikadi.net/wiki/Crystal_Caves_Sound_format).

No LGR commentary, video recording or HD soundtrack is included in the player.
