# Rung 3: O_2, the trick-2 oracle (values the start of trick 2; used by H_2 for trick-1 decisions)

## O_2 provisional, round 1 — out of spec (2026-10-18)

From the trick-2 audit loop `m3-hybrid-t2-loop-1` (manifest `docs/vast/m3-hybrid-t2-loop.manifest.json`, binaries
`ebc5d7a`; AMD EPYC 7742, 245 threads, RTX 4060 Ti 16 GB, $0.72/h). Promoted as provisional on the owner's word
(2026-10-17: "promote t2's first network trained as is ... later the in-spec t2 network gets shipped"). It is **not**
proven to the 0.01% standard: no trick-1 golden set exists (owner, 2026-10-16), the loop's own audit and rough test
come later in the run, and the silver set (`docs/m3-hybrid/silver.md`) is the planned check at the end.

- **Integer network:** `inputs/sha256/505a0c7c65147ffa9565e06059f47664ea9f1bfdf0c5ea2f8d392b0e6e1337b0`
  (16,951,680 bytes), 2048-1024-256, outcome head (39 outcomes at trick 2). Float twin in the run's partial upload
  (`runs/m3-hybrid-t2-loop-1/partial/rung/000000/out/hybrid/net/float.safetensors`).
- **Data:** 1.2M random-play + 300k rules-v1 exact trick-2 labels (`inputs/sha256/fb51c217…` 2.26 GB,
  `inputs/sha256/2160fefc…` 0.56 GB), validation 50k (s601). Trained in 1.8 min on the GPU; best epoch 3 of 9
  (early stop, patience 6).
- **Validation (float):** loss 2.0456, outcome accuracy 68.2%, value error SD 3.62% of the money chance per position,
  mean |error| 1.46%, team bias within ±0.12%. Integer twin on 20,000 rows: SD 3.75%, bias within ±0.14%.
- Every later round the run ranks best replaces this pointer; the run's end report and the silver set decide the
  adopted O_2.
