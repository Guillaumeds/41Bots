# Choosing vast.ai machines for our CPU-heavy jobs

Reference table and ranking, first written 2026-10-14 (market snapshot 13:02 UTC). Nothing was rented to make it: it combines a read-only look at the vast.ai market, published benchmark results and our own runs so far. Refresh it with one command (see "How to refresh").

## The short answer

- **The price on vast.ai is set by the graphics cards, not by the processor.** We do not use the graphics cards, so the bargains are machines with a big processor and a cheap card. Today the price of one hardware thread [one of the two "virtual cores" each physical core runs; vast.ai calls them "effective cores"] varies about 30-fold between offers, while the speed of one thread varies only about 3-fold between the processor types on offer. **So the price matters more than the processor model**, and the right machine changes from hour to hour. Always rank the live market; never rent a fixed model by habit.
- **Cheapest work right now, for every job except exact trick-1 solves:** an AMD EPYC 9965 machine (768 threads, 503 GB, $2.40/h). By our estimate it does about 3 times more work per dollar than any other machine our launcher may rent, and it is also the fastest machine on offer, so the owner's rule (fastest within +20% of the cheapest) also picks it. Next: EPYC 7B13 (128 threads, $1.00/h), EPYC 9V74 (320 threads, $4.01/h).
- **Exact trick-1 solves** also need a lot of memory (a single deal used up to 65 GB of table). Among machines with at least 480 GB, the cheapest work today is the EPYC 7B13 and EPYC 9V74 boxes. The EPYC 9734 machine running today's trick-1 measurement (448 threads, 1.5 TB, $6.43/h) is a sound choice: about 35–45% dearer per unit of work than the best today, but it can hold three times as many deals in memory at once.
- **These rankings are estimates until we measure.** Each one is based on published benchmark results, not our own program. The biggest unknown is not the hardware. It is how well one trick-1 deal spreads over hundreds of threads. One deal on 368 threads ran each thread about 7 times slower than one deal on 4 threads. Two short benchmarks (below) would settle the main questions for roughly $10–15.

## What we are buying

- **Threads, not cores.** vast.ai's "effective cores" are hardware threads: on almost every machine, each physical core runs two threads (SMT [simultaneous multithreading: two instruction streams sharing one core, which together typically run 10–30% faster than one alone]). Every speed in this document is **per thread with every thread of the machine busy**, which is how we run: one deal per thread.
- **We get 96% of them.** Every container we have rented so far was given a CPU quota [a cap set by the host on how much processor time the container may use] of 96% of the advertised threads (61.4 of 64, 76.8 of 80, 122.9 of 128, 368.6 of 384). One host is worse: the engineering-sample machine 137718 gave us **192 of 384** (and its 251 GB of memory was too small for a trick-1 run, which was killed). The tool knows this machine.
- **The launcher's rules** (`python/launcher/launch.py`): on-demand price only [a fixed hourly price; the machine cannot be taken back, unlike cheaper "interruptible" offers], verified hosts, reliability above 0.98, the whole machine (no sharing with other renters), network price at most $0.02/GB. Only offers that pass all of these are ranked below; offers refused for network price or sharing are listed separately, because two of today's cheapest machines were refused for exactly those reasons.
- **One unit of work** = one hour of one thread of an AMD EPYC 7502 (our measured reference machine, `docs/m2-epyc/`) with all threads busy. "$ per 1k work" is the price of 1,000 such units. Lower is better.

## How the speed of each processor was estimated

Our solver is a branchy game-tree search on whole numbers, with transposition tables [a memory of positions already solved, so the same position reached by another order of play is solved once]. Two of the ten programs in SPECrate2017_int [the industry-standard processor benchmark; "rate" means one copy runs on every thread at once, as we do] are game-tree searches of the same kind:

- **531.deepsjeng_r**: a chess program, a tree search with a large hash table, like ours;
- **541.leela_r**: a Go program, a tree search.

For each processor we took published SPEC results (one or two per model, nearest the median of all published results for that model), and divided the speed of one copy by the same figure for the EPYC 7502:

- **Compute speed** = the average of deepsjeng and leela. It is used for every job whose tables stay small: exact solves from trick 2 on (tables of 0.1–0.5 GB), equity sweeps, rollouts and tournaments.
- **TT speed** also averages in 505.mcf_r and 520.omnetpp_r, two programs limited by memory delays. It is used for exact trick-1 solves, whose tables of tens of GB are far bigger than any cache [the processor's small, fast on-chip memory], so every table lookup waits for main memory.

Where a processor has no SPEC result (cloud-only variants such as the 7B13, 9V74, 9J14; workstation chips such as Threadripper; engineering samples [pre-release chips identified only by a part number]), it is mapped to the nearest published chip of the same design, adjusted for its clock speed, and given a wider uncertainty. Every figure, its source and its reasoning are in `docs/vast/speed-factors.json`.

**How sure are we?** About ±15% for chips with SPEC results, ±20–25% for mapped chips, ±35–40% for engineering samples. SPEC uses vendor-tuned compilers; our program is compiled for any recent x86 processor, without vendor tuning, so its absolute speed differs, but the *ratios* between machines should carry over within those margins. Vast.ai hosts are also less well set up than SPEC's test machines (fewer memory sticks, older settings), which the measurements will reveal.

### What our own runs say (calibration)

| Run | Machine | What it shows |
|---|---|---|
| m2-spike-epyc | EPYC 7502, 61 threads, $0.34/h | Same speed per thread as our 4-core Xeon development machine on identical worlds when lightly loaded (trick 3: 8.8 ms median on both; trick 2: 2.8 s vs 3.1 s). This is the reference (1.00). |
| m2-trick1-1 | EPYC 9654, 368 threads | Four trick-1 deals, each on all 368 threads: 7, 32, 140, 25 s. That is about 0.5 million positions per second per thread, against 3.6 million on 4 threads of the Xeon development machine. **One deal spread over hundreds of threads wastes about six sevenths of each thread.** Running several deals side by side with fewer threads each should be much more efficient, if memory allows. |
| m3-agree-t2d, -t3b, -t4c | EPYC 9654 (368), Xeon E5-2673 v4 (76), EPYC 7J13 (122) | Different tricks on different machines, so they cannot be compared with each other. Leader decisions are the slowest at every trick (median time per world, leader vs 8th player: trick 2 on the 9654, 12 s vs 0.05 s; trick 3 on the Xeon, 42 ms vs 0.4 ms; trick 4 on the 7J13, 0.13 ms vs 0.03 ms). So a benchmark must use a fixed set of decisions with all eight positions. |
| m3-agree-t1-b (today, running) | EPYC 9734, 448 threads, 1.5 TB, $6.43/h | about 1.2 trick-1 leader worlds per minute with 4 deals × 112 threads: about 72 worlds per hour, **11 worlds per dollar** |
| m3-agree-t1-a5 (today, running) | EPYC 9754, 512 threads, 1 TB, $11.24/h | about 0.7 worlds per minute on different (possibly harder) decisions: 42 per hour, **3.7 worlds per dollar** |

**The 9734 vs 9754 comparison is not a fair test.** SPEC rates the two processors' threads within 2% of each other (1.18 vs 1.16). The 9754 box has 14% more threads and costs 53% more per thread-hour, so on equal work the 9734 box should do about 1.5 times more worlds per dollar. It did 3 times more. The remaining factor of about 2 most likely comes from the decisions: trick-1 worlds vary 20-fold in cost (7 s to 140 s for the four worlds above), and the two machines were solving different decisions. It could also be the host (memory layout, power settings). Only the same decisions on both machines can separate the two.

**Bottom line:** no run so far has measured the *same* work on two machine types with every thread busy. Every speed in the tables is still an estimate.

## Table 1: speed per thread of each processor seen on vast.ai today

Speeds are per thread with all threads busy, EPYC 7502 = 1.00; higher is faster. "L3 per core" is the largest on-chip cache divided by the number of cores; "dense" chips (Zen 4c, Zen 5c) have half as much per core as the standard ones. "Memory per socket": a socket is one processor chip (big servers have two); "12 ch DDR5-4800" means 12 memory channels [parallel links between the chip and its memory sticks; more channels move more data] of DDR5 memory at 4,800 million transfers per second. Zen 2 to Zen 5 are AMD's processor designs from 2019 to 2024; Intel's are named after places (Cascade Lake, Sapphire Rapids, Emerald Rapids, Granite Rapids).

| CPU model | Architecture | L3 per core | Memory per socket | Compute speed | TT speed | ± | How estimated |
|---|---|---|---|---|---|---|---|
| AMD Threadripper 9970X | Zen 5 Threadripper (2025) | 4 MB | 4 ch DDR5-6400 (workstation socket) | 1.95 | 1.55 | 25% | estimate |
| AMD Threadripper PRO 9975WX | Zen 5 Threadripper (2025) | 4 MB | 8 ch DDR5-6400 (workstation socket) | 1.95 | 1.80 | 25% | estimate |
| AMD EPYC 9135 | Zen 5 (Turin, 2024) | 4 MB | 12 ch DDR5-6400 | 1.83 | 2.12 | 15% | SPEC results |
| AMD EPYC 9355 | Zen 5 (Turin, 2024) | 8 MB | 12 ch DDR5-6400 | 1.83 | 2.12 | 15% | SPEC results |
| AMD Threadripper 7970X | Zen 4 Threadripper (2023) | 4 MB | 4 ch DDR5-5200 (workstation socket) | 1.80 | 1.45 | 25% | estimate |
| AMD Threadripper PRO 7975WX | Zen 4 Threadripper (2023) | 4 MB | 8 ch DDR5-5200 (workstation socket) | 1.80 | 1.65 | 25% | estimate |
| AMD EPYC 9555 | Zen 5 (Turin, 2024) | 4 MB | 12 ch DDR5-6400 | 1.65 | 1.83 | 15% | SPEC results |
| AMD EPYC 9575F | Zen 5 (Turin, 2024) | 4 MB | 12 ch DDR5-6400 | 1.63 | 1.81 | 15% | SPEC results |
| AMD EPYC 9334 | Zen 4 (Genoa, 2022) | 4 MB | 12 ch DDR5-4800 | 1.53 | 1.71 | 15% | SPEC results |
| AMD EPYC 9454 | Zen 4 (Genoa, 2022) | 5.33 MB | 12 ch DDR5-4800 | 1.48 | 1.63 | 15% | SPEC results |
| AMD EPYC 9554 | Zen 4 (Genoa, 2022) | 4 MB | 12 ch DDR5-4800 | 1.47 | 1.55 | 15% | SPEC results |
| AMD EPYC 9124 | Zen 4 (Genoa, 2022) | 4 MB | 12 ch DDR5-4800 | 1.46 | 1.68 | 15% | SPEC results |
| AMD Threadripper PRO 9995WX | Zen 5 Threadripper (2025) | 4 MB | 8 ch DDR5-6400 (workstation socket) | 1.45 | 1.30 | 30% | estimate |
| AMD Threadripper PRO 5975WX | Zen 3 Threadripper (2022) | 4 MB | 8 ch DDR4-3200 (workstation socket) | 1.40 | 1.35 | 25% | estimate |
| Intel Xeon 6767P | Intel Granite Rapids (Xeon 6 P-core, 2024-25) | 5.25 MB | 8 ch DDR5-6400 (6700P) | 1.38 | 1.50 | 25% | estimate |
| Intel Xeon 6776P | Intel Granite Rapids (Xeon 6 P-core, 2024-25) | 5.25 MB | 8 ch DDR5-6400 (6700P) | 1.38 | 1.50 | 25% | estimate |
| AMD EPYC 9655 | Zen 5 (Turin, 2024) | 4 MB | 12 ch DDR5-6400 | 1.36 | 1.51 | 15% | SPEC results |
| AMD EPYC 9755 | Zen 5 (Turin, 2024) | 4 MB | 12 ch DDR5-6400 | 1.35 | 1.39 | 15% | SPEC results |
| AMD EPYC 9B45 (cloud) | Zen 5 (Turin, 2024) | 4 MB | 12 ch DDR5-6400 | 1.35 | 1.39 | 35% | SPEC results of EPYC 9755 |
| AMD EPYC 9V74 (Azure) | Zen 4 (Genoa, 2022) | 4 MB | 12 ch DDR5-4800 | 1.35 | 1.40 | 25% | estimate |
| AMD Threadripper PRO 7995WX | Zen 4 Threadripper (2023) | 4 MB | 8 ch DDR5-5200 (workstation socket) | 1.30 | 1.15 | 25% | estimate |
| AMD EPYC 9654 | Zen 4 (Genoa, 2022) | 4 MB | 12 ch DDR5-4800 | 1.28 | 1.33 | 15% | SPEC results |
| AMD EPYC 9684X | Zen 4 + 3D V-Cache (Genoa-X, 2023) | 12 MB | 12 ch DDR5-4800 | 1.28 | 1.37 | 15% | SPEC results |
| AMD EPYC 9B14 (cloud) | Zen 4 (Genoa, 2022) | 4 MB | 12 ch DDR5-4800 | 1.28 | 1.33 | 20% | SPEC results of EPYC 9654 |
| AMD EPYC 9J14 (Oracle) | Zen 4 (Genoa, 2022) | 4 MB | 12 ch DDR5-4800 | 1.28 | 1.33 | 20% | SPEC results of EPYC 9654 |
| AMD EPYC 7543 | Zen 3 (Milan, 2021) | 8 MB | 8 ch DDR4-3200 | 1.26 | 1.31 | 15% | SPEC results |
| AMD EPYC 9965 | Zen 5c dense (Turin Dense, 2024) | 2 MB | 12 ch DDR5-6400 | 1.24 | 1.09 | 15% | SPEC results |
| AMD Eng Sample 100-000001535 (Turin ES?) | Zen 5 (Turin, 2024) | 4 MB | 12 ch DDR5-6400 | 1.20 | 1.25 | 35% | estimate |
| AMD EPYC 9734 | Zen 4c dense (Bergamo, 2023) | 2.29 MB | 12 ch DDR5-4800 | 1.18 | 1.20 | 15% | SPEC results |
| AMD EPYC 7513 | Zen 3 (Milan, 2021) | 4 MB | 8 ch DDR4-3200 | 1.17 | 1.19 | 15% | SPEC results |
| AMD EPYC 9754 | Zen 4c dense (Bergamo, 2023) | 2 MB | 12 ch DDR5-4800 | 1.16 | 1.14 | 15% | SPEC results |
| AMD Eng Sample 100-000000894 (Genoa ES) | Zen 4 (Genoa, 2022) | 4 MB | 12 ch DDR5-4800 | 1.15 | 1.20 | 35% | estimate |
| AMD Threadripper PRO 3975WX | Zen 2 Threadripper (2020) | 4 MB | 8 ch DDR4-3200 (workstation socket) | 1.12 | 1.00 | 25% | estimate |
| AMD Eng Sample (other) | unknown (engineering sample) | ? | ? | 1.10 | 1.10 | 40% | estimate |
| AMD Eng Sample 100-000000997 | unknown (engineering sample) | ? | ? | 1.10 | 1.10 | 40% | estimate |
| AMD EPYC 7763 | Zen 3 (Milan, 2021) | 4 MB | 8 ch DDR4-3200 | 1.07 | 0.95 | 15% | SPEC results |
| AMD EPYC 7B13 (Google) | Zen 3 (Milan, 2021) | 4 MB | 8 ch DDR4-3200 | 1.07 | 0.95 | 20% | SPEC results of EPYC 7763 |
| AMD EPYC 7J13 (Oracle) | Zen 3 (Milan, 2021) | 4 MB | 8 ch DDR4-3200 | 1.07 | 0.95 | 20% | SPEC results of EPYC 7763 |
| AMD EPYC 7V13 (Azure) | Zen 3 (Milan, 2021) | 4 MB | 8 ch DDR4-3200 | 1.07 | 0.95 | 20% | SPEC results of EPYC 7763 |
| Intel Xeon Platinum 8558 | Intel Emerald Rapids (Xeon Scalable 5th gen, 2023) | 5.42 MB | 8 ch DDR5-5600 | 1.04 | 1.34 | 15% | SPEC results |
| Intel Xeon Platinum 8559C | Intel Emerald Rapids (Xeon Scalable 5th gen, 2023) | ? | 8 ch DDR5-5600 | 1.04 | 1.34 | 20% | SPEC results of Xeon Plat 8558 |
| Intel Xeon Platinum 8563C | Intel Emerald Rapids (Xeon Scalable 5th gen, 2023) | ? | 8 ch DDR5-5600 | 1.04 | 1.34 | 20% | SPEC results of Xeon Plat 8558 |
| AMD EPYC 7542 | Zen 2 (Rome, 2019) | 4 MB | 8 ch DDR4-3200 | 1.02 | 1.01 | 15% | SPEC results |
| Intel Xeon Platinum 8592+ | Intel Emerald Rapids (Xeon Scalable 5th gen, 2023) | 5 MB | 8 ch DDR5-5600 | 1.02 | 1.25 | 15% | SPEC results |
| AMD EPYC 7V73X (Azure Milan-X) | Zen 3 + 3D V-Cache (Milan-X, 2022) | 12 MB | 8 ch DDR4-3200 | 1.01 | 0.96 | 20% | SPEC results of EPYC 7773X |
| AMD Threadripper PRO 3995WX | Zen 2 Threadripper (2020) | 4 MB | 8 ch DDR4-3200 (workstation socket) | 1.01 | 0.85 | 20% | SPEC results |
| AMD EPYC 7502 | Zen 2 (Rome, 2019) | 4 MB | 8 ch DDR4-3200 | 1.00 | 1.00 | 15% | SPEC results |
| Intel Xeon Gold 5418Y | Intel Sapphire Rapids (Xeon Scalable 4th gen, 2023) | ? | 8 ch DDR5-4800 | 1.00 | 1.22 | 20% | SPEC results of Xeon Plat 8480+ |
| Intel Xeon Gold 6448Y | Intel Sapphire Rapids (Xeon Scalable 4th gen, 2023) | ? | 8 ch DDR5-4800 | 1.00 | 1.22 | 20% | SPEC results of Xeon Plat 8480+ |
| Intel Xeon Platinum 8469C | Intel Sapphire Rapids (Xeon Scalable 4th gen, 2023) | ? | 8 ch DDR5-4800 | 1.00 | 1.22 | 20% | SPEC results of Xeon Plat 8480+ |
| AMD EPYC 7532 | Zen 2 (Rome, 2019) | 8 MB | 8 ch DDR4-3200 | 0.98 | 1.04 | 15% | SPEC results |
| AMD EPYC 7663 | Zen 3 (Milan, 2021) | 4.57 MB | 8 ch DDR4-3200 | 0.97 | 0.96 | 15% | SPEC results |
| AMD EPYC 7452 | Zen 2 (Rome, 2019) | 4 MB | 8 ch DDR4-3200 | 0.96 | 0.98 | 15% | SPEC results |
| AMD EPYC 7713 | Zen 3 (Milan, 2021) | 4 MB | 8 ch DDR4-3200 | 0.93 | 0.89 | 15% | SPEC results |
| AMD EPYC 7K62 (Tencent) | Zen 2 (Rome, 2019) | 4 MB | 8 ch DDR4-3200 | 0.93 | 0.89 | 25% | SPEC results of EPYC 7642 |
| AMD EPYC 7R12 (AWS) | Zen 2 (Rome, 2019) | 4 MB | 8 ch DDR4-3200 | 0.93 | 0.89 | 25% | SPEC results of EPYC 7642 |
| Intel Xeon Gold 6438N | Intel Sapphire Rapids (Xeon Scalable 4th gen, 2023) | 1.88 MB | 8 ch DDR5-4800 | 0.93 | 1.28 | 15% | SPEC results |
| Intel Xeon Gold 6430 | Intel Sapphire Rapids (Xeon Scalable 4th gen, 2023) | 1.88 MB | 8 ch DDR5-4800 | 0.90 | 1.26 | 15% | SPEC results |
| AMD EPYC 7742 | Zen 2 (Rome, 2019) | 4 MB | 8 ch DDR4-3200 | 0.88 | 0.79 | 15% | SPEC results |
| AMD EPYC 7B12 (Google) | Zen 2 (Rome, 2019) | 4 MB | 8 ch DDR4-3200 | 0.88 | 0.79 | 20% | SPEC results of EPYC 7742 |
| Intel Xeon Gold 6154 | Intel Skylake-SP (Xeon Scalable 1st gen, 2017) | 1.38 MB | 6 ch DDR4-2666 | 0.86 | 0.91 | 15% | SPEC results |
| Intel Xeon Platinum 8272CL (Azure) | Intel Cascade Lake (Xeon Scalable 2nd gen, 2019) | 1.38 MB | 6 ch DDR4-2933 | 0.85 | 0.93 | 20% | SPEC results of Xeon Plat 8280 |
| Intel Xeon Gold 6248 | Intel Cascade Lake (Xeon Scalable 2nd gen, 2019) | 1.38 MB | 6 ch DDR4-2933 | 0.82 | 0.95 | 15% | SPEC results |
| AMD EPYC 7702 | Zen 2 (Rome, 2019) | 4 MB | 8 ch DDR4-3200 | 0.80 | 0.73 | 15% | SPEC results |
| Intel Xeon Silver 4314 | Intel Ice Lake-SP (Xeon Scalable 3rd gen, 2021) | 1.5 MB | 8 ch DDR4-3200 | 0.79 | 1.02 | 25% | SPEC results of Xeon Gold 6338 |
| Intel Xeon Gold 6133 | Intel Skylake-SP (Xeon Scalable 1st gen, 2017) | 1.38 MB | 6 ch DDR4-2666 | 0.72 | 0.79 | 25% | SPEC results of Xeon Gold 6148 |
| Intel Xeon Gold 6138 | Intel Skylake-SP (Xeon Scalable 1st gen, 2017) | 1.38 MB | 6 ch DDR4-2666 | 0.72 | 0.79 | 20% | SPEC results of Xeon Gold 6148 |
| AMD EPYC 7551 | Zen 1 (Naples, 2017) | 2 MB | 8 ch DDR4-2666 | 0.68 | 0.69 | 15% | SPEC results |
| Intel Xeon E5 v4 (other) | Intel Broadwell (Xeon E5 v4, 2016) | 2.5 MB | 4 ch DDR4-2400 | 0.68 | 0.62 | 20% | estimate |
| Intel Xeon Platinum 8160 | Intel Skylake-SP (Xeon Scalable 1st gen, 2017) | 1.38 MB | 6 ch DDR4-2666 | 0.66 | 0.73 | 15% | SPEC results |
| Intel Xeon Phi 7250 | Intel Knights Landing many-core (2016) | ? | 6 ch DDR4 + 16 GB on-package MCDRAM | 0.15 | 0.12 | 50% | estimate |

## Table 2: the market at 13:02 UTC, 2026-10-14

One row per processor model, sorted by the cheapest work. "Rentable" offers pass the launcher's rules; "all" includes offers refused for network price, sharing, reliability or verification. "$ per 1k work" and "Work per hour" are for the cheapest offer of that model (rentable if any; otherwise a refused offer, one of our own rentals today, or the 12:43 look at the market, as marked). A model's price varies a lot between hosts: compare the last column.

| CPU model | Offers now: rentable / all | Threads per offer | $/h now | $ per 1k thread-hours | $ per 1k work | Work per hour | GB RAM per thread | 12:43 UTC: $ per 1k thread-hours |
|---|---|---|---|---|---|---|---|---|
| AMD EPYC 9684X | 0 / 0 | 96/192 | 0.53-0.69 (12:43 only) | 2.9 | 2.3 | 236 | 0.5 | 2.89-7.43 |
| AMD EPYC 9965 | 1 / 1 | 768 | 2.40 | 3.3 | 2.6 | 914 | 0.7 | 3.26 |
| AMD EPYC 7502 | 0 / 0 | 64 | 0.34 (our rental) | 5.5 | 5.5 | 61 | 4.1 | - |
| Intel Xeon E5 v4 (other) | 0 / 6 | 80 | 0.31 (our rental) | 4.0 | 5.9 | 52 | 1.6 | 57.76 |
| AMD EPYC 7B13 (Google) | 1 / 3 | 128 | 1.00 | 8.2 | 7.6 | 131 | 4.1 | 28.26 |
| AMD Eng Sample 100-000000894 (Genoa ES) | 0 / 0 | 384 | 1.87 (our rental) | 9.7 | 8.5 | 221 | 1.3 | 9.87 |
| AMD EPYC 9V74 (Azure) | 1 / 4 | 320 | 4.01 | 13.1 | 9.7 | 415 | 1.6 | 62.52 |
| AMD EPYC 7J13 (Oracle) | 0 / 0 | 128 | 1.28 (our rental) | 10.4 | 9.7 | 131 | 2.1 | - |
| Intel Xeon Platinum 8272CL (Azure) | 0 / 0 | 104 | 0.87 (12:43 only) | 8.7 | 10.3 | 85 | 14.9 | 8.72 |
| AMD Threadripper PRO 9995WX | 0 / 1 | 96 | 1.41 (refused offer) | 15.3 | 10.5 | 134 | 2.7 | - |
| AMD EPYC 9734 | 0 / 0 | 448 | 6.43 (our rental) | 15.0 | 12.7 | 507 | 3.5 | - |
| AMD Threadripper PRO 7995WX | 0 / 1 | 96 | 1.67 (refused offer) | 18.2 | 14.0 | 120 | 1.4 | 6.19-18.17 |
| AMD EPYC 7702 | 0 / 3 | 64 | 0.74 (refused offer) | 12.1 | 15.1 | 49 | 2.0 | - |
| AMD EPYC 7542 | 1 / 2 | 128 | 1.92 | 15.7 | 15.4 | 125 | 4.1 | - |
| AMD EPYC 7663 | 0 / 3 | 112 | 1.80 (refused offer) | 16.7 | 17.2 | 104 | 2.3 | 16.72 |
| AMD EPYC 9554 | 2 / 5 | 256 | 6.94 | 28.2 | 19.2 | 361 | 2.0 | 28.23-130.30 |
| AMD EPYC 9754 | 1 / 8 | 512 | 11.21 | 22.8 | 19.7 | 570 | 2.0 | 6.60-69.47 |
| AMD EPYC 7742 | 0 / 1 | 64 | 1.09 (refused offer) | 17.8 | 20.2 | 54 | 2.0 | 21.85-54.31 |
| AMD EPYC 7V13 (Azure) | 1 / 3 | 256 | 5.45 | 22.2 | 20.7 | 263 | 2.0 | 22.20 |
| AMD Threadripper 7970X | 1 / 1 | 64 | 2.46 | 40.0 | 22.2 | 111 | 2.0 | 40.02 |
| Intel Xeon Phi 7250 | 0 / 1 | 272 | 0.87 (refused offer) | 3.3 | 22.3 | 39 | 1.0 | - |
| AMD EPYC 7V73X (Azure Milan-X) | 0 / 0 | 64 | 1.48 (12:43 only) | 24.0 | 23.8 | 62 | 4.1 | 24.02 |
| AMD Eng Sample 100-000001535 (Turin ES?) | 0 / 1 | 512 | 14.41 (refused offer) | 29.3 | 24.4 | 590 | 2.3 | 30.62 |
| AMD Threadripper PRO 7975WX | 1 / 1 | 64 | 2.71 | 44.1 | 24.5 | 111 | 3.1 | 44.08 |
| AMD Threadripper 9970X | 0 / 0 | 64 | 2.94 (12:43 only) | 47.8 | 24.5 | 120 | 4.1 | 47.83 |
| AMD Eng Sample 100-000000997 | 0 / 1 | 96 | 2.54 (refused offer) | 27.5 | 25.0 | 101 | 2.0 | 27.55 |
| Intel Xeon Platinum 8469C | 0 / 1 | 96 | 2.46 (refused offer) | 26.7 | 26.7 | 92 | 5.5 | 28.25 |
| AMD EPYC 9654 | 1 / 6 | 192 | 7.01 | 38.0 | 29.7 | 236 | 1.4 | 13.62-38.04 |
| AMD EPYC 9655 | 0 / 3 | 96/109/219 | 3.88-16.33 (refused offer) | 77.5 | 30.9 | 125 | 4.1 | 42.07-77.53 |
| AMD EPYC 7K62 (Tencent) | 0 / 5 | 96 | 2.68 (refused offer) | 29.1 | 31.3 | 86 | 2.0 | 19.37-23.21 |
| AMD Threadripper PRO 5975WX | 0 / 1 | 64 | 2.74 (12:43 only) | 44.6 | 31.8 | 86 | 2.0 | 44.57 |
| Intel Xeon Gold 6248 | 1 / 3 | 80 | 2.01 | 26.2 | 31.9 | 63 | 9.8 | 26.18-27.85 |
| AMD EPYC 9555 | 1 / 5 | 256 | 13.34 | 54.3 | 32.9 | 406 | 6.1 | 51.67-54.31 |
| Intel Xeon Platinum 8558 | 0 / 2 | 64/96 | 2.31-3.20 (refused offer) | 36.2 | 33.4 | 96 | 8.2 | 34.76 |
| AMD EPYC 7513 | 0 / 1 | 64 | 2.41 (refused offer) | 39.2 | 33.5 | 72 | 4.1 | 39.29 |
| AMD EPYC 9J14 (Oracle) | 0 / 1 | 384 | 16.01 (refused offer) | 43.4 | 33.9 | 472 | 2.7 | 43.43-44.73 |
| AMD Threadripper PRO 9975WX | 1 / 1 | 64 | 4.14 | 67.5 | 34.6 | 120 | 3.1 | - |
| Intel Xeon Gold 6154 | 1 / 1 | 72 | 2.23 | 32.2 | 37.5 | 59 | 10.9 | 32.23-78.23 |
| AMD Threadripper PRO 3975WX | 1 / 2 | 64 | 2.74 | 44.6 | 39.8 | 69 | 2.0 | 44.57 |
| AMD EPYC 7B12 (Google) | 1 / 2 | 256 | 9.35 | 38.1 | 43.2 | 216 | 4.1 | 11.84-38.06 |
| Intel Xeon Gold 6133 | 1 / 3 | 80 | 2.41 | 31.3 | 43.5 | 55 | 3.3 | 31.32-54.36 |
| AMD Threadripper PRO 3995WX | 0 / 1 | 64 | 2.74 (refused offer) | 44.6 | 44.1 | 62 | 2.0 | 44.57 |
| AMD EPYC 7452 | 2 / 3 | 64/128 | 2.66-5.35 | 43.4 | 45.1 | 59 | 4.1 | 9.13-43.55 |
| Intel Xeon Platinum 8592+ | 0 / 5 | 85/170/208 | 4.34-59.53 (refused offer) | 55.0 | 51.9 | 167 | 4.1 | 51.30-298.11 |
| AMD EPYC 7532 | 1 / 4 | 128 | 7.27 | 59.1 | 60.3 | 120 | 8.2 | 41.12 |
| Intel Xeon Platinum 8563C | 0 / 1 | 104 | 6.42 (refused offer) | 64.3 | 61.9 | 104 | 5.0 | 64.32 |
| AMD EPYC 9124 | 1 / 1 | 64 | 6.09 | 99.0 | 67.8 | 90 | 8.2 | 99.05 |
| AMD EPYC 7713 | 1 / 2 | 64 | 4.01 | 65.2 | 70.1 | 57 | 8.2 | 20.25-65.20 |
| AMD EPYC 9135 | 1 / 1 | 64 | 8.27 | 134.7 | 73.6 | 112 | 12.3 | - |
| AMD EPYC 9755 | 0 / 0 | 64 | 6.51-8.02 (12:43 only) | 118.2 | 78.5 | 83 | 3.8 | 106.00-130.48 |
| AMD EPYC 9B14 (cloud) | 1 / 2 | 192 | 21.34 | 115.8 | 90.4 | 236 | 4.1 | 10.16-115.77 |
| Intel Xeon 6767P | 0 / 1 | 128 | 18.67 (refused offer) | 152.0 | 110.1 | 170 | 8.2 | 151.95 |
| AMD EPYC 7543 | 0 / 2 | 64 | 11.95 (refused offer) | 194.5 | 154.4 | 77 | 8.2 | 18.49-194.53 |
| AMD EPYC 9575F | 0 / 0 | 128 | 45.34 (12:43 only) | 369.0 | 226.4 | 200 | 12.3 | 368.98 |
| Intel Xeon 6776P | 0 / 1 | 128 | 48.76 (refused offer) | 396.8 | 287.5 | 170 | 12.3 | 396.77 |
| Intel Xeon Platinum 8559C | 0 / 0 | 96 | 33.53-37.53 (12:43 only) | 385.5 | 349.8 | 96 | 10.8 | 363.78-407.18 |
| AMD EPYC 9454 | 0 / 1 | 192 | 106.67 (12:43 only) | 578.7 | 391.0 | 273 | 8.2 | 578.73 |

Notes: GB of RAM per thread is the machine's memory divided by its usable threads. The EPYC 9684X ($0.53/h for 192 threads) was the cheapest machine seen today, but it was refused for its network price ($0.05/GB) and its 94 GB of memory means several of its 12 memory channels are probably empty. The EPYC 7502, 7J13, 9734, the engineering sample and the Xeon E5-2673 v4 rows are our own rentals today (not in the market while we held them).

## Rankings

### (a) Cheapest per unit of work (trick 2–5 solves, equity sweeps, rollouts, tournaments)

| # | Machine | Threads | RAM | $/h | Speed per thread | Work per hour | $ per 1k work |
|---|---|---|---|---|---|---|---|
| 1 | AMD EPYC 9965 | 768 | 503 GB | 2.40 | 1.24 | 914 | **2.6** |
| 2 | AMD EPYC 7B13 (Google variant of the 7763) | 128 | 504 GB | 1.00 | 1.07 | 131 | 7.6 |
| 3 | AMD EPYC 9V74 (Azure variant, 80 cores) | 320 | 503 GB | 4.01 | 1.35 | 415 | 9.7 |
| 4 | AMD EPYC 7542 | 128 | 504 GB | 1.92 | 1.02 | 125 | 15.4 |
| 5 | AMD EPYC 9554 | 256 | 504 GB | 6.94 | 1.47 | 361 | 19.2 |

Just outside the launcher's rules today: EPYC 9684X ($2.3 per 1k work, network price), EPYC 9V74 halves (shared machine, $9.7), EPYC 9654 half machine (shared, $10.6), EPYC 9754 half machine (shared, $10.9).

### (b) Fastest within +20% of the cheapest cost per work (the owner's rule)

Today the EPYC 9965 is both the cheapest per unit of work and the fastest machine on offer (914 units per hour, against 570 for the next fastest), so the rule picks it, and nothing else comes within 20% of its price per work. Applying the rule again each time the chosen offer is taken gives this order:

| # | Machine | Work per hour | $ per 1k work | Offers within 20% of the cheapest at that step |
|---|---|---|---|---|
| 1 | AMD EPYC 9965, 768 threads, $2.40/h | 914 | 2.6 | 1 |
| 2 | AMD EPYC 7B13, 128 threads, $1.00/h | 131 | 7.6 | 1 |
| 3 | AMD EPYC 9V74, 320 threads, $4.01/h | 415 | 9.7 | 1 |
| 4 | AMD EPYC 7542, 128 threads, $1.92/h | 125 | 15.4 | 1 |
| 5 | AMD EPYC 9754, 512 threads, $11.21/h | 570 | 19.7 | 5 (the two 9554s, the 9754, the 7V13 and the Threadripper 7970X) |

Several cheap machines can also run side by side: the launcher rents one machine per run, and runs can go in parallel. So a small cheap machine is not a handicap for jobs made of many independent deals.

### (c) Best for exact trick-1 solves (huge transposition tables)

Machines with at least 480 GB of memory and at least 1 GB per thread, ranked by TT speed. "Deals at once" = how many 64 GB tables fit in 90% of the memory; fewer threads per deal waste less (see m2-trick1-1 above).

| # | Machine | Threads | RAM | $/h | TT speed | $ per 1k work | Deals at once | Threads per deal |
|---|---|---|---|---|---|---|---|---|
| 1 | AMD EPYC 7B13 | 128 | 504 GB | 1.00 | 0.95 | **8.6** | 7 | 18 |
| 2 | AMD EPYC 9654 (the box we used for trick 2, if it comes back) | 384 | 503 GB | 4.27 | 1.33 | 8.7 | 7 | 53 |
| 3 | AMD EPYC 9V74 | 320 | 503 GB | 4.01 | 1.40 | 9.3 | 7 | 44 |
| 4 | AMD EPYC 9734 (today's trick-1 machine) | 448 | 1,511 GB | 6.43 | 1.20 | 12.5 | **21** | 20 |
| 5 | AMD EPYC 7542 | 128 | 504 GB | 1.92 | 1.01 | 15.5 | 7 | 18 |

Next: EPYC 9554 ($18.2), EPYC 9754 ($20.0, 14 deals at once). By the owner's rule the 9654 box would be picked if offered (most work per hour within 20%), otherwise the 9V74. **For trick-1 the deciding numbers are probably the memory and the threads per deal, not the processor**: a 1.5 TB machine can keep three times as many deals running as a 512 GB one, each on fewer threads. How much that is worth is the first thing to measure.

## Caveats

- **Dense chips (EPYC 9734, 9754, 9965, 9845; "c" cores).** They put twice as many cores in the same space, with half the L3 cache per core (2 MB instead of 4 MB) and lower clock speeds. SPEC says their threads are still as fast as the 7502's or faster (1.13–1.24), and they are cheap per thread. But with 768 threads sharing 24 memory channels [the parallel links between the processor and its memory sticks], the 9965 is the weakest chip per thread for memory-heavy work (TT speed 1.09). Fine for tricks 2–5; measure before trusting it for trick 1.
- **Engineering samples** ("AMD Eng Sample: 100-…"). Pre-release chips: their identity is only a guess (the 894-04 shows Zen 4 instructions and 2 × 96 cores, so it is probably a 9654-class Genoa), their clocks may be lower than the final product and they may have bugs fixed later. Treat with ±35–40%, and note the quota and memory trouble on host 137718.
- **Cloud variants** (7B12, 7B13, 7V13, 7J13, 7K62, 9J14, 9B14, 9V74, 8272CL, 8469C, 8559C…) were made for large cloud companies (Google, Microsoft Azure, Oracle, Tencent and others). AMD and Intel do not publish their specifications; they are mapped to the nearest retail chip with the same core count.
- **Memory sticks.** A machine with less memory than "sockets × channels × 16 GB" almost certainly has empty memory channels, which slows memory-heavy work. The tool flags these offers ("RAM below full population").
- **CPU quota.** Assume 96% of the advertised threads; one host gave 50%. Our runs record the real quota (`hardware.json`), so the tool's `machines` list can be extended.
- **Shared machines** (offers for part of a machine, `gpu_frac < 1`) are refused by the launcher. They are often half the price per thread of whole machines, but other renters' jobs share the memory and caches, so measured speed would be unreliable.
- **Network price.** The launcher refuses offers above $0.02/GB. Our runs download and upload a few GB at most, so even $0.05/GB would add only about $0.10–0.25 per run; today that rule excluded the cheapest machine seen (EPYC 9684X). Worth revisiting.
- **The market moves fast.** Between 12:43 and 13:02 UTC most offer numbers changed and several of the cheapest machines disappeared. Re-run the tool right before renting.

## What a benchmark should settle (in order of importance)

1. **Trick-1 deals at once vs threads per deal.** On one large-memory machine (the 9734 type, 1.5 TB), solve the same 16 trick-1 worlds as 1 × 448 threads, 4 × 112, 16 × 28 and 32 × 14 (with table caps of 86, 45 and 20 GB). This gives worlds per hour for each layout, which tells us how much memory per thread is worth, and whether trick-1 should go to big-memory machines or to many cheap ones. Cost: about $5–10.
2. **Our own speed factors.** A fixed 15-minute benchmark (the same trick-2 and trick-3 worlds, one per thread on every thread, plus 8 fixed trick-1 worlds) on one machine of each family seen today: Zen 2 (7502/7542), Zen 3 (7B13/7V13), Zen 4 (9654/9554), dense Zen 4c (9734/9754), dense Zen 5c (9965), and one Intel (8272CL or Gold 6248). It replaces the SPEC estimates with measured ones (`vast_market.py record`). Cost: about $3–5.
3. **Same decisions on the 9734 and the 9754**, to separate machine speed from decision difficulty in today's trick-1 numbers.
4. **Dense chips with big tables**: does the 9965's small cache per core and crowded memory hurt trick-2 tables (0.1–0.5 GB per thread, 768 threads) or trick-1 tables more than SPEC says?
5. **Host-to-host spread**: the same model on two different hosts (memory sticks, power settings, quota).
6. **SMT benefit for our program**: if the second thread of a core adds much less than 15–35% for our solver, machines whose cores are faster but fewer gain ground.

## How to refresh

```
# live market, ranked (read-only: it never rents anything)
python3 python/tools/vast_market.py
python3 python/tools/vast_market.py --include-other     # also show offers the launcher refuses today
python3 python/tools/vast_market.py --markdown          # tables ready to paste into this document

# save a dated snapshot, and re-rank it later without the network
python3 python/tools/vast_market.py --save docs/vast/market-$(date -u +%F).json
python3 python/tools/vast_market.py --from docs/vast/market-2026-10-14.json

# after a benchmark: store the measured speed of one processor model (the measured value then replaces the estimate)
python3 python/tools/vast_market.py record --cpu "AMD EPYC 9965 192-Core Processor" \
    --compute 1.31 --tt 0.97 --run bench-9965-1 --note "fixed trick-2/3 set, 737 threads"
```

- The tool needs only Python 3 (no extra packages). It asks vast.ai for offers the way the launcher does; a `VAST_API_KEY` is used if set, but searching does not need one. Tests: `python3 -m unittest discover -s python/tools -p "test_*.py"`.
- **How a benchmark turns into a speed factor:** run the same fixed work on the new machine and on a machine whose factor is already measured (or the 7502). Speed factor = (worlds per hour per usable thread on the new machine) ÷ (the same on the reference) × (the reference's factor). Use the compute benchmark for `--compute` and the trick-1 benchmark for `--tt`.
- **Unknown processors** get a deliberately low default speed (0.6) so they never top a ranking by accident; the tool lists them at the end of its output. Add them to `docs/vast/speed-factors.json` (copy an entry of the same design; give a source).
- **Changing the rules:** `--premium 0.2` is the owner's 20% rule; `--t1-min-ram-gb 480` and `--t1-min-gb-per-thread 1.0` set the trick-1 memory floor; `--t1-table-gb 64` sets the memory reserved per trick-1 deal.

## Files

- `docs/vast/market-2026-10-14.json`: the snapshot (127 offers at 13:02 UTC under five searches, plus the 138 offers of the 12:43 first look), trimmed to the fields that matter.
- `docs/vast/speed-factors.json`: speed per thread for 180 processor models, with sources, uncertainty, our measured values once we have them (`measured`), the host quirks (`machines`) and our runs so far (`observations`).
- `python/tools/vast_market.py` and its tests `python/tools/test_vast_market.py`.

## Sources

- SPEC CPU2017 integer rate results, all published systems: https://www.spec.org/cpu2017/results/rint2017.html. Per-model result pages used for the main chips (one copy per thread, all threads busy):
  EPYC 7502 https://www.spec.org/cpu2017/results/res2019q4/cpu2017-20191111-19709.html ·
  EPYC 7542 https://www.spec.org/cpu2017/results/res2019q3/cpu2017-20190831-17279.html ·
  EPYC 7742 https://www.spec.org/cpu2017/results/res2019q4/cpu2017-20191001-18923.html ·
  EPYC 7763 https://www.spec.org/cpu2017/results/res2021q1/cpu2017-20210301-25119.html ·
  EPYC 9554 https://www.spec.org/cpu2017/results/res2024q1/cpu2017-20240212-41484.html ·
  EPYC 9654 https://www.spec.org/cpu2017/results/res2024q2/cpu2017-20240326-42707.html ·
  EPYC 9684X https://www.spec.org/cpu2017/results/res2024q3/cpu2017-20240730-44359.html ·
  EPYC 9734 https://www.spec.org/cpu2017/results/res2024q3/cpu2017-20240813-44504.html ·
  EPYC 9754 https://www.spec.org/cpu2017/results/res2024q1/cpu2017-20240130-40906.html ·
  EPYC 9555 https://www.spec.org/cpu2017/results/res2024q4/cpu2017-20241104-45236.html ·
  EPYC 9755 https://www.spec.org/cpu2017/results/res2024q4/cpu2017-20240920-44768.html ·
  EPYC 9965 https://www.spec.org/cpu2017/results/res2024q4/cpu2017-20240923-44837.html ·
  Xeon Gold 6248 https://www.spec.org/cpu2017/results/res2019q2/cpu2017-20190415-11907.html ·
  Xeon Platinum 8280 https://www.spec.org/cpu2017/results/res2019q2/cpu2017-20190319-11465.html ·
  Xeon Platinum 8480+ https://www.spec.org/cpu2017/results/res2023q3/cpu2017-20230630-37555.html ·
  Xeon Platinum 8592+ https://www.spec.org/cpu2017/results/res2023q4/cpu2017-20231128-40081.html ·
  Xeon 6980P https://www.spec.org/cpu2017/results/res2025q1/cpu2017-20241230-45863.html.
  These pages also give each chip's clock speeds, cache layout (for example "16 MB shared / 8 cores" on the 9754) and memory type. Every other model's pages are listed in `speed-factors.json`.
- Dense Zen 4c (Bergamo) design and memory delays: Chips and Cheese, https://chipsandcheese.com/p/testing-amds-bergamo-zen-4c-spam ; ServeTheHome, https://www.servethehome.com/amd-epyc-bergamo-epyc-9754-cloud-native-sp5/3/ ; Phoronix, https://www.phoronix.com/review/amd-epyc-9754-bergamo
- Zen 5 / Zen 5c (Turin) design, 16 dense cores sharing 32 MB of L3: AMD EPYC 9005 architecture overview, https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/user-guides/58462_amd-epyc-9005-tg-architecture-overview.pdf
- SMT benefit (+13% on average over Phoronix's test suite, EPYC 9575F): https://www.phoronix.com/review/amd-epyc-zen5-smt/6
- Threadripper: Phoronix 7995WX review, https://www.phoronix.com/review/amd-linux-threadripper-pro-7995wx ; TechPowerUp, Threadripper 7000 line-up (4 vs 8 memory channels), https://www.techpowerup.com/314900/amd-ryzen-threadripper-7000-series-lineup-revealed ; 9970X specifications, https://www.scan.co.uk/products/amd-ryzen-threadripper-9970x-tr5-zen-5-32-core-64-thread-40ghz-54ghz-128mb-cache-pcie-50-350w-cpu
- Cloud and new parts with no maker's page (reseller listings, so weaker evidence): EPYC 9V74 (80 cores, 2.6/3.7 GHz, 320 MB), https://serverflow.ru/catalog/komplektuyushchie/protsessory/amd-epyc-9v74/ ; EPYC 9J14 ("9654 OEM"), https://www.ebay.de/itm/387364030673 ; Xeon 6767P (64 cores, 2.4 GHz, 3.6 GHz all-core, 336 MB), https://www.neweggbusiness.com/intel-xeon-6767p-fclga-4710/p/9B-19-118-604
- vast.ai offer fields and search format: the `vastai` command-line tool 1.8.3 (`search offers`), as used by our launcher.
- Our runs: `docs/M2-SPIKE.md`, `docs/m2-epyc/`, run reports m3-agree-t2d, -t3b, -t4, -t4c, -t1-a3 (hardware.json, report.json, times per world), and today's trick-1 runs as reported by the session.
