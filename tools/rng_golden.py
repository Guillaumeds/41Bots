#!/usr/bin/env python3
"""Independent, stdlib-only generator of golden vectors for the 41 engine RNG.

The engine (crates/engine/src/rng.rs) must reproduce these exactly on every platform.

Algorithms (all 64-bit, wrapping arithmetic):
  SplitMix64 (seed expansion):
    state += 0x9E3779B97F4A7C15
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)
  xoshiro256** (generation), seeded with four consecutive SplitMix64 outputs:
    result = rotl(s1 * 5, 7) * 9
    t = s1 << 17
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = rotl(s3, 45)
  bounded(n) in [0, n), unbiased (Lemire's nearly-divisionless method):
    threshold = (2^64 - n) % n
    loop: x = next(); m = x * n; low = m mod 2^64; if low >= threshold: return m >> 64
  shuffle (Fisher-Yates, from the end): for i = len-1 down to 1: j = bounded(i + 1); swap(i, j)

Usage: python3 tools/rng_golden.py > crates/engine/tests/data/rng_golden.txt
"""
MASK = (1 << 64) - 1


def splitmix64_next(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return state, (z ^ (z >> 31)) & MASK


def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & MASK


class Xoshiro256StarStar:
    def __init__(self, seed):
        st = seed & MASK
        s = []
        for _ in range(4):
            st, v = splitmix64_next(st)
            s.append(v)
        self.s = s

    def next_u64(self):
        s = self.s
        result = (rotl((s[1] * 5) & MASK, 7) * 9) & MASK
        t = (s[1] << 17) & MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        return result

    def bounded(self, n):
        assert n > 0
        threshold = ((1 << 64) - n) % n
        while True:
            x = self.next_u64()
            m = x * n
            low = m & MASK
            if low >= threshold:
                return m >> 64

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            j = self.bounded(i + 1)
            items[i], items[j] = items[j], items[i]


def main():
    seeds = [0, 1, 42, 0xDEADBEEF, 0x9E3779B97F4A7C15, (1 << 64) - 1]
    for seed in seeds:
        print(f"seed {seed:016x}")
        r = Xoshiro256StarStar(seed)
        print("next " + " ".join(f"{r.next_u64():016x}" for _ in range(8)))
        print("bounded6 " + " ".join(str(r.bounded(6)) for _ in range(16)))
        print("bounded53 " + " ".join(str(r.bounded(53)) for _ in range(16)))
        print("bounded8 " + " ".join(str(r.bounded(8)) for _ in range(16)))
        big = (1 << 63) + 12345
        print("boundedbig " + " ".join(str(r.bounded(big)) for _ in range(4)))
        deck = list(range(53))
        r.shuffle(deck)
        print("shuffle " + " ".join(str(c) for c in deck))
        deck2 = list(range(53))
        r.shuffle(deck2)
        print("shuffle " + " ".join(str(c) for c in deck2))


if __name__ == "__main__":
    main()
