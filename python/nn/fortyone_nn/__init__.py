"""41Bots oracle value networks (docs/M3-HYBRID-SPEC.md section 6).

- ``formats``: label shards (.npy records written by Rust) and safetensors files.
- ``intnet``: the integer forward pass and softmax, bit for bit the Rust ``crates/nn`` one.
- ``quantize``: float network -> the Rust integer scheme (format ``41bots-int-mlp-v1``).
- ``train``: the PyTorch trainer (plain MLP, soft cross-entropy, early stopping; CPU or GPU).
- ``device``: the GPU check and the device facts recorded with a training run.
- ``audit``: the auditor (docs/m3-hybrid/audit-loop.md B1): the network's weak groups of positions,
  found on half of a fresh shard and confirmed on the other half, as predicates Rust evaluates.

The encoder lives only in Rust; Python reads encoded features from the shards.
"""
