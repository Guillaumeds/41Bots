"""41Bots oracle value networks (docs/M3-HYBRID-SPEC.md section 6) and the student network
(docs/STUDENT.md).

- ``formats``: label shards (.npy records written by Rust) and safetensors files.
- ``intnet``: the integer forward pass and softmax, bit for bit the Rust ``crates/nn`` one.
- ``quantize``: float network -> the Rust integer scheme (format ``41bots-int-mlp-v1``).
- ``train``: the PyTorch trainer (plain MLP, soft cross-entropy, early stopping; CPU or GPU).
- ``device``: the GPU check and the device facts recorded with a training run.
- ``audit``: the auditor (docs/m3-hybrid/audit-loop.md B1): the network's weak groups of positions,
  found on half of a fresh shard and confirmed on the other half, as predicates Rust evaluates.
- ``studentint``: the student's feature shards (written by ``student encode``), the decision loss
  against the teacher's means, and its integer forward pass, bit for bit ``crates/student``'s
  (numpy only).
- ``student``: the student's PyTorch trainer (one scorer per candidate, soft four-way targets,
  validation by position, early stopping, warm start with shrink-and-perturb, the keep rule).

The encoders live only in Rust; Python reads encoded features from the shards.
"""
