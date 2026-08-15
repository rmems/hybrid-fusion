# ReverseHybridPath Orchestrator (#23) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ReverseHybridPath<R: ExpertRouter>` — the reverse-path host that runs `SpikeActivity → project_spike_activity → ExpertRouter::route → HybridOutput` with MoE fields populated — closing epic #20 item #23 without checkpoints or SAAQ.

**Architecture:** Dual-host design (Approach B from the design spec). Keep ANN→SNN `HybridNetwork<T, S>` unchanged. New generic host `ReverseHybridPath<R>` owns only projection mode + router + step counter (corinth-canal `Model::forward_activity` projector+router half). No Sentry capture on reverse v1. File-free tests via in-test mock routers; optional `SyntheticExpertRouter` under `backends`.

**Tech Stack:** Rust 2024 edition crate `hybrid-fusion` 0.3.0; existing traits (`ExpertRouter`, `SpikeActivity`), `project_spike_activity`, pure `routing` math, optional `backends` feature. No new Cargo dependencies.

**Spec:** [docs/superpowers/specs/2026-08-14-reverse-path-orchestrator-design.md](../specs/2026-08-14-reverse-path-orchestrator-design.md)

**Issue:** [rmems/hybrid-fusion#23](https://github.com/rmems/hybrid-fusion/issues/23)

---

## File structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `src/reverse.rs` | **Create** | `ReverseHybridPath`, unit tests, private `last_fired` helper |
| `src/lib.rs` | **Modify** | `pub mod reverse`; re-export `ReverseHybridPath` |
| `src/hybrid.rs` | **Modify** | Doc comment only: point MoE fields at `ReverseHybridPath` (not "later") |
| `tests/reverse_path.rs` | **Create** | Integration: mock router always; Synthetic under `#[cfg(feature = "backends")]` |
| `docs/implementing-backends.md` | **Modify** | §10: document orchestrator + plug-in steps for real MoE backends |
| `README.md` | **Modify** | Scope + public surface row for `ReverseHybridPath` |
| `CHANGELOG.md` | **Modify** | Unreleased bullet for #23 |
| `AGENTS.md` | **Modify** | One line: reverse host vs ANN→SNN host |
| `docs/extraction-map.md` | **Modify** | Optional: mark #23 row done-ish only after merge — skip in this PR unless already partial |

Do **not** touch: `Cargo.toml` features, concrete backends, #24/#27, Sentry wiring for reverse.

---

### Task 1: Unit tests for `ReverseHybridPath` (TDD — fail first)

**Files:**
- Create: `src/reverse.rs` (tests first; stub type so the module compiles if needed — prefer failing compile on missing type)
- Modify: `src/lib.rs` (add `pub mod reverse` so unit tests compile under the crate)

- [ ] **Step 1: Create `src/reverse.rs` with only the unit-test module and a minimal scaffold that will not pass yet**

Write the full file below (complete type + unit tests). Prefer one commit with the full implementation once unit tests pass; pure red/green TDD is optional if the skeleton would not compile without methods.

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reverse-path host: SNN activity → embedding → MoE route.
//!
//! Extracted from corinth-canal `Model` (`projector` + `router` half of
//! `forward_activity`). Does not own neuron dynamics or checkpoint I/O.
//! Dual host alongside [`crate::HybridNetwork`] (ANN → SNN).

use crate::error::{HybridError, Result};
use crate::projector::project_spike_activity;
use crate::traits::{ExpertRouter, SpikeActivity};
use crate::types::{HybridOutput, ProjectionMode};

/// Reverse-path host: SNN activity → embedding → MoE route.
///
/// Extracted from corinth-canal `Model` (`projector` + `router` half of
/// `forward_activity`). Does not own neuron dynamics or checkpoint I/O.
pub struct ReverseHybridPath<R: ExpertRouter> {
    mode: ProjectionMode,
    n_neurons: usize,
    embed_dim: usize,
    router: R,
    global_step: u64,
}

impl<R: ExpertRouter> ReverseHybridPath<R> {
    /// Rejects `n_neurons == 0` or `embed_dim == 0`.
    pub fn new(
        mode: ProjectionMode,
        n_neurons: usize,
        embed_dim: usize,
        router: R,
    ) -> Result<Self> {
        if n_neurons == 0 {
            return Err(HybridError::InvalidConfig(
                "ReverseHybridPath: n_neurons must be > 0".into(),
            ));
        }
        if embed_dim == 0 {
            return Err(HybridError::InvalidConfig(
                "ReverseHybridPath: embed_dim must be > 0".into(),
            ));
        }
        Ok(Self {
            mode,
            n_neurons,
            embed_dim,
            router,
            global_step: 0,
        })
    }

    pub fn mode(&self) -> ProjectionMode {
        self.mode
    }

    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub fn router(&self) -> &R {
        &self.router
    }

    pub fn router_mut(&mut self) -> &mut R {
        &mut self.router
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn reset(&mut self) {
        self.global_step = 0;
    }

    /// Project activity, route through MoE, return `HybridOutput` with MoE fields set.
    ///
    /// Semantics match corinth-canal `Model::forward_activity` (projector + router half):
    /// 1. `global_step = saturating_add(1)`
    /// 2. `embedding = project_spike_activity(...)`
    /// 3. `route = router.route(&embedding)` — **no Sentry capture** on reverse v1
    /// 4. Build `HybridOutput` (empty `stimuli`; `fired_neurons` = last non-empty spike step)
    pub fn forward_activity(&mut self, activity: &SpikeActivity) -> Result<HybridOutput> {
        self.global_step = self.global_step.saturating_add(1);

        let embedding = project_spike_activity(
            self.mode,
            activity,
            self.n_neurons,
            self.embed_dim,
        )?;

        // v1: propagate router errors without Sentry (validation-heavy path;
        // avoid flooding on bad activity / empty embeddings).
        let route = self.router.route(&embedding)?;

        let fired_neurons = last_fired(&activity.spike_train);

        Ok(HybridOutput {
            embedding,
            stimuli: Vec::new(),
            fired_neurons,
            global_step: self.global_step,
            expert_weights: Some(route.expert_weights),
            selected_experts: Some(route.selected_experts),
            routing_entropy: route.routing_entropy,
        })
    }
}

/// Last non-empty step of the spike train; empty if none fired.
fn last_fired(spike_train: &[Vec<usize>]) -> Vec<usize> {
    spike_train
        .iter()
        .rev()
        .find(|step| !step.is_empty())
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ExpertRouteOutput;

    /// Deterministic mock router for unit tests (no `backends` feature).
    struct MockRouter {
        num_experts: usize,
        top_k: usize,
        calls: usize,
    }

    impl MockRouter {
        fn new(num_experts: usize, top_k: usize) -> Self {
            Self {
                num_experts,
                top_k: top_k.min(num_experts),
                calls: 0,
            }
        }
    }

    impl ExpertRouter for MockRouter {
        fn num_experts(&self) -> usize {
            self.num_experts
        }

        fn top_k(&self) -> usize {
            self.top_k
        }

        fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput> {
            if embedding.is_empty() {
                return Err(HybridError::InvalidConfig(
                    "MockRouter: empty embedding".into(),
                ));
            }
            self.calls += 1;
            let n = self.num_experts;
            let mut expert_weights = vec![0.0f32; n];
            let k = self.top_k;
            let w = 1.0 / k as f32;
            let mut selected = Vec::with_capacity(k);
            for i in 0..k {
                expert_weights[i] = w;
                selected.push(i);
            }
            Ok(ExpertRouteOutput {
                expert_weights,
                selected_experts: selected,
                routing_entropy: Some(0.5),
            })
        }
    }

    #[test]
    fn new_rejects_zero_n_neurons() {
        let r = MockRouter::new(4, 2);
        let err = ReverseHybridPath::new(ProjectionMode::RateSum, 0, 8, r).unwrap_err();
        match err {
            HybridError::InvalidConfig(msg) => assert!(msg.contains("n_neurons")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn new_rejects_zero_embed_dim() {
        let r = MockRouter::new(4, 2);
        let err = ReverseHybridPath::new(ProjectionMode::RateSum, 4, 0, r).unwrap_err();
        match err {
            HybridError::InvalidConfig(msg) => assert!(msg.contains("embed_dim")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn forward_activity_increments_global_step() {
        let r = MockRouter::new(4, 2);
        let mut path =
            ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, r).unwrap();
        assert_eq!(path.global_step(), 0);
        let act = SpikeActivity::from_fired(&[0, 2], 4).unwrap();
        let out = path.forward_activity(&act).unwrap();
        assert_eq!(out.global_step, 1);
        assert_eq!(path.global_step(), 1);
        let out2 = path.forward_activity(&act).unwrap();
        assert_eq!(out2.global_step, 2);
    }

    #[test]
    fn reset_clears_global_step() {
        let r = MockRouter::new(4, 2);
        let mut path =
            ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, r).unwrap();
        let act = SpikeActivity::from_fired(&[1], 4).unwrap();
        path.forward_activity(&act).unwrap();
        path.reset();
        assert_eq!(path.global_step(), 0);
    }

    #[test]
    fn forward_activity_sets_moe_fields_and_empty_stimuli() {
        let r = MockRouter::new(4, 2);
        let mut path =
            ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, r).unwrap();
        let act = SpikeActivity::from_fired(&[0, 3], 4).unwrap();
        let out = path.forward_activity(&act).unwrap();

        assert_eq!(out.embedding.len(), 8);
        assert!(out.stimuli.is_empty());
        assert_eq!(out.fired_neurons, vec![0, 3]);
        let weights = out.expert_weights.expect("MoE weights");
        assert_eq!(weights.len(), 4);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let selected = out.selected_experts.expect("selected");
        assert_eq!(selected.len(), 2);
        assert!(out.routing_entropy.is_some());
    }

    #[test]
    fn last_fired_prefers_last_non_empty_step() {
        let train = vec![vec![0], vec![], vec![1, 2]];
        assert_eq!(last_fired(&train), vec![1, 2]);
        let empty: Vec<Vec<usize>> = vec![vec![], vec![]];
        assert!(last_fired(&empty).is_empty());
    }

    #[test]
    fn accessors_match_construction() {
        let r = MockRouter::new(8, 3);
        let path =
            ReverseHybridPath::new(ProjectionMode::MembraneSnapshot, 16, 32, r).unwrap();
        assert_eq!(path.mode(), ProjectionMode::MembraneSnapshot);
        assert_eq!(path.n_neurons(), 16);
        assert_eq!(path.embed_dim(), 32);
        assert_eq!(path.router().num_experts(), 8);
        assert_eq!(path.router().top_k(), 3);
    }
}
```

- [ ] **Step 2: Wire the module in `src/lib.rs`**

Add after the existing `pub mod routing;` line (keep alphabetical-ish order: after `projector`, before `routing` is fine — use this block):

```rust
pub mod reverse;
```

Add re-export next to `pub use hybrid::HybridNetwork;`:

```rust
pub use reverse::ReverseHybridPath;
```

Full relevant region of `src/lib.rs` after edit:

```rust
#[cfg(feature = "backends")]
pub mod backends;
pub mod error;
pub mod hybrid;
pub mod projector;
pub mod reverse;
pub mod routing;
pub mod telemetry;
pub mod tensor;
pub mod traits;
pub mod types;

pub use error::{HybridError, Result};
pub use hybrid::HybridNetwork;
pub use projector::{project_spike_activity, spike_activity_features};
pub use reverse::ReverseHybridPath;
// ... rest unchanged
```

- [ ] **Step 3: Run unit tests**

```bash
cargo test --lib reverse:: --all-features
```

Expected: all `reverse::tests::*` **PASS**.

If you intentionally left stubs empty for pure TDD, first run should FAIL with missing methods; then fill in the impl from Step 1 and re-run until PASS.

- [ ] **Step 4: Commit**

```bash
git add src/reverse.rs src/lib.rs
git commit -m "$(cat <<'EOF'
feat(api): ReverseHybridPath reverse-path orchestrator (#23)

Extract corinth-canal Model projector+router half as ReverseHybridPath:
SpikeActivity → project_spike_activity → ExpertRouter → HybridOutput MoE fields.

Co-authored-by: Grok <grok@x.ai>
EOF
)"
```

---

### Task 2: Integration tests (`tests/reverse_path.rs`)

**Files:**
- Create: `tests/reverse_path.rs`

- [ ] **Step 1: Write the integration test file**

Always compiles without `backends`. Uses a local mock `ExpertRouter`. One additional test under `#[cfg(feature = "backends")]` for `SyntheticExpertRouter`.

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for reverse-path orchestration (`ReverseHybridPath`).
//!
//! File-free only: local mock `ExpertRouter` always; optional
//! `SyntheticExpertRouter` when feature `backends` is enabled.
//! No machine-local paths or network.

use hybrid_fusion::{
    ExpertRouteOutput, ExpertRouter, HybridError, ProjectionMode, Result, ReverseHybridPath,
    SpikeActivity,
};

// ---------------------------------------------------------------------------
// Mock ExpertRouter (always available)
// ---------------------------------------------------------------------------

struct MockRouter {
    num_experts: usize,
    top_k: usize,
}

impl MockRouter {
    fn new(num_experts: usize, top_k: usize) -> Self {
        assert!(num_experts > 0);
        Self {
            num_experts,
            top_k: top_k.min(num_experts).max(1),
        }
    }
}

impl ExpertRouter for MockRouter {
    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn top_k(&self) -> usize {
        self.top_k
    }

    fn route(&mut self, embedding: &[f32]) -> Result<ExpertRouteOutput> {
        if embedding.is_empty() {
            return Err(HybridError::InvalidConfig(
                "MockRouter: embedding must be non-empty".into(),
            ));
        }
        // Embedding-dependent but deterministic selection: pick experts by
        // absolute magnitude of successive embedding chunks.
        let n = self.num_experts;
        let mut scores = vec![0.0f32; n];
        for (i, &v) in embedding.iter().enumerate() {
            scores[i % n] += v.abs();
        }
        // Softmax-ish normalize for weight sum ≈ 1
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let expert_weights: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut order: Vec<usize> = (0..n).collect();
        // total_cmp avoids Option/Ordering::Equal pitfalls with NaN.
        order.sort_by(|&a, &b| expert_weights[b].total_cmp(&expert_weights[a]));
        let selected_experts: Vec<usize> = order.into_iter().take(self.top_k).collect();

        Ok(ExpertRouteOutput {
            expert_weights,
            selected_experts,
            routing_entropy: Some(0.42),
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_moe_ok(out: &hybrid_fusion::HybridOutput, embed_dim: usize, num_experts: usize, top_k: usize) {
    assert_eq!(out.embedding.len(), embed_dim);
    for v in &out.embedding {
        assert!(v.is_finite());
        assert!(*v >= -1.0 && *v <= 1.0, "tanh-bounded embedding, got {v}");
    }
    assert!(out.stimuli.is_empty(), "reverse path has no ANN→SNN stimuli");
    let weights = out
        .expert_weights
        .as_ref()
        .expect("expert_weights must be Some on reverse path");
    assert_eq!(weights.len(), num_experts);
    assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    for w in weights {
        assert!(w.is_finite() && *w >= 0.0);
    }
    let selected = out
        .selected_experts
        .as_ref()
        .expect("selected_experts must be Some");
    assert_eq!(selected.len(), top_k);
    let mut seen = std::collections::HashSet::new();
    for &idx in selected {
        assert!(idx < num_experts);
        assert!(seen.insert(idx), "duplicate selected expert {idx}");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn reverse_path_happy_rate_sum_mock_router() {
    let n_neurons = 8;
    let embed_dim = 16;
    let num_experts = 4;
    let top_k = 2;

    let router = MockRouter::new(num_experts, top_k);
    let mut path =
        ReverseHybridPath::new(ProjectionMode::RateSum, n_neurons, embed_dim, router).unwrap();

    let activity = SpikeActivity::from_fired(&[0, 2, 5], n_neurons).unwrap();
    let out = path.forward_activity(&activity).unwrap();

    assert_moe_ok(&out, embed_dim, num_experts, top_k);
    assert_eq!(out.fired_neurons, vec![0, 2, 5]);
    assert_eq!(out.global_step, 1);
    assert!(out.routing_entropy.is_some());
}

#[test]
fn reverse_path_membrane_snapshot_still_routes() {
    let n_neurons = 4;
    let embed_dim = 8;
    let router = MockRouter::new(3, 1);
    let mut path = ReverseHybridPath::new(
        ProjectionMode::MembraneSnapshot,
        n_neurons,
        embed_dim,
        router,
    )
    .unwrap();

    let activity = SpikeActivity {
        spike_train: vec![vec![1]],
        potentials: vec![0.1, 0.5, -0.2, 0.9],
        iz_potentials: vec![],
    };
    let out = path.forward_activity(&activity).unwrap();
    assert_moe_ok(&out, embed_dim, 3, 1);
    assert_eq!(out.fired_neurons, vec![1]);
}

#[test]
fn reverse_path_determinism_same_activity_twice() {
    let n_neurons = 6;
    let embed_dim = 12;
    let router = MockRouter::new(5, 2);
    let mut path =
        ReverseHybridPath::new(ProjectionMode::RateSum, n_neurons, embed_dim, router).unwrap();

    let activity = SpikeActivity::from_fired(&[0, 1, 4], n_neurons).unwrap();
    let a = path.forward_activity(&activity).unwrap();
    // New path for independent step counter; same router math
    let router2 = MockRouter::new(5, 2);
    let mut path2 =
        ReverseHybridPath::new(ProjectionMode::RateSum, n_neurons, embed_dim, router2).unwrap();
    let b = path2.forward_activity(&activity).unwrap();

    assert_eq!(a.embedding, b.embedding);
    assert_eq!(a.expert_weights, b.expert_weights);
    assert_eq!(a.selected_experts, b.selected_experts);
}

#[test]
fn reverse_path_rejects_bad_activity_from_projector() {
    let router = MockRouter::new(2, 1);
    let mut path =
        ReverseHybridPath::new(ProjectionMode::RateSum, 4, 8, router).unwrap();
    // potentials length mismatch vs n_neurons
    let bad = SpikeActivity {
        spike_train: vec![vec![0]],
        potentials: vec![0.0, 0.0], // len 2 != 4
        iz_potentials: vec![],
    };
    assert!(path.forward_activity(&bad).is_err());
    // global_step still increments before project in current design — verify
    // design: step increments first, then project. Spec §5.3 step 1 then 2.
    // If project fails after increment, step is 1.
    assert_eq!(path.global_step(), 1);
}

#[test]
fn reverse_path_new_rejects_zeros() {
    assert!(ReverseHybridPath::new(ProjectionMode::RateSum, 0, 8, MockRouter::new(2, 1)).is_err());
    assert!(ReverseHybridPath::new(ProjectionMode::RateSum, 4, 0, MockRouter::new(2, 1)).is_err());
}

#[cfg(feature = "backends")]
mod with_backends {
    use super::*;
    use hybrid_fusion::SyntheticExpertRouter;

    #[test]
    fn reverse_path_synthetic_router_moe_fields() {
        let n_neurons = 8;
        let embed_dim = 16;
        let num_experts = 4;
        let top_k = 2;
        let router = SyntheticExpertRouter::new(num_experts, top_k).unwrap();
        let mut path =
            ReverseHybridPath::new(ProjectionMode::RateSum, n_neurons, embed_dim, router)
                .unwrap();
        let activity = SpikeActivity::from_fired(&[1, 3, 7], n_neurons).unwrap();
        let out = path.forward_activity(&activity).unwrap();
        assert_moe_ok(&out, embed_dim, num_experts, top_k);
        assert!(out.routing_entropy.is_some());
    }
}
```

**Note:** Spec §5.3 increments `global_step` *before* project/route; if project fails, step is already 1 (asserted in `reverse_path_rejects_bad_activity_from_projector`).

- [ ] **Step 2: Run integration tests without and with backends**

```bash
cargo test --test reverse_path
cargo test --test reverse_path --features backends
cargo test --all-features
```

Expected: all reverse_path tests PASS; existing `hybrid_network` still passes (ANN→SNN MoE fields remain `None`).

- [ ] **Step 3: Commit**

```bash
git add tests/reverse_path.rs
git commit -m "$(cat <<'EOF'
test: reverse-path integration for ReverseHybridPath (#23)

Happy path, MembraneSnapshot, determinism, projector errors, and optional
SyntheticExpertRouter under feature backends.

Co-authored-by: Grok <grok@x.ai>
EOF
)"
```

---

### Task 3: Doc comment touch-up on `HybridNetwork` + docs surface

**Files:**
- Modify: `src/hybrid.rs` (comment only)
- Modify: `docs/implementing-backends.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update `HybridNetwork::forward` MoE comment**

In `src/hybrid.rs`, replace the trailing comment on MoE fields:

From:

```rust
            // Reverse-path MoE fields stay unset on the ANN→SNN forward pass
            // (see ExpertRouter / SpikeActivity; full reverse orchestration is later).
            expert_weights: None,
            selected_experts: None,
            routing_entropy: None,
```

To:

```rust
            // Reverse-path MoE fields stay unset on the ANN→SNN forward pass.
            // Use ReverseHybridPath::forward_activity for activity → MoE.
            expert_weights: None,
            selected_experts: None,
            routing_entropy: None,
```

- [ ] **Step 2: Update `docs/implementing-backends.md` §10**

Replace the obsolete paragraph:

```markdown
`HybridNetwork::forward` leaves `HybridOutput` MoE fields as `None` until a
reverse-path orchestrator lands (follow-ups #26 / #23).
```

With:

```markdown
`HybridNetwork::forward` (ANN → SNN) always leaves MoE fields as `None`.
For the reverse path, use [`ReverseHybridPath`](../src/reverse.rs):

```rust
use hybrid_fusion::{
    ProjectionMode, ReverseHybridPath, SpikeActivity,
    // with --features backends:
    // SyntheticExpertRouter,
};

// let router = my_expert_router; // or SyntheticExpertRouter::new(4, 2)?
// let mut path = ReverseHybridPath::new(
//     ProjectionMode::RateSum,
//     /* n_neurons */ 8,
//     /* embed_dim */ 16,
//     router,
// )?;
// let activity = SpikeActivity::from_fired(&[0, 2], 8)?;
// let out = path.forward_activity(&activity)?;
// assert!(out.expert_weights.is_some());
// assert!(out.selected_experts.is_some());
// assert!(out.stimuli.is_empty());
```

### Plugging in a real MoE backend later

1. Implement `ExpertRouter` with checkpoint-backed gate matmul (engram-parser +
   cortex-tensor — not in this crate).
2. Construct `ReverseHybridPath::new(mode, n_neurons, embed_dim, my_router)`.
3. Call `forward_activity` with live `SpikeActivity` from neuromod / funnel.
4. Do **not** use this path for SAAQ / latent calibration.

**v1 error policy:** reverse-path router/project errors propagate without Sentry
capture (validation-heavy; avoids flooding on bad activity).
```

Also update the TOC intro line if it still says only contracts: mention `ReverseHybridPath` once in the section title area.

Change section heading optionally to:

```markdown
## 10. Reverse path: SpikeActivity + ExpertRouter + ReverseHybridPath
```

And fix TOC entry 10 to match.

- [ ] **Step 3: Update `README.md`**

Under **This crate owns**, add a bullet:

```markdown
- Reverse-path orchestration (`ReverseHybridPath`): SNN activity → embedding → MoE route
  ([#23](https://github.com/rmems/hybrid-fusion/issues/23)).
```

Under **Public surface** table, add row:

```markdown
| `ReverseHybridPath<R>` | Reverse-path host: activity → project → `ExpertRouter` → MoE fields. |
```

- [ ] **Step 4: Update `CHANGELOG.md`** under `### Added`:

```markdown
- `ReverseHybridPath<R: ExpertRouter>` reverse-path orchestrator + integration tests
  (`tests/reverse_path.rs`); dual host alongside `HybridNetwork` (#23).
```

- [ ] **Step 5: Update `AGENTS.md`**

After the `HybridNetwork<T: Transformer, S: SpikingNetwork>` sentence in Trait contracts, add:

```markdown
`ReverseHybridPath<R: ExpertRouter>` is the dual reverse-path host (SNN activity →
project → MoE); do not fold it into `HybridNetwork` generics.
```

Under **This crate owns**, add:

```markdown
- Reverse-path host `ReverseHybridPath` (activity → embedding → ExpertRouter)
```

- [ ] **Step 6: Format, clippy, full test**

```bash
cargo fmt
cargo test --all-features
cargo clippy --all-features -- -D warnings
```

Expected: all green; no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/hybrid.rs docs/implementing-backends.md README.md CHANGELOG.md AGENTS.md
git commit -m "$(cat <<'EOF'
docs: ReverseHybridPath surface and backend plug-in guide (#23)

Co-authored-by: Grok <grok@x.ai>
EOF
)"
```

---

### Task 4: Acceptance check + PR

**Files:** none new

- [ ] **Step 1: Map #23 acceptance criteria**

| Criterion | Evidence |
|-----------|----------|
| Integration test asserts MoE fields | `tests/reverse_path.rs` `assert_moe_ok` + happy path |
| ≥1 ProjectionMode + mock ExpertRouter | RateSum + MembraneSnapshot + MockRouter |
| No machine-local paths / network | pure mocks only |
| Document real MoE backends | `docs/implementing-backends.md` §10 plug-in steps |
| ANN→SNN unchanged | existing hybrid tests; MoE still `None` |

- [ ] **Step 2: Final verification**

```bash
cargo fmt --check
cargo test --all-features
cargo clippy --all-features -- -D warnings
```

- [ ] **Step 3: Push branch and open PR**

Branch name suggestion: `feat/23-reverse-hybrid-path`

```bash
git checkout -b feat/23-reverse-hybrid-path
# if commits were made on main during plan execution, base PR from that branch
git push -u origin HEAD
gh pr create --title "feat: ReverseHybridPath reverse-path orchestrator (#23)" --body "$(cat <<'EOF'
## Summary

- Add `ReverseHybridPath<R: ExpertRouter>` dual host (corinth-canal `Model::forward_activity` projector+router half).
- Integration tests with mock router (always) and optional `SyntheticExpertRouter` (`backends`).
- Docs: implementing-backends plug-in steps; README / AGENTS / CHANGELOG.

Closes #23.

## Test plan

- [x] `cargo test --all-features`
- [x] `cargo clippy --all-features -- -D warnings`
- [x] `cargo fmt --check`

EOF
)"
```

Prefer GitHub MCP for PR create when available (`create_pull_request`). Land with **squash-merge + delete branch** per user preference (human merge only unless asked).

- [ ] **Step 4: Update design spec status** (optional small commit or same PR)

In `docs/superpowers/specs/2026-08-14-reverse-path-orchestrator-design.md`, set:

```markdown
**Status:** implemented (see plan `docs/superpowers/plans/2026-08-15-reverse-path-orchestrator.md`)
```

---

## Spec coverage checklist (plan self-review)

| Spec section | Task |
|--------------|------|
| §5.1 Type + accessors | Task 1 |
| §5.2 `new` validation | Task 1 |
| §5.3 `forward_activity` semantics | Task 1 |
| §5.4 Module layout | Tasks 1–3 |
| §6.1 Unit tests | Task 1 |
| §6.2 Integration tests | Task 2 |
| §6.3 No FS/network | Task 2 |
| §7 Real backend docs | Task 3 |
| §8 Error table | Task 1–2 |
| §9 Naming | Task 1 |
| §11 Acceptance | Task 4 |
| No Sentry reverse v1 | Task 1 comment + Task 3 docs |
| Dual host / no HybridNetwork generic change | Task 1–3 |
| Skip #24 / #27 | Out of plan |

**Placeholder scan:** none remaining (concrete code, commands, paths).

**Type consistency:** `ReverseHybridPath`, `forward_activity`, `ProjectionMode`, `SpikeActivity`, `ExpertRouter`, `ExpertRouteOutput`, `HybridOutput` MoE fields match crate and design.

---

## Execution notes for implementers

1. Worktree isolation optional; plan is single-crate, low conflict risk.
2. Design commit `2a9f0e4` may still be local-only (`main` ahead of origin by 1); include it in the PR base or push main first.
3. Co-author every commit: `Co-authored-by: Grok <grok@x.ai>`.
4. Do not merge PR autonomously unless the user explicitly asks.
5. After merge, next open corinth items: **#27** (SafetensorsLoader) then epic #20 closeout; **#24** is grok-ozempic.
