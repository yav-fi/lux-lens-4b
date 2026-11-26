# Lux Lens V2.2 research log

## 2026-11-25 · Research-only execution

The V2.1 HTML, CSS, JavaScript, local server, browser-ready data copies and presentation-only test remain on disk under scoped Git ignores. None were in the Git index. The package data declarations and default test suite have no presentation dependency. Backend ray exports, histories, manufacturing records, grading, and Harbor remain available.

### Authorized live exploration

The frozen scout uses Qwen3.5-4B with thinking disabled. It samples the same two feasible development briefs and one no-solution control across three prompt representations, two temperatures, both sampling strategies and two display rotations. There are 72 groups of eight, for 576 completions. The first two briefs have 32-build yield checks and exact random chances of 37.5% and 25.0%. The negative control has zero physical solutions. All three briefs share one optical ancestry.

The V2.1 gate calls for meaningful optical spread across multiple briefs and stability under order rotation. Before examining the outcomes, the V2.2 readout operationalized this per representation, temperature and sampling lane. Each feasible brief must produce at least one group with a valid-only reward range of at least 0.10. Invalid responses must be at most 10%, the pooled feasible pass difference between rotations must be at most 25 percentage points, and the lane must reach the exact feasible-panel random pass rate of 31.25%. This is a plumbing gate for one optimizer smoke. The panel is too small to establish a representation advantage or performance across optical lineages. The no-solution task is reported separately.

### Pupil convergence and the 0.426 MTF change

Remeasured all 24 refined candidates in one Cooke triplet, one telescope triplet and one telescope doublet brief. The comparison uses 256 samples across the pupil diameter as its numerical reference. The threshold subset contains three designs within 0.03 MTF of a target with no other failing constraint. All 24 nominal pass labels agreed across 32, 64, 128 and 256. Agreement on labels masks substantial measurement error at 32.

| Samples per pupil diameter | Maximum absolute MTF difference from 256 | Maximum near-threshold difference | Median CPU seconds per prescription |
| ---: | ---: | ---: | ---: |
| 32 | 0.4266 | 0.0500 | 0.217 |
| 64 | 0.0193 | 0.0128 | 0.216 |
| 128 | 0.0040 | 0.0040 | 0.270 |
| 256 | reference | reference | 0.455 |

The largest discrepancy is candidate 3 on the inspected development triplet. Its tangential MTF at 1°, 0.55 µm and 30 lp/mm is 0.427553 at 32 samples. Changing FFT padding to 2×, 4× or 8× at the same pupil density leaves this value unchanged. Increasing density yields 0.002119 at 64, 0.001796 at 128, 0.000966 at 256 and 0.000688 at 512. The maximum adjacent unwrapped phase change falls from 7.31 waves at 32 to 0.493 at 512. At 32, 89.3% of adjacent valid pupil pairs exceed half a wave. The padding control and wavefront measurements support pupil-phase aliasing in this case. The absolute MTF near zero stabilized much earlier than the worst local phase slope. [Optiland scalar diffraction and MTF methods](https://optiland.readthedocs.io/en/stable/api/api_wavefront.html)

Use 128 across prospectively generated prompt features, refinement and manufacturing labels. It cost about 1.24× the 32-grid median and stayed within 0.004 MTF of 256 in this panel. Predeclare a 0.01 MTF uncertainty band and confirm borderline results at 256. Raise the resolution farther when 128→256 changes a decision or exceeds the allowed MTF error. This is a measured protocol choice for the studied optics. Reflective, aspheric and more strongly aberrated optics require their own convergence checks. The live scout still uses its previously frozen V2.1 features and labels.

The 24 initial prescriptions used in the scout were also remeasured at 128. The maximum initial-feature MTF changes were 0.162 and 0.162 on the two feasible briefs and 0.315 on the no-solution brief. The closest-to-spec rule chose the same candidate on all three after remeasurement. The differences still matter for prompt learning and warrant regeneration before a training study.

### Optimizer budget

Ran each of 24 starters through 97 trials and read the incumbent after 7, 25, 49, 73 and 97. Each checkpoint ends at a completed trial pair. The 25-trial search objective exactly matched the original frozen result. Every incumbent was separately measured at 128. The three eight-way scans took about 130 wall seconds on four workers, with 455 CPU seconds for the search.

| Brief | Nominal passes at 7 / 25 / 49 / 73 / 97 | Pairwise score order agreement, 25 vs. 97 |
| --- | --- | ---: |
| Telescope triplet | 2 / 2 / 2 / 2 / 2 | 28/28 |
| Telescope doublet | 1 / 5 / 6 / 7 / 7 | 21/28 |
| Cooke triplet | 2 / 4 / 4 / 5 / 4 | 25/28 |

The 32-grid optimizer could choose a later Cooke incumbent near the MTF boundary that failed the 128-grid check. Five targeted 128-grid refits showed the two doublet late-pass cases reaching feasibility by trial 49, and the boundary Cooke candidate 3 passed at 49, 73 and 97 with 128-grid search. The same candidate's 32-grid search produced a passing incumbent only at the 73-trial checkpoint under 128-grid remeasurement. The higher-grid refits averaged 0.235 CPU seconds per objective trial. For the next study, freeze the budget before training and compare a 25-trial reference with a 49- or 73-trial sensitivity panel on independently sourced lineages. A small-budget benchmark defines a particular optimization problem. Its rankings cannot serve as a proxy for arbitrary optimization budgets.

### Independent Cooke manufacturing confirmation

Held the two previously selected nominally successful Cooke prescriptions fixed. Generated 128 fresh, shared error draws under the specified tolerance distribution on a purpose-specific stream. Both passed the nominal requirements at 32, 128 and 256. The 128-grid manufacturing count gave 28/128 (21.9%, Wilson 95% interval 15.6–29.8%) for candidate 6 and 108/128 (84.4%, 77.1–89.7%) for candidate 4. The difference is 62.5 percentage points. All 80 discordant pairs favored candidate 4 (exact paired p = 1.65e-24 for this simulated comparison). The first 32 paired error draws produced identical pass decisions at 32, 128 and 256 for both prescriptions. Most failures were off-axis MTF violations. The 128-grid measurements averaged 0.362 CPU seconds per simulated build. These measurements confirm a simulated manufacturing effect for this pair under the chosen process assumptions.

### Independent lineages and compute

Start with two- or three-element spherical air-spaced prescriptions that document all radii, spacings, aperture and material dispersion. Screen published tables and downloadable stock or patent models before requesting a larger corpus. Verify sign conventions and units, compare paraxial focal length and exact on-axis rays, then check nominal feasibility, decision headroom, and convergence at 128/256. Group sister patents, successive zoom positions, shared layouts with provenance links, and near-duplicate normalized geometry before assigning splits. Freeze whole families across splits. Aperture and focus adaptations require explicit records and review. A known-source candidate with a clear manufacturing contrast is more useful than another set of prompts for the Cooke geometry. [Edmund Optics catalog](https://www.edmundoptics.eu/products/services/zemax-catalog/) · [Photonium corpus scope and access](https://photonium.ai/research/the-photonium-patent-corpus/index.html)

For a concrete compatibility scout, two published doublet forms were transcribed and independently ray-checked. The Steinheil puts negative flint in front of positive crown, and the Gauss uses a positive crown meniscus followed by a negative flint meniscus. Modeled rims, front stops and the green paraxial sensor positions are recorded as assumptions. Their exact low-angle focal lengths were 100.326 and 99.884 mm, within 1.6e-9 mm of the corresponding ABCD checks. Both are development candidates. They will never be part of a sealed test panel once inspected here. The published binocular objective in US4877318 is a lower-priority source because its text and claim give different radii and the unfolded path requires independent reconciliation. [Published doublet recipes, Table 11](https://optics.udjat.nl/achromats.html) · [US4877318 patent text](https://patents.justia.com/patent/4877318)

At 128 pupil samples and 25 refinement trials, eight scout briefs crossed two field settings with two process profiles. The eight-candidate generator had zero physical rejections, and five of eight briefs were certified for capability by construction. Closest to spec matched the construction oracle reward on seven of eight. On independent eight-build development draws, three briefs had no passing choices. The two forms establish compatibility and physical variety. They do not yet supply a sufficiently difficult evaluation, and their shared source and glass pair warrant conservative independence accounting. The complete scout took 128 wall seconds on four workers. [Screened recipes](v22-lineage-recipes.json)

For five primary contrasts, eight independent test lineages is only the resolution floor for a two-sided exact sign-flip test at 0.01. If the minimum meaningful benefit is 15 percentage points and desired power is 80%, a normal-approximation plan at that comparison level calls for 12, 33 or 64 lineages at assumed between-lineage paired SDs of 0.15, 0.25 or 0.35. These are scenarios. Start with roughly six to ten independent development lineages to estimate variation, and size the final panel from that evidence. Grouping related briefs is mandatory even when a lineage generates many tasks.

With four briefs per lineage, two shared fits across paired process variants, eight candidate starters per fit and around three feasible candidates per brief, 100 final draws per feasible candidate require about 7.2 CPU minutes per lineage at the measured 128-grid build rate. Nominal fitting adds about 1.6 minutes at 25 trials or 4.6 minutes at 73, for roughly 9–12 CPU minutes per lineage. At the longer budget, 12/33/64 lineages would take roughly 2.4/6.5/12.6 CPU-hours before extra screening or boundary confirmation. Source and material verification remains the larger unmeasured cost.

### Reproducible local work

The raw Tinker ledger and group journal are in `outputs/v22`. Reference inputs, study outputs and generated ray and manufacturing records remain in their original frozen manifests or separate study envelopes. The research suite passed 145 tests. A fresh Harbor export and its shipped verifier ran against the selected solution and returned the expected reward of 0.88125.

```bash
rtk proxy .venv/bin/python -m lux_frontier.v2.exploration \
  outputs/v22/base-exploration.json outputs/v21-dev-32/manifest.json \
  --output outputs/v22/exploration-readout.json
rtk proxy .venv/bin/python -m lux_frontier.v2.optical_study factory \
  outputs/v21-expanded/manifest.json --task train-cohort000-contrast-spec006-standard-adjusted \
  --output outputs/v22/cooke-confirmation-reproduction.json
rtk proxy .venv/bin/python -m lux_frontier.v2 generate outputs/v22/lineage-reproduction \
  --lineages research/v22-lineage-recipes.json --workers 4 --scenarios 2 --pools 1 \
  --evaluations 25 --pupil-samples 128 --builds-per-phase 8 --profiles precision-fixed standard-adjusted
```

## 2026-11-26 · Live results reviewed

### Untouched Qwen

| Panel | Qwen | Exact random | Closest to spec | Physics heuristic |
| --- | ---: | ---: | ---: | ---: |
| All three briefs, 576 completions | 293/576 (50.9%) | 20.8% | 33.3% | 66.7% |
| Feasible briefs, 384 completions | 293/384 (76.3%) | 31.25% | 50.0% | 100% |
| No-solution control, 192 completions | 0/192 | 0% | 0% | 0% |

The two feasible briefs had pass rates of 92.2% and 60.4%. Families A and E both contain those same two briefs. Family B contains the 92.2% brief. Families C and D had no exploration briefs. Repeated prompts and overlapping labels are not independent observations, so these are descriptive contrasts for this exact panel.

There were 25 invalid replies (4.3%). Groups averaged 1.61 distinct valid IDs, with 43 groups yielding one ID and two yielding none. Mean valid-only reward variance was 0.0336 overall and 0.0503 on feasible briefs. The original exact-spread flag counted 23/72 informative groups, including ten no-solution groups with only tiny near-miss differences. Only 13/72 groups met the 0.10 useful-spread threshold, all on feasible briefs (13/48, 27.1%). This provides a selective learning signal. It also shows why nonzero variance alone overstates useful training capacity.

Rotating the feasible prompts by four positions dropped pass rate from 89.6% to 63.0%. Among 180 paired valid completions, 86 retained the same design ID and 27 retained the same position. The sweep is too small to attribute the drop to a single position or isolate candidate-order effects from decoding interactions. The only lane to clear the smoke gate was full prompts at temperature 1.25 with independently seeded requests. It had useful variance in all four feasible groups, 56.25% pass, 9.4% invalid and a 25-point rotation difference. This lane favors exploration over immediate accuracy and is provisional.

### Verified optimizer smoke

The live smoke admitted two of four training groups and used 15 completion datums in one Tinker optimizer update. The largest fixed-completion token log-probability change was 0.19232, and the sum of the completion log probabilities changed by −0.31554. A new sampler reloaded the saved weights and reproduced the probed token log probabilities exactly. Development pass was 25% before and after on the four-brief smoke panel. The robust Cooke reversal group provided no useful contrast because all eight replies chose candidate 6, the lower-yield design. The admitted groups came from nominal feasibility and optimization potential. The smoke verifies a real parameter update and checkpoint reload. It leaves evidence of learned manufacturing judgment open.

Exploration recorded 576 completions and $1.424404 in buffered token estimates. The smoke recorded 48 completions and $0.210925 in buffered token estimates, plus a $0.10 checkpoint storage reserve. Combined application-accounted usage and reserve were $1.735329, below the approved $3.00 ceiling. These ledger estimates use the published token rates with a 3× margin. Provider invoices were not reconciled in this run. [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)

### Recommended next experiment

First regenerate the relevant prompts, refinement results and manufacturing labels at 128 samples per pupil diameter, with 256 for the numerical boundary checks. Freeze a single optimization budget and audit a larger budget on a separate sensitivity subset. Source enough genuinely distinct prescriptions for a prospective development panel across at least six lineages. Emphasize manufacturing robustness and engineering trade-offs where the physics heuristic has headroom. Keep a representative panel as well so results do not depend solely on selecting failures.

Then seek approval for a two-arm, six-step pilot with untouched Qwen as a third evaluation comparator. The two LoRA runs use continuous optical reward or seeded random reward, the same seed and exactly eight attempted groups of eight per step, with up to two accepted groups. Retain adaptive curriculum calibration in development, then freeze its mixed allocation across the arms for matched exposure. Evaluate the prespecified end checkpoint on up to 24 fresh development briefs with two samples per arm. Report admitted groups and actual updates for each arm. This estimates a small directional effect before spending on the nominal-only comparison, a second seed, or a fully powered test panel.

The conservative estimate is $2.71936 per training arm plus $0.597819 for the three-way evaluation at 24 briefs, or **$6.04 total**. If it finds a gain, rechecking with rotated prompts would cost up to another $0.60 at the same sample count. The proposed settings use the smoke lane provisionally. [Proposed settings](v22-controlled-pilot.json)

- **Continue** if the yield-aware arm takes at least four useful optimizer updates, exceeds both untouched and random-reward Qwen by at least 10 percentage points on the prospectively selected overall panel, and improves over the random-reward control in at least four of six lineages. Check the gain under candidate-order rotation and require an invalid-rate increase of no more than five points. Also report the comparison with the physics heuristic and the manufacturing-robustness subset.
- **Modify** if manufacturing-robustness groups still collapse, the gain is limited to one lineage or one order, or the heuristic already exhausts the useful headroom. Rebalance families and improve the initial physical features before increasing the optimizer budget.
- **Abandon the current reward-and-prompt recipe** if two matched seeds on a corrected panel fail to exceed the random-reward control, or if the model cannot get useful valid variation on the manufacturing choices even after feature correction and prompt reordering.

The matched pilot is proposed and awaits separate approval. The live ledger, optical convergence, manufacturing statistics and simulation timings are in the [V2.2 evidence record](v22-evidence.json).
