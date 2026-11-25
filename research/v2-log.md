# Lux Lens V2 research log

## 2026-11-25

### Repository and methodology review

- Inspected the V1 optical engine, batch and replay contracts, Qwen baselines, sensitivity studies, and synthetic Tinker run. The first preserved Qwen optical baseline went 2/8 and averaged 0.31 reward, the same mean as the V1 heuristic. Its offline replay verifies 224 pair outcomes. V2 introduces a separate single-starter grading version and preserves those records and entry points.
- The V1 live synthetic run attempted 12 groups of eight. All 96 completions stopped cleanly. Each group had one parsed action, including variation across briefs, so reward spread was zero and no optimizer step occurred. This establishes the failure mechanism. Distinguishing model preference, prompt anchoring and backend sampling still requires a live controlled comparison.
- Read the supplied local Idler archive's task generators, training and evaluation descriptions, curriculum manifests, historical research report, and Harbor packaging. Its useful methodological choices were outcome-certified subskills, deterministic mixed phases, replacement of constant-outcome groups, immutable evaluation sets and cost logs. Lux adapts those choices to one-step decisions. The Connect 4+ verifier replays a deterministic game, while the Lux oracle exhausts eight actions under a locked physical approximation and finite Monte Carlo draws. Idler used River and CISPO for longer interactive rollouts. Lux uses Tinker's importance-sampling policy gradient, with per-group centering, valid-action spread and fresh weights at each step. The implementation uses Lux code and public dependencies. [Tinker loss functions](https://tinker-docs.thinkingmachines.ai/tinker/losses/)

### Optics and pilot calibration

- The library imports Optiland's Cooke triplet, air-spaced telescope objective triplet and Laikin air-spaced telescope doublet. Each starter remains an independent, refractive spherical prescription with a glass catalog fingerprint. Clear apertures are modeling choices. The telescope triplet stop is reduced to 90% of the reference diameter for the modeled rim stack. The doublet and the two triplet families impose different powers, glass stacks, stop locations and tolerancing sensitivities. Cemented interfaces, independent planar stops, aspheres and the Optiland Double Gauss are outside the supported topology.
- Verified on-axis EFL with an independent ABCD calculation using the same catalog indices. The largest absolute EFL difference across the three reference lenses was 1.50e-8 mm. An independent spherical intersection and vector Snell tracer confirmed nine low-angle sensor intersections with a largest difference of 1.11e-15 mm. These checks cover centered geometric refraction. The manufacturing process, tilted geometry clearance and MTF require separate physical validation.
- A passing and a failing refined Cooke prescription also reproduced the numerical EFL decisions under the ABCD check. Their independent EFL values were 50.166 mm and 51.921 mm for a 50 mm brief with about ±1.4 mm tolerance. The centered Snell tracer agreed on their sensor intersections.
- One all-eight Cooke budget study took 11.9, 43.2 and 83.6 seconds for 7, 25 and 49 trials on this machine. All three budgets found one feasible design. Of 28 pair orderings, 26 agreed between 7 and 25 trials and 25 agreed between 25 and 49. Re-running the 25-trial fit reproduced the frozen objectives and nominal decisions. Use 25 as the pilot's cost-performance compromise and check more cohorts before freezing a production budget.
- The preserved V1 focus study changed from 44/100 to 71/100 passing builds when a common ±0.2 mm travel grid went from 9 to 33 positions. V2 uses a defined assembly proxy. It minimizes the geometric on-axis RMS spot at one wavelength, quantizes sensor placement and clips it to allowed travel before scoring the full optical brief. In one 24-build V2 Cooke study, the fixed sensor passed 20/24 and every tested increment between 60 and 0.75 µm passed 24/24. The pilot's standard increment is 15 µm with ±0.12 mm travel. The result supports the increment on that case. Its transfer to a real fixture is an open question.
- At 32 pupil samples, the four required MTF values for that refined design differed from the 128-sample checks by at most 0.00212. All four tested resolutions agreed on nominal pass for that design. Near-threshold surfaces, field clipping, source spectrum, coatings, ghosting, sensor response and thermal effects need broader study. Ray survival represents sampled geometric transmission through modeled apertures. Manufacturing errors use bounded independent draws and rigid element motion. Their magnitudes, correlations, focus measurement noise and placement repeatability need shop data.

### Exhaustive task generation and evaluation

The 42-brief pilot took 249 seconds with six workers. Sixteen train-only contrast briefs took 211 additional seconds. The combined data contain 58 briefs, 464 graded action slots and 3,840 simulated build outcomes across four draw purposes. Counterfactual profiles reuse nominal refinement when the optical constraints are identical. Every successfully simulated representative brief is retained.

| Split | Representative briefs | Capability briefs | No nominal solution | Other capability rejections |
| --- | ---: | ---: | ---: | ---: |
| Train | 30 | 22 | 6 | 2 |
| Development | 14 | 4 | 10 | 0 |
| Pilot test | 14 | 6 | 8 | 0 |

The training set has 20 A, 12 B, 6 C, 9 D and 22 E tags, with overlap. Four of the six C briefs repeated a nominal-quality/yield reversal of at least 25 percentage points on disjoint training draws. In one standard-adjusted Cooke brief, nominally passing candidate 6 averaged 0.574 across the four required MTF points, compared with 0.529 for candidate 4. Candidate 6 passed 3/8 construction builds and 1/8 independent training builds. Candidate 4 passed 8/8 on both streams. Candidate 4 also had the better yield on development and final draws recorded for that training task. Nine training briefs had disjoint optimal sets after a brief change. Changing the process or permitted assembly compensation reversed one measured preference between two designs. Best-yield sets overlapped under all pilot profile changes.

Held-out test results below use the locked final draw stream, with every method on the same tasks. Regret uses the common cached yield-aware reward. The full evaluation files also report validity, nominal pass rates and failure reasons.

| Control | Target pass, representative | Mean yield, representative | Mean regret, representative | Target pass, capability |
| --- | ---: | ---: | ---: | ---: |
| Uniform random | 19.6% | 18.8% | 0.225 | 45.8% |
| Closest to spec | 28.6% | 27.7% | 0.133 | 66.7% |
| Physics heuristic | 28.6% | 27.7% | 0.134 | 66.7% |
| Ridge ranker | 28.6% | 28.6% | 0.125 | 66.7% |
| Exhaustive oracle | 42.9% | 41.1% | 0 | 100% |

The physics heuristic improved on closest-to-spec in the development ancestry, from 2/14 to 4/14 target passes. The two rules both passed 4/14 in the pilot test ancestry. Ancestry and application co-vary in these splits. There is one independent reference lineage per held-out split, so paired confidence intervals are unavailable. Eight builds per action per stream give a coarse yield estimate. A larger confirmation should hold out several independent lineages, cross application/process conditions across lineages, and regrade borderline designs with more draws and finer pupil sampling.

### Offline end-to-end checks

The local suite exercises the V1 and V2 paths, including strict JSON, nominal and yield reward ordering, split leakage checks, deterministic build errors, physical focus equivalence, stored sensor coordinates, freeze/reload of the shared cached contract, mock dynamic sampling and mock checkpoint resume. Two cold six-brief runs reproduced the same task index and byte-identical replay inputs. A real Qwen tokenizer from the checkout's cache verified the non-thinking chat template. Measured mean prompt lengths on the 14 development briefs were 1,226 summary, 2,386 full and 3,788 physics tokens. At the published uncached rate those cost approximately $0.405, $0.788 and $1.250 in prefill per thousand requests. All 58 physics prompts fit the 4,096-token cap. Model quality comparisons across representations await live sampling. [Tinker model pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)

The complete eight-candidate replay is `outputs/v2-pilot/complete-replay.json.gz`. It stores Optiland paths, spot positions and MTF curves, and includes one offline heuristic/oracle comparison. The corresponding figure is `outputs/v2-pilot/replay-figure.png`. The Harbor export ran its generated oracle and malformed-response tests locally, returning 1 and 0. Docker/Harbor container orchestration awaits verification on a host with those tools. [Harbor task guide](https://github.com/harbor-framework/docs/blob/main/guides/creating-tasks.mdx)

### Bounded live plan

The local Tinker client has credentials configured. No paid inference, training or checkpoint creation was performed for V2. Base-model reward distributions, collapse attribution, the first real update and a trained-versus-untouched comparison remain open.

| Proposed experiment | Maximum sampled completions | Conservative application estimate |
| --- | ---: | ---: |
| Three briefs × three representations × three temperatures × two sampling strategies | 432 | $1.79 |
| One-step smoke, up to eight attempted groups of eight | 80 including evaluation | $0.59 |
| Twenty-step matched experiment, per reward arm | 2,640 including evaluation | $17.22 |
| Three matched pilot test inference runs together | 84 | $0.35 |

Estimates assume 4,096 prompt tokens, 32 generated tokens, uncached pricing, a 3× token-cost margin and a per-checkpoint storage reserve. They are application estimates. We propose exploration followed by one smoke at a combined estimate of $2.39. If the smoke observes a real probability shift and verifies reload, the example matched experiment uses rank 16, learning rate 2e-5, group size 8, target 4 accepted groups, minimum 2 and at most 16 attempts per step. Choose temperature and representation from the measured exploration results and keep those fixed across reward arms. Interpret the 20-step run as an exploratory feasibility check. Use fresh sealed test cases and more independent reference lineages for a generalization claim. [Tinker model pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)
