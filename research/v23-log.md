# Lux Lens V2.3 research log

## 2026-11-26 · Benchmark regeneration and matched development pilot

The benchmark uses 128 samples across the pupil diameter. When any required MTF measurement falls within 0.01 of its cutoff, the simulator repeats that optical measurement at 256 and uses the result. This rule applies to nominal measurements during refinement and to manufactured draws. The optimizer receives 25 objective trials per starter including the initial trial, with the same coordinate order and bounds for every candidate and both model arms. The 49-trial results are a sensitivity check on the frozen problem.

The development panel draws from six separate prescription histories. The 32 Cooke training briefs share one ancestry. The Steinheil and Gauss use the same crown/flint glass pair and share a published table. They are different classical forms with reversed glass order and different element shapes and surface powers. The TOA-type model has two ED positive elements and a wide first air gap. The Herzberger–Hoadley uses fluorite positive elements and a modeled dense barium crown. Beecher–Miles uses the front objective only, including a nominal 1 mm negative-element thickness, a catalog glass substitution, a flat sensor at calculated paraxial focus, and the stated descriptive R3. The Laikin triplet uses two positive BK7 elements and a rear SF1 negative element. See the recipe records for radii, spacings, modeled stops and rims, glass hashes and source details. [Lineage recipes](v23-lineage-recipes.json) · [Achromat Table 11](https://optics.udjat.nl/achromats.html) · [TOA-type Table 12](https://telescope-optics.net/semiapo_and_apo_examples.htm) · [Herzberger and Hoadley](https://patents.google.com/patent/US2487873A/en) · [Beecher and Miles](https://patents.justia.com/patent/4877318)

Each lineage contributes its first surviving representative brief and one construction-prioritized challenger, each under fixed and adjusted focus process profiles. Initial construction results showed sparse competition across the original brief ranges. The supplemental development sweep used tighter perturbations around each published reference and calibrated optical thresholds to bring more candidates into the competition. The selected challenger ranks alternatives by the presence of a qualifying choice, the number of profiles with at least a 25-point simulated-yield gap straddling the target, sharpness-to-yield reversals, multiple nominally feasible competitors, physics-heuristic regret of at least 0.10, qualifying profiles, then total reward range. The selection uses eight construction draws. Development scores use 32 fresh draws per nominally passing design. Training scores use 16 draws with a separate purpose. Designs within a brief share error quantiles for paired comparisons. Counts and intervals describe Monte Carlo results under the assumed manufacturing process.

## Measured benchmark results

The main sweep generated 80 briefs in 899 wall seconds. The supplemental sweep generated 24 in 442 wall seconds while overlapping the main run. Both used seven workers and had zero physical rejections. The frozen pilot contains 32 training briefs and 24 development briefs. [Frozen manifest](../outputs/v23/pilot/manifest.json) · [Offline evidence](v23-evidence.json)

| Development measure | Result |
| --- | ---: |
| At least one qualifying starter | 14/24 |
| At least two nominally feasible starters | 22/24 |
| At least two starters meeting simulated yield target | 12/24 |
| Reward range of at least 0.10 | 22/24 |
| Exact uniform selection pass rate | 26.56% |
| Closest to spec and physics heuristic pass rate | 45.83% each |
| Cooke-trained ridge ranker pass rate | 41.67% |
| Oracle pass ceiling | 58.33% |

Five development briefs across Gauss, Herzberger–Hoadley and Beecher–Miles have two nominally feasible designs with yield estimates at least 25 points apart on opposite sides of the 62.5% threshold. In the adjusted Beecher–Miles challenger, starters 5 and 3 passed 2/32 and 30/32 paired builds. Their Wilson intervals were 1.7–20.1% and 79.9–98.3%. All 28 discordant draws favored starter 3. The five development briefs tagged as sharpness-to-yield reversals during construction did not repeat a 25-point reversal on the independent scoring draws. The panel measures robust-versus-fragile selection, with limited evidence on accepting lower nominal sharpness for better yield. No starter reaches the yield threshold on the four wide-gap ED briefs. The other five lineages have at least one qualifying brief.

The optimizer audit used all eight starters on one fixed-process representative brief per lineage. Extending 25 trials to 49 increased nominal passes from 23/48 to 33/48, preserved 133/168 pairwise merit orderings and retained the best nominal-feasible starter in each of the six briefs. The 49-trial refits used 718 CPU seconds. The audit supports a fixed short-budget estimand and shows that feasibility and lower-ranked positions depend on the trial budget. Median CPU time per fixed-prescription measurement was 0.311 seconds at 128 and 0.508 seconds at 256, about 1.63× for a standalone 256 measurement. In the 3,584 development build evaluations, 184 triggered the 256 recheck and 12 individual required MTF comparisons changed sides of their thresholds. [Optimizer audit](v23-optimizer.json) · [Offline evidence](v23-evidence.json)

A separate pupil audit repeated all 96 unique frozen nominal designs at both 128 and 256, covering 384 required MTF comparisons. Every nominal pass label and every MTF threshold comparison agreed. The largest absolute MTF error was 0.15973 on a strongly aberrated wide-gap ED candidate. Its on-axis 30 lp/mm value was 0.160255 at 128, 0.000526 at 256 and 0.000122 at 512, all below the 0.4624 requirement. This limits quantitative use of 128-grid MTF far from specification in this lineage. The panel supports comparison of threshold decisions under the frozen protocol. The independent build outcomes retain the 256-grid rechecks for borderline cases. [Pupil cross-check](../outputs/v23/pupil-cross-check.json)

The offline CLI checks passed for both training arms and the full evaluation. A Harbor bundle for the adjusted Beecher–Miles case exported and its shipped verifier graded starter 3 at 0.940625.

## Matched training plan for review

Use the Tinker Qwen/Qwen3.5-4B non-thinking renderer with full optical prompts, one strict-format example, temperature 1.25, individual requests, LoRA rank 16 and learning rate 0.00002. Use the V2.2 exploration to calibrate the mixed schedule, then freeze it across the two reward arms. Weight the A/B/C/D/E families 15/15/30/10/30 before availability and difficulty adjustment. Shuffle the eight candidates independently for each attempted training group with a deterministic seed common to the two arms. The starter IDs remain attached to their designs.

| Planned quantity | Yield-aware | Random-reward control |
| --- | ---: | ---: |
| Attempted training groups | 48 | 48 |
| Attempted training completions | 384 | 384 |
| Maximum admitted groups | 12 | 12 |
| Optimizer updates | Up to 6 | Up to 6 |
| In-run development completions | 32 | 32 |
| Full development evaluation completions | 48 | 48 |

Both arms require useful assigned-reward variation among valid answers and an optical reward range of at least 0.10 on those answers. The control assigns independent seeded Bernoulli rewards with probability 0.5. An arm can admit fewer groups or skip an optimizer step when no group qualifies. Report attempted groups, valid actions, groups that satisfy the optical spread threshold, admitted groups, actual training datums, actual optimizer updates, and checkpoint reload checks per arm.

Evaluate untouched Qwen and both end checkpoints on all 24 development briefs. Use one completion each on the frozen order and on its four-position rotation, with identical task-specific seeds and decoding across arms. Score independently using the same frozen simulation outcomes. Report all-brief and oracle-feasible pass rates, continuous reward, per-lineage and per-permutation effects, invalid rates, the manufacturing reversal subset, exact uniform random and engineering heuristics. Use the end checkpoint for a single development decision. The in-run eight-brief readouts serve as diagnostics.

- **Continue** if the yield-aware checkpoint records at least four useful optimizer updates, exceeds both untouched Qwen and the random-reward checkpoint by at least 10 percentage points on all briefs, improves over random reward on at least four lineages, shows a positive gain over random reward on both permutations, and increases the invalid rate by at most five points versus untouched Qwen. This is a descriptive development gate. The sample of lineages and completions does not establish statistical generalization.
- **Modify** if optical variance or qualifying updates are scarce, the gain depends on one order or a small subset of lineages, or the engineering heuristic leaves little demonstrable headroom. Inspect failure categories and paired manufacturing tasks before revising the prompt or curriculum.
- **Abandon this reward and prompt recipe** if two separately approved matched seeds on a corrected development panel fail to exceed the random-reward control, or if the manufacturing comparisons continue to draw too few distinct valid choices after correction.

The fixed schedule uses all 15 construction-certified training briefs, including seven with construction manufacturing reversals. Across 48 attempts the sampled family labels are A 6, B 9, C 13, D 4 and E 16. All 48 scheduled tasks have at least 0.10 reward range over the eight choices on independent training draws. This is the available signal before rollout. The offline Qwen tokenizer checked 48 training presentations at 2,400–2,509 tokens and every development permutation at 2,134–2,518 tokens, below the 4,096 cap.

The application estimate is $2.719360 per training arm and $0.199273 for each 48-completion development evaluation, or $6.036539 combined. These are preflight estimates for a proposed run. Untouched Qwen previously passed 50.9% overall and 76.3% on oracle-feasible briefs in the V2.2 exploration. Its accuracy on the regenerated panel will be measured by the separately approved baseline evaluation.

No paid calls were made for V2.3. The final evaluation panel was untouched. The local preflight estimates are an application budget with a 3× token margin and checkpoint storage reserves. The plan requires separate spending approval. If approved, run the untouched Qwen evaluation first. A base result of 24/48 or better on the frozen panel would make the 10-point gate unreachable because the oracle can pass at most 28/48. In that case revise the development gate or panel before running either training arm. [Training settings](v23-matched.json) · [Matched development evaluation settings](v23-evaluation.json) · [Previous model evidence](v22-log.md)

## Reproduction

Run the optics and grading commands locally. Use a new output directory if re-running a published manifest.

```bash
export MPLCONFIGDIR=.cache/matplotlib
base_flags=(--workers 7 --evaluations 25 --pupil-samples 128 --borderline-mtf .01
  --builds-per-phase 8 --scoring-builds 16 --profiles standard-fixed standard-adjusted)
.venv/bin/python -m lux_frontier.v2 generate outputs/v23/regenerated \
  --lineages research/v23-lineage-recipes.json "${base_flags[@]}"
jq --slurpfile contest research/v23-contest-recipes.json \
  '[.[] as $base | $contest[0][] | select(.ancestry == $base.prescription.ancestry) |
    $base + del(.ancestry)]' research/v23-lineage-recipes.json > outputs/v23/contest-input.json
.venv/bin/python -m lux_frontier.v2 generate outputs/v23/contests \
  --lineages outputs/v23/contest-input.json "${base_flags[@]}"
.venv/bin/python -m lux_frontier.v2 development-panel outputs/v23/regenerated/manifest.json \
  --contests outputs/v23/contests/manifest.json --output outputs/v23/curated
.venv/bin/python -m lux_frontier.v2 manufacture outputs/v23/curated/manifest.json \
  --phase development --fidelity investigation --seed 28301 --workers 7 --output outputs/v23/pilot
PYTHONPATH=. .venv/bin/python research/v23-audit.py \
  outputs/v23/curated/manifest.json research/v23-optimizer.json --optimizer-workers 5
PYTHONPATH=. .venv/bin/python research/v23-audit.py \
  outputs/v23/pilot/manifest.json outputs/v23/pupil-cross-check.json --pupil-workers 7
HF_HUB_OFFLINE=1 HF_HUB_CACHE=.cache/huggingface/hub PYTHONPATH=. \
  .venv/bin/python research/v23-audit.py outputs/v23/pilot/manifest.json research/v23-evidence.json
.venv/bin/python -m lux_frontier.v2 train outputs/v23/pilot/manifest.json \
  --settings research/v23-matched.json --reward yield
.venv/bin/python -m lux_frontier.v2 train outputs/v23/pilot/manifest.json \
  --settings research/v23-matched.json --reward random
.venv/bin/python -m lux_frontier.v2 sample outputs/v23/pilot/manifest.json \
  --settings research/v23-evaluation.json --split development
```

## 2026-11-26 · Untouched Qwen development baseline

The approved Tinker run completed all 48 requests with a buffered application estimate of $0.110399. It used the frozen development manifest and the registered 0- and 4-position orders. No checkpoint or training was run. Binary pass was 21/48 overall and 21/28 on oracle-feasible brief/order pairs. Mean simulated yield was 45.77% over all responses, with zero assigned to invalid or nominally failed selections, and 53.58% among the 41 nominally passing choices. Mean regret was 0.11876 on the continuous reward scale. Both engineering heuristics passed 11/24 briefs. Closest to spec and the physics heuristic had mean simulated yields of 49.61% and 49.74% and mean regrets of 0.07886 and 0.07762, respectively. Qwen matched both on the five manufacturing-gap briefs at 80% pass and 71.88% mean simulated yield. [Graded evaluation](../outputs/v23/untouched-qwen-evaluation.json) · [Run ledger](../outputs/v23/untouched-qwen-baseline.json)

| Lineage | Frozen order, passes / 4 | Rotated order, passes / 4 | Qwen mean simulated yield | Qwen mean oracle regret | Heuristics, passes / 4 each |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gauss | 3 | 3 | 69.53% | 0.007 | 3 |
| Steinheil | 3 | 1 | 46.09% | 0.227 | 3 |
| Wide-gap ED | 0 | 0 | 8.59% | 0.015 | 0 |
| Herzberger–Hoadley | 2 | 3 | 72.27% | 0.019 | 2 |
| Beecher–Miles | 1 | 1 | 32.03% | 0.335 | 1 |
| Laikin telescope triplet | 2 | 2 | 46.09% | 0.109 | 2 |

The original and rotated orders passed 11/24 and 10/24, respectively. Two briefs switched from pass to fail and one switched from fail to pass. Of 23 pairs with two valid responses, 18 retained the same starter and one retained the same display position. The largest lineage order difference was Steinheil's two-brief decline. The paired observations are descriptive.

The pass-rate gate is arithmetically possible. A score of 26/48 is needed to improve on untouched Qwen by 10 points, against an oracle ceiling of 28/48. Seven additional passes are possible across the observed brief/order pairs. Four are in Beecher–Miles, two in Steinheil and one in Herzberger–Hoadley. The other three lineages offer continuous reward headroom only against this baseline. The measured regret supports an exploratory learning hypothesis, but the binary gate requires capturing at least five of seven available passes and leaves little margin. The four-lineage criterion compares to the unmeasured random-reward arm. Keep the training run on hold for review and consider pre-specifying an oracle-feasible or continuous-regret endpoint before using this pilot as a binary continuation decision. [Machine-readable summary](../outputs/v23/untouched-qwen-summary.json)
