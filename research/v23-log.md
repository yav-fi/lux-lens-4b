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

## 2026-11-26 · Matched pilot registration before either adapter trains

These complementary criteria were established after the 48-response untouched-Qwen baseline and before training. The benchmark, optical grader, 62.5% simulated manufacturing yield threshold, and registered development presentations are frozen. The comparison uses the two independently created Qwen3.5-4B adapters from the same unadapted model and seed, with thinking disabled and the strict one-starter JSON contract. Use the scheduled end-of-step-six checkpoint from each arm. The full evaluation weights all 24 development briefs equally, with one response per registered candidate order. The 48 responses span six optical lineages. The five manufacturing contrast briefs are defined by the previously frozen evidence of a gap of at least 25 points that straddles the yield threshold.

- Retain a gain of at least 10 percentage points in overall binary pass rate over both untouched Qwen and the random-reward arm as an ambitious, descriptive target. Separately report binary performance on the prespecified 14 oracle-feasible briefs.
- Compare the yield-aware arm's mean simulated yield with both model comparators. Reduce shaped-reward oracle regret by at least 25% relative to untouched Qwen, from 0.1187627 to at most 0.0890720. Calculate both metrics over the full paired matrix.
- Inspect the mean-yield contrasts for both orders and all six lineages. A broadly distributed positive signal requires positive yield contrasts against each comparator on both orders and in at least two lineages per comparator. Report all lineage and order results.
- Check genuine training signal using valid-only optical reward span of at least 0.10, within-group optical and assigned reward variance, and distinct valid selections. Report attempted and admitted groups and optimizer updates in both arms. Use four or more actual updates in the yield-aware arm with verified output-probability movement and checkpoint reload as the training adequacy check.

Report mixed evidence if the criteria disagree. The planned final analysis keeps both engineering heuristics, the oracle, the two registered orders, all 24 development briefs and the baseline. The final panel is outside this study. The per-arm checkpoints include resumable optimizer state and sampler weights with a 24-hour TTL. The additional application estimate is $5.837266, including storage reserves and both full development evaluations. The sum of configured application caps is $5.84. [Preflight record](../outputs/v23/matched/preflight.json)

## 2026-11-26 · Matched pilot result

Both fresh adapters completed all six scheduled steps. All 48 attempted groups per arm matched the registered tasks, candidate permutations, prompt hashes and sampling seeds. The first pre-update probe matched exactly across arms and the 16 initial diagnostic selections also matched. Each arm sampled 384 training completions and 32 in-run development completions. Both end checkpoints evaluated on the same 48 requests as the untouched baseline, representing 24 distinct development briefs, two candidate orders and six independent optical lineages.

| Method | Binary pass | Oracle-feasible pass | Mean simulated yield | Mean shaped oracle regret | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untouched Qwen | 21/48 (43.75%) | 21/28 (75.00%) | 45.77% | 0.11876 | 1/48 |
| Random-reward GRPO | 20/48 (41.67%) | 20/28 (71.43%) | 45.31% | 0.12499 | 1/48 |
| Yield-aware GRPO | 20/48 (41.67%) | 20/28 (71.43%) | 44.27% | 0.13252 | 0/48 |
| Closest to spec | 11/24 (45.83%) | 11/14 (78.57%) | 49.61% | 0.07886 | 0/24 |
| Physics heuristic | 11/24 (45.83%) | 11/14 (78.57%) | 49.74% | 0.07762 | 0/24 |
| Oracle | 14/24 (58.33%) | 14/14 (100.00%) | 57.55% | 0.00000 | 0/24 |

The simulated binary target was 62.5%, or 20/32 builds, following nominal feasibility. Mean yield uses zero for invalid responses or nominal failures. Shaped regret is the difference from the best available continuous reward on each brief. The exact uniform baseline was 26.56% pass, 30.00% mean yield and 0.28104 regret.

| Lineage | Untouched pass and yield | Random-reward pass and yield | Yield-aware pass and yield |
| --- | ---: | ---: | ---: |
| Gauss | 6/8 · 69.53% | 6/8 · 69.53% | 4/8 · 52.73% |
| Steinheil | 4/8 · 46.09% | 5/8 · 57.03% | 6/8 · 65.62% |
| Wide-gap ED | 0/8 · 8.59% | 0/8 · 6.64% | 0/8 · 7.81% |
| Herzberger–Hoadley | 5/8 · 72.27% | 4/8 · 71.88% | 5/8 · 66.80% |
| Beecher–Miles | 2/8 · 32.03% | 2/8 · 32.03% | 2/8 · 32.03% |
| Laikin telescope triplet | 4/8 · 46.09% | 3/8 · 34.77% | 3/8 · 40.62% |

The original and four-position rotation gave 11/24 and 10/24 passes for untouched Qwen, and 11/24 and 9/24 for each trained arm. Mean yield fell from 48.57% to 42.97% for untouched Qwen, 48.31% to 42.32% for random reward and 47.01% to 41.54% for yield-aware. The same starter survived the rotation on 18/23 valid baseline pairs, 15/23 random-reward pairs and 18/24 yield-aware pairs.

On the five prespecified manufacturing contrast briefs, yield-aware reached 6/10 passes, 58.44% yield and 0.22106 regret. Untouched and random-reward Qwen each reached 8/10, 71.88% and 0.08906, matching both engineering heuristics at 4/5. The two net pass losses came from the same adjusted Gauss brief across both orders. Both comparators selected the 26/32 build starter in both orders. The yield-aware arm selected a 9/32 nominally feasible starter on the original order and a nominal failure on the rotation. Steinheil improved by two passes over untouched Qwen, including one repaired invalid response, and by one over random reward.

| Training check | Yield-aware | Random reward |
| --- | ---: | ---: |
| Attempted / admitted groups | 48 / 12 | 48 / 12 |
| Groups with valid optical reward span ≥0.10 | 27/48 | 27/48 |
| Actual optimizer updates / trained datums | 6 / 83 | 6 / 87 |
| Mean distinct valid starters per attempted group | 2.48 | 2.85 |
| Mean valid-only optical reward variance | 0.03642 | 0.03444 |
| Mean valid-only assigned reward variance | 0.03642 | 0.21094 |
| Invalid training completions | 47/384 | 23/384 |
| First-update maximum token log-probability change | 0.35360 | 0.15634 |
| Initial-to-final maximum token log-probability change | 2.35101 | 0.36748 |
| Sampler reload maximum token log-probability change | 0 | 0 |

The admitted yield-aware groups covered all five task labels, including five groups assigned to the construction-defined manufacturing reversal family. The two step-six sampler checkpoints and optimizer states were saved with a 24-hour TTL, through approximately November 27 at 7:26 p.m. Eastern. Additional buffered application token estimates were $1.494764 and $1.511403 for training and $0.110854 and $0.110248 for full evaluation, totaling $3.227269. Including the two $0.10 storage reserves, the application-accounted total was $3.427269 against the $6.00 ceiling.

The optimization check passed. The yield-aware arm missed the +10-point descriptive target, fell 1.50 and 1.04 yield points below untouched and random-reward Qwen, and increased oracle regret by 11.58% versus untouched. It improved simulated yield over both comparators only in Steinheil and under neither order when averaging the full panel. The six updates changed the policy and some selections, with no improvement in optical starter selection across this development panel. The endpoint data do not justify scaling this recipe. The next smallest diagnostic would score the eight canonical JSON actions under both registered orders on the adjusted Gauss regression and the Steinheil improvement using the saved policies and unadapted Qwen. It would test whether one sampled response per order masked the candidate probability rankings. The diagnostic would require separate approval. [Matched summary](../outputs/v23/matched/summary.json) · [Scored evaluation](../outputs/v23/matched/evaluation.json)

## 2026-11-26 · Checkpoint preservation and fixed-choice preparation

Tinker showed a zero effective checkpoint-storage rate before the retention update. Extended the existing sampler weights and optimizer states for both arms in place by seven days, then verified the new expirations for all four objects. Their earliest expiration is December 3 at approximately 7:52 p.m. Eastern. This storage operation used no inference or training tokens and created no checkpoint copies. No paid probability calls have run. [Preservation record](../outputs/v23/matched/checkpoint-preservation.json) · [Tinker checkpoint storage](https://tinker-docs.thinkingmachines.ai/administration/checkpoint-storage-pricing/)

The probability plan compares untouched Qwen with both step-six checkpoints. It uses the first logged presentation and its four-position rotation for each of the 15 encountered Cooke briefs. It uses the exact frozen two presentations for all 24 development briefs. Eight of the encountered Cooke briefs contributed examples to an update in each arm. No compatible independent Cooke set was available. All of the 17 unused V2.3 Cooke briefs share starter geometries with training, and the eight with unseen physical requirements have zero nominally feasible starters. Older Cooke artifacts inspected in the training split use 32 pupil samples. Those sets do not give a comparable intermediate transfer test. The 17 reused-geometry briefs remain an optional exploratory stratum.

The diagnostic sums every completion token log probability, including JSON punctuation and the renderer stop token. It reports normalized probabilities across the eight compact token paths, their unnormalized total mass, expected shaped-reward regret, mass above the uniform mean reward, oracle mass, expected simulated yield and pass probability, and reward-ranking concordance. Results are grouped by category, lineage and order. On the encountered briefs they are also grouped by whether the yield-aware arm admitted the brief for an update. The adjusted Gauss regression and the two Steinheil gains receive individual readouts. Teacher-forced model log probabilities and the prior unrestricted samples at temperature 1.25 answer different questions.

Only 4/48 yield-aware outputs used the exact compact token path, versus 33/48 for untouched Qwen and 47/48 for random reward. The yield-aware arm used the two-space-indented JSON form in 37/48 responses. Both forms are valid and used the same grading. The format-sensitivity plan scores the same two forms for every policy and choice, reports each form separately, and sums the two disjoint token-path probabilities per starter before normalization. The summed mass covers only those paths. Other valid JSON representations and other tokenizations remain in the residual unrestricted mass. [Offline training and format audit](../outputs/v23/matched/fixed-choice-audit.json)

The saved training journals show a real but narrow selection signal. Both arms used the same eight distinct Cooke briefs for updates and shared 11 of 12 admitted groups. The yield-aware arm never sampled an oracle-reward candidate in 4/12 admitted groups. Of its 83 nonzero-advantage datums, 15 below-target nominal passes received positive advantages because rewards are centered within sampled groups. Mean absolute advantages were 0.18364 in yield-aware and 0.43322 in random-reward training, with RMS values of 0.26764 and 0.46542. Centering without standard-deviation scaling makes equal optimizer step counts compatible with different signal magnitudes. These observations support plausible limits from sampled options, reward contrast, and prescription breadth. They cannot establish a cause or distinguish within-Cooke acquisition from transfer failure. [Offline training audit](../outputs/v23/matched/fixed-choice-audit.json)

| Proposed scoring scope | Requests | Uncached token estimate | Application estimate with 3× margin |
| --- | ---: | ---: | ---: |
| Compact form, 39 briefs | 1,872 | $1.471785840 | $4.415357520 |
| Both forms, 39 briefs | 3,744 | $2.946660480 | $8.839981440 |
| Both forms, 15 encountered Cooke briefs only | 1,440 | $1.189463040 | $3.568389120 |
| Both forms, 24 development briefs only | 2,304 | $1.757197440 | $5.271592320 |

Actual prompt lengths were counted offline using the saved non-thinking Qwen renderer. The quote uses $0.33 per million uncached input tokens and reserves one output token per scoring request at $1.005 per million. Both-form scoring uses 8,917,872 input tokens before prefix cache savings. Ideal same-prompt prefix reuse would reduce the full two-form token estimate to $0.748290240, before the application margin. Cache hits are uncertain and the approval estimate assumes zero hits. The diagnostic was tested end to end against a local fake sampling client. [Compact plan](../outputs/v23/matched/fixed-choice-plan.json) · [Two-form plan](../outputs/v23/matched/fixed-choice-two-form-plan.json) · [Diagnostic runner](v23-fixed-choice.py) · [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)

## 2026-11-26 · Encountered-Cooke probability diagnostic

Completed the approved 15-brief scope against untouched Qwen and both saved step-six policies. All eight choices were scored in both JSON forms and both orders, for 90 complete brief/order/policy rows and 1,440 requests. Application accounting is $3.568389120 at the uncached 3× margin, within the $3.60 approval. The corresponding uncached provider estimate is $1.189463040 for 3,600,048 input tokens and the one-output-token allowance per request. The development scoring stage and any further training were outside this execution. [Execution plan](../outputs/v23/matched/encountered-cooke-fixed-choice/plan.json) · [Raw log probabilities](../outputs/v23/matched/encountered-cooke-fixed-choice/full-logprobs.jsonl)

The eight gradient briefs are the intersection of the two arms' update exposure sets, which were identical. The other seven occurred in attempted training rollouts only. On five of the eight gradient briefs, the first logged order was also used in an admitted group in both arms. The diagnostic scored the first logged order and its four-position rotation. Metrics use the frozen per-starter training-phase grades. Probabilities in this table are conditional on the two scored JSON token paths per starter.

| Subset and policy | Expected optical reward | Expected simulated yield | Expected shaped oracle regret | Mass on oracle choices | Reward-pair concordance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gradient · untouched | 0.46647 | 44.96% | 0.34799 | 24.23% | 53.57% |
| Gradient · random reward | 0.43578 | 41.76% | 0.37867 | 16.84% | 51.16% |
| Gradient · yield-aware | 0.53657 | 52.08% | 0.27789 | 30.23% | 59.88% |
| Attempted only · untouched | 0.62865 | 62.28% | 0.25260 | 54.59% | 49.57% |
| Attempted only · random reward | 0.61066 | 60.49% | 0.27059 | 52.63% | 54.35% |
| Attempted only · yield-aware | 0.65343 | 64.78% | 0.22782 | 53.95% | 55.79% |

| Subset and normalization | Untouched expected reward | Random-reward expected reward | Yield-aware expected reward |
| --- | ---: | ---: | ---: |
| Gradient · compact only | 0.46590 | 0.43578 | 0.52861 |
| Gradient · indented only | 0.46950 | 0.45384 | 0.53729 |
| Attempted only · compact only | 0.62875 | 0.61066 | 0.65563 |
| Attempted only · indented only | 0.62616 | 0.61274 | 0.65300 |

Raw compact token-path mass across all 15 briefs averaged 68.90% for untouched, 99.57% for random reward and 10.61% for yield-aware. Raw indented mass averaged 11.82%, below 0.00001%, and 74.54%, respectively. Normalized by the two forms, the indented share averaged 14.73%, effectively zero, and 87.48%. Neither raw nor renormalized probabilities enumerate all valid alternative spellings, tokenizations or temperatures. The rare random-control indented branch has very small absolute mass even when its conditional candidate rankings are defined.

On the eight gradient briefs, the combined reward improvement over untouched was 0.07010. A symmetric decomposition assigns 0.06565 to within-form starter preference and 0.00445 to shifts in the format mixture. Combined reward was higher than untouched in both orders, by 0.1211 in the first and 0.0190 in the rotation. Five of eight gradient briefs improved when averaging both orders. On attempted-only briefs reward rose by 0.02478 overall versus untouched, but fell by 0.0367 in the first order and rose by 0.0863 in the rotation. Their oracle-choice mass fell by 0.64 percentage points versus untouched. Expected reward for all 15 briefs was 0.54215 untouched, 0.51739 random reward and 0.59110 yield-aware.

Yield-aware absolute advantage averaged 0.18364 per nonzero datum versus 0.43322 for random reward, with RMS 0.26764 versus 0.46542. Summed across the completion tokens to which advantages were applied, the absolute coefficients were 146.97 versus 274.86. Yield-aware was lower in five of six steps. The objective applied smaller raw optical weights, making weaker optical optimization pressure plausible. The training ledgers contain neither gradient nor parameter-update norms. Adam adapts step sizes, and the observed probe moved more under yield-aware than under random reward. The scalar weights alone cannot establish a smaller effective parameter update or explain the development result.

The diagnostic demonstrates descriptive capability acquisition on the updated Cooke briefs, with mixed evidence of transfer to Cooke briefs seen only in attempts. It leaves the previous six-lineage decline unresolved. The next diagnostic, if approved, should score both formats on the frozen 24 development briefs to test whether their candidate distributions support a transfer failure or whether formatting and one sample per order masked selection changes. No training expansion is supported yet. [Machine-readable readout](../outputs/v23/matched/encountered-cooke-fixed-choice/readout.json)

## 2026-11-26 · Six-lineage development fixed-choice diagnostic

The unchanged runner scored all eight choices in both JSON forms and both original orders for all three policies across 24 frozen development briefs. The completed matrix has 144 brief/order/policy rows and 2,304 requests. All task hashes and prompt-token hashes matched the earlier sampled evaluation. Application accounting was $5.27159232 at the uncached 3× margin, within the $5.30 approval. [Verified readout](../outputs/v23/matched/development-fixed-choice/verified-readout.txt) · [Raw log probabilities](../outputs/v23/matched/development-fixed-choice/full-logprobs.jsonl)

| Policy | Expected reward | Expected simulated yield | Oracle regret | Oracle-choice mass | Pair ranking agreement | Expected pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untouched | 0.46123 | 44.11% | 0.13551 | 36.74% | 57.65% | 42.69% |
| Random reward | 0.47041 | 45.01% | 0.12634 | 36.44% | 56.92% | 43.78% |
| Yield-aware | 0.48640 | 46.57% | 0.11034 | 40.01% | 58.89% | 44.73% |

The combined results normalize over both token paths per starter. Untouched, random-reward and yield-aware expected rewards were 0.46129, 0.47041 and 0.49053 under compact-only normalization, and 0.45893, 0.46467 and 0.48522 under indented-only normalization. Raw compact mass averaged 75.21%, 99.51% and 16.60%, while indented mass averaged 9.24%, effectively zero and 66.39%. Within-form preference contributed +0.02776 to the yield-aware reward gain versus untouched. The format mixture contributed −0.00259.

| Lineage | Fixed-choice mean yield, untouched / random / yield-aware | Earlier sampled mean yield, same order |
| --- | ---: | ---: |
| Gauss | 67.19% / 67.91% / 68.85% | 69.53% / 69.53% / 52.73% |
| Steinheil | 62.25% / 62.58% / 62.65% | 46.09% / 57.03% / 65.62% |
| Wide-gap ED | 6.28% / 6.29% / 7.15% | 8.59% / 6.64% / 7.81% |
| Herzberger–Hoadley | 65.63% / 67.78% / 69.90% | 72.27% / 71.88% / 66.80% |
| Beecher–Miles | 27.15% / 27.69% / 29.63% | 32.03% / 32.03% / 32.03% |
| Laikin triplet | 36.14% / 37.77% / 41.22% | 46.09% / 34.77% / 40.62% |

Yield-aware expected yield exceeded both controls in both orders and on all six lineages. On the prespecified five manufacturing contrast briefs, fixed-choice yield was 63.68%, 65.91%, and 68.44%, respectively. The previously sampled comparison was 71.88%, 71.88%, and 58.44%. On the adjusted Gauss brief, the yield-aware arm put about 94.7% conditional mass on the 26/32 starter in both orders, despite sampling weaker starters in both orders. The Steinheil sampled gain was much larger than the fixed-choice gain.

The probability diagnostic supports modest transfer of candidate preference. It does not overturn the observed end-to-end sampled regression. The earlier experiment used one unrestricted sample at temperature 1.25 per order, and this diagnostic scores two exact JSON token paths per starter using model log probabilities. The comparison cannot apportion the discrepancy between sampling variation, decoding temperature, and probability outside the two scored forms. [Complete metrics by form, lineage and order](../outputs/v23/matched/development-fixed-choice/results.json)
