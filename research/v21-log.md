# Lux Lens V2.1 research log

## 2026-11-25 · Offline implementation and validation

Lux tests whether a non-thinking Qwen3.5-4B can learn to select the starter that will meet a camera brief after bounded numerical refinement and simulated manufacturing. The model returns a single local candidate ID in strict JSON. Optiland traces rays and measures performance, and the environment handles all optimization and manufacturing. The primary outcome is a nominally passing selection whose observed simulated yield meets the brief's target.

### Verified V2 evidence

The frozen representative pilot contained 42 briefs. Sixteen additional train-only briefs brought the total to 58, with 464 graded actions and 3,840 recorded candidate-build evaluations. Four of the six training briefs originally tagged for manufacturing robustness reproduced a yield gap of at least 25 percentage points on disjoint training draws. On 14 representative pilot test briefs, exhaustive evaluation found a passing choice on six, each engineering heuristic passed four and exact uniform chance was 19.6%. The original test and development panels each have one reference ancestry. Treat both as previously inspected pilots. The V2 archive has a complete ray replay and figure. Its report records zero paid V2 calls.

Reviewed the supplied private reference implementation and its run and artifact documentation. The V2.1 implementation uses task certification, simultaneous practice of constituent skills and the full task, all-attempt diagnostics, a seeded reward control and a separately frozen final evaluation. The public-facing UI contains Lux replay data and measurements. The linked reference artifact page could not be loaded through the research browser.

### Curriculum and capacity

The default mixture samples A nominal feasibility, B optimization potential, C manufacturing robustness, D engineering trade-offs and E complete selection from the first step with weights 15/15/20/10/40. A construction chance proxy sets the initial mix of difficulty tiers. Bounded adjustments use rolling optical pass rate and informative-group frequency, including rejected groups. Low complete-task success shifts some exposure toward constituent skills and easier tasks. A rolling pass rate above 80% reduces a family's weight while keeping it in rotation. The group journal retains selection entropy, valid action and position counts, pass rate, optical and training reward statistics, failure counts and actual optimizer admission.

The seed controls candidate IDs and presentation order independently. The measured expanded train split has per-position pass prevalence of 44–55% and per-ID prevalence of 48–54%. A rotation of four positions in exploration tests position bias within the same task, without changing the candidate IDs or ground truth. The format example uses integer 0, which is outside the eight available IDs. The run records allow a direct comparison of representation, rotation, sampling strategy and temperature. Offline tokenization of all 114 training and development prompts with the actual non-thinking renderer found a maximum of 3,926 tokens for the physics representation.

| Frozen pool | Briefs | Distinct starter prescriptions | Independent reference ancestries |
| --- | ---: | ---: | ---: |
| V2 complete pilot | 58 | 31 | 3 |
| V2.1 expanded training | 100, of which 85 are curriculum eligible | 22 | 1 |
| V2.1 development | 14 | 8 | 1 |
| V2.1 previously inspected test pilot | 14 | 8 | 1 |

Sixteen newly simulated briefs came from a second train-only contrast pool. Fifty-four further eight-way situations recombine validated starters across compatible pools. The 100 training situations cover 30 distinct requirement and process contexts. There are no presentation-only duplicates in this count. Families overlap, with 68 A, 75 B, 32 C, 40 D and 85 E tags on the full training split. On all 100 training briefs exact random selection has 51% expected pass. In an 800-group cold-start scheduling check, the selected tasks had a mean exact construction chance of 28.3%. Base-model performance and the useful mix remain empirical questions. [Recorded evidence](v21-evidence.json)

The fixed-pool task builder reuses exact nominal fits only for the same physical brief, process profile, simulator and optimization settings. It generates shared manufacturing draws once per compatible candidate, scenario, profile and draw purpose, then reuses those results across candidate sets. A new objective, budget or pupil grid requires a new nominal fit. A changed process requires new manufacturing simulations. This supports enumerating or deterministically sampling thousands of candidate sets as new validated prescriptions are introduced. Combinations and counterfactuals share ancestry and must contribute accordingly to statistical inference. The custom lineage importer accepts documented air-spaced spherical references, independently checks centered rays and first-order focal length, and crosses application and process recipes across the supplied lineages. It does not establish ancestry by provenance label alone. No new independent optical ancestry was added to the verified pool in this iteration.

### Optics and manufacturing evidence

Twenty of the 32 expanded training robustness tasks reproduced a construction yield gap of at least 25 points on separate training draws. These tasks share the Cooke ancestry and some draw streams. Thirty-nine training situations have at least 0.10 oracle reward headroom over closest to spec. These measurements show decision consequences in the simulator. They do not quantify the part a small model can infer from the prompt.

Added 32 fresh paired builds each for five feasible development designs and nine feasible training designs, plus 100 each for ten feasible designs on two previously inspected pilot briefs. On two preselected Cooke manufacturing contrasts, the less nominally sharp candidates passed 32/32 versus 12/32, and 29/32 versus 5/32. Both members of each pair met nominal requirements. The second comparison has 24 paired changes in the same direction and a 75-point simulated yield gap. In that same brief a third candidate's 25/32 estimate had a Wilson interval of 61.2–89.0%, which crosses the 62.5% brief target. The two 100-build pilot cohorts had 76–94% observed yield across their ten nominal successes. These are sample results for simulated processes.

Use 8 builds for screening, 32 for investigating ambiguous or useful contrasts, and at least 100 fresh builds per nominally successful action in final evaluation. Separate streams are keyed by draw purpose, group and fidelity. Equal trial indices share physical error quantiles across prescriptions under the same comparison. The holdout report uses observed yield for its locked binary metric and reports Wilson interval status and sensitivity to unresolved designs alongside it. Use a larger fixed build count before sealing if threshold ambiguity in development could change conclusions.

At 64 and 128 pupil samples, all eight nominal labels held in each of the two cross-lineage checks. The most extreme change in target MTF was 0.426 on a failing development design. The two passing development candidates each differed by less than 0.007 at 128. Pairwise nominal objective order agreement with the 32-sample run was 21/28 on the development triplet and 28/28 on the pilot doublet. At 7 versus 25 optimization trials, the doublet changed four nominal decisions. At 49 versus 25 it changed one. Lock the budget, check near-boundary geometry at higher pupil grids, and keep variables in the physics and grading record.

An external optics review is needed for glass and clear aperture source data, tolerance distributions and correlations, whether rim clearances are conservative enough, real focus fixture noise and quantization, and the correspondence between monochromatic FFT MTF and polychromatic measured performance. Coatings, ghosting, sensor response, thermal effects, and actual production yield are outside the current simulator. A look at published recipes surfaced related designs and unsupported topologies. Source genuinely distinct, compatible prescriptions before attributing changes to broad optical generalization.

### Local research artifact

```bash
rtk proxy .venv/bin/python -m lux_frontier.v2.serve
```

Open `http://127.0.0.1:8765/`. The first task shows the full eight-candidate V2 development replay and every available eight-build outcome. The second shows the Cooke robustness contrast with 32 fresh builds per nominally feasible candidate. Compare candidates 6 and 4 there. Use the field selector and **Zoom optics** for the lens elements, **Play** or the slider for evaluated incumbent changes, and the factory grid to inspect geometry, traced rays, sensor spots, tolerancing errors and individual failures. Rejected optimizer trials are listed separately. Replay uses discrete evaluated states and the simulator's measured coordinates.

The browser has no live training dependency. Replay packs are versioned, content addressed and compressed, with state and prompt deduplication. The local server reads the small catalog first and the browser fetches the selected replay. The model comparison and training views accept recorded results. With the bundled data they show engineering controls and pending Qwen results. Factory draws whose geometry is physically invalid show the recorded errors and reason without an optical trace.

The offline test suite passed 144 tests. Local Safari verification advanced the first replay to its second measured optimization state, inspected the failed seventh build, and confirmed the optional 100-build case rendered its complete grid. Offline validation reran the three analytical reference checks, tokenized the new prompts with the real Qwen chat renderer and exercised all 72 exploration groups with an injected fixture sampler.

For an optional 100-build replay, export the inspected pilot task and pass it to the local server.

```bash
rtk proxy .venv/bin/python -m lux_frontier.v2 browser-replay outputs/v21-pilot-100/manifest.json \
  --split test --task test-cohort000-spec001-economy-adjusted \
  --output outputs/v21-evidence/pilot-100-replay.json.gz
rtk proxy .venv/bin/python -m lux_frontier.v2.serve --replay outputs/v21-evidence/pilot-100-replay.json.gz
```

Attach real run outputs using `--run LABEL=FILE` and an evaluation with `--evaluation FILE`. Use `browser-replay --predictions LABEL=FILE` to export a task with predictions that match the frozen prompt and task hashes.

### Reproduction

The preserved pilot and generated cache directories live under `outputs/`. In this checkout the 128-brief expanded manifest is `outputs/v21-expanded/manifest.json`. The two data-generating commands below recreate the recipe in a fresh output directory. Follow them with `readiness` for counts, classifications, per-task random performance and position diagnostics.

```bash
rtk proxy .venv/bin/python -m lux_frontier.v2 generate outputs/v21-pool \
  --contrast-parent outputs/v2-pilot/manifest.json --pools 2 --scenarios 8 \
  --workers 5 --profiles standard-fixed standard-adjusted
rtk proxy .venv/bin/python -m lux_frontier.v2 expand outputs/v21-pool/manifest.json \
  --output outputs/v21-expanded --per-context 6 --seed 17292
rtk proxy .venv/bin/python -m lux_frontier.v2 readiness outputs/v21-expanded/manifest.json \
  --output outputs/v21-evidence/readiness.json
```

For an independently sourced reference panel, pass `generate --lineages FILE` with a JSON array of objects containing `split`, `source` and a complete V2 `Lens` in `prescription`. Use distinct declared ancestries and document any adapted aperture or stop. Reference validation should be supplemented by external provenance review before any reference is assigned to the final panel. Use `manufacture --phase final --fidelity definitive --seed SEED` on the complete chosen test split. A run on the current pilot would still be inspected evidence.

### Spending and next live experiments

V2.1 spending was **$0**. All commands below print application estimates with a 3× token-cost margin and checkpoint storage reserve by default. Executing each needs explicit approval and a unique run output. The study budget is $500 in total. The V1 optimizer attempt records a $0.39723 conservative estimate, and its estimated charge and any earlier fellowship charges should be reconciled against the provider ledger before authorizing a larger run. Each new run captures reserved and sampled tokens, attempts, admission, updates and estimated spend. Provider bills and storage should be added to the ledger after each approved run. [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)

**Exploration.** Use two feasible development briefs with 32-build evidence and one predeclared no-solution brief as a negative control. The matrix is 3 briefs × 3 representations × 2 temperatures × 2 sampling strategies × 2 positions × 8 samples for 576 completions, with a conservative estimate of **$2.391276** and a local estimate limit of **$2.40**. Keep Qwen3.5-4B with thinking disabled, max 32 generated tokens, max 4,096 prompt tokens and top-p 1.0. The three unweighted exact-chance pass baselines are 37.5%, 25.0% and 0%. Closest to spec passes one and the physics heuristic two. Report feasible and no-solution cases separately. Look for useful optical reward spread in repeated, valid selections across multiple tasks and for relative stability under order rotation. If the valid model choices are concentrated or near uniform chance, examine omitted or misleading optical information and reward fidelity before funding training.

```bash
rtk proxy env HF_HUB_OFFLINE=1 HF_HUB_CACHE=.cache/huggingface/hub \
  .venv/bin/python -m lux_frontier.v2 explore outputs/v21-dev-32/manifest.json \
  --task development-cohort000-spec000-standard-adjusted \
  --task development-cohort000-spec003-standard-adjusted \
  --task development-cohort000-spec001-standard-adjusted \
  --temperatures 0.6 1.25 --settings research/v21-explore.json \
  --output outputs/v21-runs/base-exploration.json
```

**One-step optimizer smoke.** Eight attempts of eight with at least one informative group, rank 16, learning rate 2e-5 and two target groups. Four development tasks each receive two samples before and after the update. Maximum 80 sampled completions and an application estimate of **$0.594465** within a **$0.60** configured limit. Use the representation, temperature and sampling strategy supported by exploration. The checked-in smoke settings are provisional. Pass requires a real optimizer update, a fixed completion's token log probability changing by more than 1e-6 and a reloaded sampler agreeing with the saved weights within 0.02 maximum absolute token log probability. This tests the plumbing and does not test improvement.

```bash
rtk proxy env HF_HUB_OFFLINE=1 HF_HUB_CACHE=.cache/huggingface/hub \
  .venv/bin/python -m lux_frontier.v2 train outputs/v21-expanded/manifest.json \
  --settings research/v21-smoke.json --output outputs/v21-runs/optimizer-smoke.json
```

The two estimates sum to **$2.985741** against a combined **$3.00** application ceiling. After approval, append `--execute` to the applicable command. The offline estimates above made no provider calls.

**Matched training.** Freeze the settings after development exploration. For the scientific comparison use the same fixed sampling policy across trained arms, derived from development and the mixed curriculum. This controls task exposure. Train each of `--reward random`, `--reward nominal` and `--reward yield` for 20 steps, rank 16, learning rate 2e-5, sixteen attempted groups of eight per step, up to four accepted groups per step, with a two-group update minimum. Each arm caps at 2,640 sampled completions including development evaluation. The random reward uses independent seeded Bernoulli draws per attempt slot. Its probability defaults to 0.5 and can be locked from development estimates before training. Invalid completions receive the corresponding assignment in the log and are excluded from optimization in all arms. Acceptance and realized updates can differ and are reported.

```bash
rtk proxy env HF_HUB_OFFLINE=1 HF_HUB_CACHE=.cache/huggingface/hub \
  .venv/bin/python -m lux_frontier.v2 train outputs/v21-expanded/manifest.json \
  --settings research/v21-matched.json --reward yield --output outputs/v21-runs/yield-seed17.json
```

Substitute `random` and `nominal` and their unique output names. The offline estimate is **$17.216230 per trained arm**, **$51.65** for all three. Untouched Qwen only incurs evaluation. With two inference completions per brief, the four-arm evaluation costs about **$0.0332 per brief** under the same conservative pricing basis. Including exploration and smoke, a 14-brief pilot comparison would total about **$55.10**. A single-seed comparison on up to 500 final briefs would total up to **$71.24**. At 500 briefs, two training seeds and shared untouched inference would total about **$135.34**. These are planning examples. The panel size and scientific budget depend on development variance, independent lineages and approval.

### Final evaluation and remaining decisions

Use multiple independent development lineages and paired differences to estimate the required panel size with `power`. The current pilot cannot estimate a between-lineage variance. Eight lineages is merely the smallest possible two-sided exact-test resolution at the five-contrast Bonferroni planning level. Depending on the observed paired variance and a prespecified meaningful effect, substantially more may be needed. Approve the meaningful-effect target before the power calculation. The pilot calculation used 15 percentage points as an example.

Before serious training, use `seal` with the definitive test manifest, frozen inference settings, the development power record and every previously inspected manifest. The seal locks task hashes, conditions, inference settings and simulated build streams. It checks that ancestry, geometry, counterfactual cohorts and prompts do not cross splits. Use `sample --split test --seal FILE` with the untouched model and three selected sampler checkpoints. Add `--checkpoint PATH --training-run FILE` for each trained arm. `sealed-evaluate` requires all four named records, validates matched provenance, and reports paired cluster-bootstrap intervals, two-sided cluster sign-flip tests, Holm-adjusted primary p-values, results by family and lineage and simulated-threshold sensitivity. Its primary measure weights briefs equally. Confidence intervals and tests group related cases by independent lineage. Development drives checkpoint selection.

The shortest defensible route is the approved exploration followed by the optimizer smoke if the groups have enough valid optical variation. Next establish that yield-aware performance can improve relative to controls in development, measure between-lineage variability, and complete the optics review and independent final panel. Freeze that panel before the matched training that supports any headline result. Model improvement, independence of new reference families, and the final experiment price remain open. The user needs to approve the live scout and smoke, then the effect target, final protocol and matched training budget.
