# Learning When to Stop: A DRL Service for Adaptive Shot Allocation

This repository accompanies the paper *"Learning When to Stop: A Deep
Reinforcement Learning Service for Adaptive Shot Allocation in Quantum Circuit
Execution"* (submitted to **ICSOC 2026**, double-blind).

Quantum circuits are executed by repeated sampling ("shots"): too few shots
give an unreliable estimate of the output distribution, too many waste costly
hardware time. The software in this repository is a **shot-allocation
service** — an intermediary between a user and a quantum-cloud backend — whose
decision logic is a Deep Q-Network that observes only the running statistics of
the measured outcomes (never the circuit or the noise model) and decides, batch
by batch, whether to **continue** or **stop**. The policy is trained against an
*a-posteriori* optimal stop derived from the **point of diminishing returns**
and evaluated against five reference methods on a strict, leakage-free split of
the [QSimBench](https://github.com/GBisi/qsimbench) benchmark.

## The agent in one paragraph

An 11-feature state (4 static instance descriptors + 7 dynamic convergence
signals, including the lagged-TVD diminishing-returns proxy, a sustained-
stability streak, and a finite-budget-corrected statistical extrapolation of
the stopping point) feeds an MLP (512-256-128) trained as a **Double DQN with
Huber loss**. The terminal reward is **asymmetric**: any undershoot is a large
negative penalty, any overshoot a positive reward decaying gently with the
surplus, so the policy hugs the optimum from above. Episodes run in batches of
50 shots up to a budget of 20,000; Oracle labels use the suffix-stable
a-posteriori optimum (TVD, δ=0.1). The production agent trains for 2,000
episodes (seed 42) with de-noised validation-MAE checkpointing, and evaluates
greedily with a 2-vote stability filter on 36 held-out benchmark traces.

## Setup

Python 3.12, PyTorch, OpenAI Gym, NumPy, QSimBench:

```bash
conda create -n qdrl python=3.12
conda activate qdrl
pip install torch gym numpy matplotlib tqdm qsimbench
# for SVG→PDF figure conversion only:
conda install cairo && pip install cairosvg
```

## Running training + evaluation

```bash
conda run --no-capture-output -n qdrl python program_enhanced_2.py
# or open program-main-enhanced-2.ipynb and run its single cell
```

The run (i) builds the Oracle-label cache for all training/validation instances
(recomputed on first run, then cached as `actual_oracle_eps0.1_batch50.{json,csv}`),
(ii) trains the agent with early stopping and best-validation checkpointing, and
(iii) evaluates on the 36 held-out traces, writing the CSV, the agent-vs-Oracle
scatter, and a training dashboard into `generic-enhanced-2-extrap/`. The
configuration block sits at the top of `main()` in `program_enhanced_2.py`.

## Reproducing the paper's tables and figures

```bash
# tables: reads generic-enhanced-2-extrap/multirun_eval-generic.csv + pdfs/tables.tex
cd paper && python3 make_tables.py

# scatter figure: crops the scatter panel out of multirun_eval-generic.svg
# and converts it to paper/figures/multirun_eval-generic.pdf
conda run -n qdrl python paper/make_figure.py

# the paper itself (llncs.cls / splncs04.bst are shipped in paper/)
cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

## Results at a glance

On the 36 held-out traces (δ=0.10, budget 20,000), the learned policy reaches a
mean absolute deviation from the Oracle of **924 shots (4.6% of the budget)** —
447 on the small (n<10) half, 1,400 on the big (n≥10) half. It is **orders of
magnitude** more efficient than the a-priori Weissman/Hoeffding bounds, beats
the Inc-Hell (2,706) and Inc-JS (3,336) hand-tuned online policies by 2.9× and
3.6×, and comes within 6% of the strongest, Inc-TVD (871) — without any
hand-crafted stopping rule. The agent stops below the optimum on 10 of 36
traces, mostly near-budget circuits; re-scored on the Oracle's reference
realisation, those estimates stay within a TVD of 0.10–0.16 of the full-budget
reference (median 0.12) against the 0.10 target.

## License

Released as open source to accompany the paper. (License to be added with the
de-anonymised, camera-ready release.)
