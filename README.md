# Epistemic-Foraging-Benchmark

A Measurement of Epistemic Foraging and Executive Function in Frontier Models

> **Competition Track:** Executive Function — Planning, Cognitive Flexibility, Hypothesis Testing & Working Memory

---

## Table of Contents

1. [Overview](#overview)
2. [The Cognitive Gap: Three Critical Blind Spots](#the-cognitive-gap-three-critical-blind-spots)
3. [The Task: Black Box Diagnostic](#the-task-black-box-diagnostic)
4. [Evaluation Metric: Turn-Cost as Entropy Reduction](#evaluation-metric-turn-cost-as-entropy-reduction)
5. [Key Design Principles & Competition Alignment](#key-design-principles--competition-alignment)
6. [SDK Implementation Details](#sdk-implementation-details)
7. [Cognitive Alignment with DeepMind's AGI Framework](#cognitive-alignment-with-deepminds-agi-framework)
8. [Quick Start](#quick-start)
9. [License](#license)

---

## Overview

Current AI evaluations overwhelmingly measure the **product** of reasoning — crystallized knowledge — rather than the **process** of reasoning — fluid intelligence. If a model correctly answers a complex riddle, it is nearly impossible to tell whether it genuinely deduced the answer or simply retrieved a memorized solution from its vast training corpus. Most current benchmarks fail to answer the most important question in AGI evaluation: *is the model actually thinking, or is it just remembering?*

**The Interrogator's Dilemma** — executed via the **Black Box Diagnostic** task — shifts the paradigm entirely.

Rather than asking a model to *answer* a static question, this benchmark places the model in a dynamic, multi-turn environment where it must **actively search** for an answer. By measuring how efficiently a model navigates a hidden, procedurally generated topology, we isolate its capacity for **epistemic foraging**: the ability to plan ahead, formulate and test hypotheses, and adapt its strategy based on incoming information — all while minimizing Shannon entropy across an unknown state space.

We measure not just *whether* the model can solve a problem, but *how efficiently* it actively reduces uncertainty to arrive at the solution. This distinction is the entire benchmark.

---

## The Cognitive Gap: Three Critical Blind Spots

Current LLM evaluations suffer from three critical blind spots that this benchmark is specifically designed to address:

### 1. Data Contamination

Static text-based logic puzzles, Q&A pairs, and reasoning challenges are heavily represented in pre-training data. When a model "solves" such a problem, there is no way to determine whether it applied genuine deductive reasoning or pattern-matched against a memorized solution. Even novel-seeming prompts can map to structural analogues absorbed during training.

### 2. Passive Processing

Models are almost universally evaluated as **Answerers** — agents given perfect, complete context and asked to respond. This fundamentally misrepresents the cognitive demands of real-world intelligence. Even complex reasoning benchmarks often fail to capture genuine fluid intelligence because they present all necessary context upfront. A truly capable AI must also function as an **Interrogator** — an agent that navigates imperfect, incomplete information by deciding *what to ask*, *in what order*, and *why*.

### 3. Conflating Syntax with Cognition

Traditional multi-turn agent benchmarks frequently penalize models for surface-level formatting failures — a missing JSON bracket, an incorrectly escaped string — rather than evaluating the underlying cognitive process. A model that reasons brilliantly but formats imperfectly scores identically to one that reasons poorly. This conflation makes it impossible to measure what actually matters: the quality of the model's planning and hypothesis-testing.

---

## The Task: Black Box Diagnostic

To solve these blind spots, we introduce the **Black Box Diagnostic**.

The model is placed into a zero-context scenario where it acts as an **expert network engineer** tasked with debugging a live server cluster. Exactly one node has experienced a fatal hardware failure — an "offline endpoint" — and the model must locate it using the fewest possible queries.

There is no context window full of clues. There is no memorizable answer. There is only the environment, the actions available, and the model's own capacity to reason.

### 2.1 Environment Topology

The environment is a **20-node directed acyclic graph (DAG)**, instantiated via a lightweight, deterministic Python script at runtime upon each evaluation:

```
Layer 1 — Core Hubs       : C1, C2          (2 nodes)
Layer 2 — Subnet Switches : S1, S2, S3, S4  (4 nodes)
Layer 3 — Endpoints       : E1 – E14        (14 nodes)
```

The Core Hubs (C1, C2) act as central routing nodes. Each Subnet Switch hangs off a Core Hub. Each Endpoint connects to a Subnet Switch as a leaf node. The exact connections between layers, and the identity of the single anomalous node, are **randomized on every initialization** — making training data contamination physically and structurally impossible.

### 2.2 The Action Space

Models interact with the environment over a maximum of **10 turns**. Each turn, the model may take exactly one of three actions, expressed as a strictly formatted JSON payload.

To preserve cognitive signal and eliminate syntactic noise, chain-of-thought reasoning is **explicitly decoupled from action execution** via regex parsing. The model is free to reason at length in natural language before outputting its action — it is evaluated exclusively on its spatial reasoning, planning efficiency, and hypothesis management, not on perfect JSON formatting.

| # | Action | JSON Payload | Environment Returns |
|---|--------|-------------|-------------------|
| 1 | **Trace a node** | `{"query_type": "check_status", "target": "Node_Name"}` | Whether the anomaly is located at or downstream from the specified node |
| 2 | **Test a connection** | `{"query_type": "check_connection", "target_1": "Node_A", "target_2": "Node_B"}` | Whether the anomaly lies along the path between the two specified nodes |
| 3 | **Declare solution** | `{"solution": "Node_Name"}` | Pass (correct) or Fail (incorrect) — terminates the episode |

> **⚠️ Action Cost Weighting:** Actions are not equal. To further prevent brute-force execution, actions are weighted by information cost. Pinging a direct endpoint node costs **3 turns**, while testing a connection between hub-level nodes costs **1 turn**. Under a 10-turn limit, a model blindly guessing endpoints will fail after just 3 guesses ($3 \times 3 = 9$ turns), mathematically enforcing deductive, top-down reasoning over blind, sequential probing.

---

## Evaluation Metric: Turn-Cost as Entropy Reduction

We abandon standard binary pass/fail accuracy metrics entirely. The primary metric is **Information Efficiency**, grounded in the principles of **Optimal Experimental Design** and **Bayesian Active Learning**.

### The Scoring Formula

```
Score = Max_Turns - Turns_Taken
```

This provides a **continuous gradient of cognitive efficiency** rather than a flat accuracy rate — enabling nuanced comparisons between models, across difficulty seeds, and against human baselines. A model that solves the problem in 3 turns scores dramatically higher than one that solves it in 9.

### Reasoning Profiles

| Reasoning Profile | Model Behavior | Epistemic Efficiency Score |
|---|---|---|
| 🔴 **Retrieval / Guessing** | Randomly pings endpoints one by one (E1, then E2, then E3...), failing the 10-turn limit after just 3 queries due to action costs | Low — may score zero or negative |
| 🟢 **Systematic Deduction** | Queries Core Hubs first to eliminate entire downstream branches, then narrows through Subnets to Endpoints — effectively executing a binary search | High — typically solves in 3–4 turns |
| ⚫ **Hallucination / Collapse** | Forgets previous query results, repeats already-answered queries, or produces structurally invalid actions | Zero — automatic failure |

### What This Reveals

A model with strong Executive Function will immediately recognize this as a **binary search problem** over a hierarchical graph. By querying C1 first, it can eliminate 10 nodes in a single turn. By then querying the relevant Subnet Switch, it eliminates 3–4 more. The anomaly can be isolated in as few as **3 to 4 turns**.

A model relying on lazy, brute-force execution — the cognitive equivalent of "poor executive function" — will test endpoints sequentially (E1, E2, E3...), quickly exhausting the 10-turn limit after only 3 queries due to the higher endpoint action costs. This failure mode is not just a wrong answer; it is a measurable and interpretable cognitive profile.

---

## Key Design Principles & Competition Alignment

### A. Zero-Cost Procedural Generation (Anti-Contamination)

The network topology and the identity of the anomalous node are randomized at runtime via the **`kaggle-benchmarks` Python SDK**. This benchmark requires **$0 in external API calls** and uses **no proprietary datasets** — adhering strictly to the competition's reasonableness standard. Because the environment is generated fresh per evaluation, no static artifact exists in any training corpus for any model to have memorized.

This makes the benchmark fundamentally and structurally **immune to training data contamination**, not merely unlikely to be contaminated.

### B. Scalable Human Baselining

Section 3.2 of the Kaggle/DeepMind brief explicitly emphasizes the importance of evaluating AI systems alongside human participants. Traditional prompt-heavy benchmarks make crowdsourcing human baselines prohibitively expensive, logistically complex, and culturally biased — the tasks simply do not translate cleanly to a non-technical audience.

The Black Box Diagnostic translates **seamlessly and naturally** into a simple, language-agnostic visual web interface: a rendered tree of clickable nodes representing the server cluster. A human participant needs no technical background — they click nodes to find the broken server. This allows researchers to theoretically crowdsource **thousands of human baseline interactions at minimal cost**, tracking human **click-efficiency** directly against LLM **turn-efficiency** using an identical underlying metric. The comparison is clean, apples-to-apples, and requires no prompt translation or cultural adaptation.

### C. Isolating Executive Function from Helpfulness Training

Standard RLHF fine-tuning optimizes heavily for "helpfulness" — producing confident, fluent, plausible-sounding responses. This creates a class of failure mode that is invisible on most benchmarks: the model that generates a polished, authoritative-sounding answer that is simply hallucinated.

By utilizing an **active-learning paradigm**, the Black Box Diagnostic structurally eliminates this failure mode. The model cannot hallucinate a polite, confident answer and receive partial credit. It is **mathematically forced** to prove its causal reasoning by physically reducing the search space — layer by layer, turn by turn — against a stateful environment that does not respond to confident language, only to correct queries. The environment is the judge, not a human rater.

---

## SDK Implementation Details

This benchmark is built natively on top of the **`kaggle-benchmarks` multi-turn conversation loop** and requires no external infrastructure.

| Component | Implementation Detail |
|---|---|
| **Graph Generation** | 20-node DAG instantiated via a lightweight, deterministic Python script at runtime |
| **State Tracking** | A localized Python dictionary maintains the active DAG configuration and tracks which nodes have been queried across the episode |
| **Turn Limit** | 10 turns maximum per episode |
| **Action Parser** | `re.search(r'\{.*?\}', response_text, re.DOTALL)` — extracts the model's action payload from free-form text, ensuring evaluation targets spatial reasoning and planning rather than JSON perfectionism |
| **Scoring** | `Max_Turns - Turns_Taken`, computed at episode termination upon a valid `solution` payload |
| **Action Weighting** | Direct endpoint pings carry higher turn costs than hub-level connection tests, enforced at the environment layer |
| **SDK Integration** | Native `kaggle-benchmarks` multi-turn loop; no additional wrappers required |
| **External Dependencies** | None — $0 in API calls, no proprietary data |
| **License** | CC0 1.0 Universal — all procedural generation code and evaluation logic is released to the public domain |

---

## Cognitive Alignment with DeepMind's AGI Framework

The Black Box Diagnostic directly operationalizes the **Executive Function** cognitive cluster as defined in DeepMind's *Measuring Progress Toward AGI* framework, targeting four sub-capabilities simultaneously:

| Cognitive Capability | How This Benchmark Measures It |
|---|---|
| **Planning** | The model must form and execute a multi-step query strategy using the DAG topology — it cannot solve the problem reactively turn by turn |
| **Working Memory** | The model must retain and integrate the results of all prior queries to correctly update its internal model of the search space — repeating a query is a measurable cognitive failure |
| **Cognitive Flexibility** | The model must update its hypothesis and redirect its search path dynamically as new evidence arrives — a rigid pre-planned strategy will fail on randomized topologies |
| **Hypothesis Testing (Bayesian Active Learning)** | Each query must be selected to maximally reduce Shannon entropy across the remaining candidate nodes — the benchmark is a direct empirical test of whether the model can identify and execute the highest-information action at each step |

No existing static benchmark captures all four of these dimensions simultaneously. The Black Box Diagnostic does so within a single, lightweight, reproducible task.

---

## Quick Start

```bash
pip install kaggle-benchmarks
```

```python
from epistemic_foraging import BlackBoxDiagnostic

# Initialize a fresh environment (randomized topology + anomalous node)
env = BlackBoxDiagnostic(seed=42)

# Note: The .run() method below is pseudo-code representing the Kaggle SDK evaluator loop.
# It safely handles the multi-turn interaction environment behind the scenes.
result = env.run(model="your-model-endpoint")

# View the full episode transcript and efficiency score
print(result.transcript)
print(f"Score: {result.score} / {result.max_score}")
print(f"Turns taken: {result.turns_taken} / {env.max_turns}")
print(f"Reasoning profile: {result.profile}")  # 'deductive', 'brute_force', or 'hallucination'
```

---

## License

Released under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). All procedural generation code, evaluation logic, and scoring infrastructure is in the public domain. Use it, fork it, build on it.

---

*"To measure AGI, we must stop asking models what they know — and start measuring how they search."*
