# 3. Method

**Target:** 1.5 pages. Status: first draft (Apr 21).

---

## 3.1 Single-block tool-integrated reasoning

A tool-integrated rollout interleaves natural-language reasoning with one
execution of a Python interpreter and a final boxed answer. Concretely, for a
problem $x$ and a reasoning-tuned policy $\pi_\theta$, a trajectory has the form
\[
\texttt{<think>} r_1 \texttt{</think>} \;\;
\texttt{<tool\_call>} c \texttt{</tool\_call>} \;\;
\texttt{<tool\_response>} o \texttt{</tool\_response>} \;\;
r_2 \;\; \texttt{\textbackslash boxed\{}\hat{y}\texttt{\}}
\]
where $r_1$ is initial reasoning, $c$ is Python code, $o$ is the sandbox's
stdout, $r_2$ is post-tool interpretation, and $\hat{y}$ is the extracted
answer. The model emits $c$ in a format native to its chat template
(hermes-style JSON for Qwen3-Thinking); the sandbox executes $c$ and the
stdout $o$ is injected back into the conversation before $r_2$ is generated.

We restrict the trajectory to **at most one** tool call. This keeps
rollout complexity bounded, simplifies credit assignment for RL, and
separates tool-use benefit from orchestration complexity — a multi-turn
setting would confound the two. Chain-of-thought rollouts remove the
\texttt{<tool\_call>} step entirely, producing a single uninterrupted
reasoning trace. This is the only difference between our TIR and CoT
conditions.

## 3.2 Rollout mechanism

**Sandbox.** Code runs in an isolated subprocess with a 30 s wall-clock
timeout, a 2 GB address-space limit, and a stdout cap of 4 kB. An
AST-level import whitelist permits `numpy`, `scipy`, `sympy`, `pint`, and
the Python standard library `math`, `cmath`, `statistics`, `fractions`,
`decimal`, `itertools`, `functools`, `collections`, `random`, `re`,
`json`, `io`, `typing`, `time`. On error, a short reminder
(``your single tool execution has been used; give your best answer
now'') is appended to the injected response; this prevents a
documented failure mode where the model retries the tool, triggering a
format mismatch against the single-call contract.

**Think-interrupt.** Reasoning-tuned bases can over-spend tokens on the
`<think>` block and starve the tool-call budget. Following ScaleRL
[CITE:scalerl], we allocate an explicit thinking budget (12,288 tokens in
our runs). If a rollout reaches the budget without closing `</think>`,
we inject a short interrupt phrase
(``Okay, I've thought enough. Time to write my response.</think>'')
with training mask set to zero, then continue generation for the tool
call with a dedicated budget (2,048 tokens). The final answer phase
has its own budget (4,096 tokens). This three-phase structure converts
the `max_response_length` hyperparameter into a semantically meaningful
allocation, and lets us measure per-phase token usage as a training
signal. Our implementation patches the VeRL \texttt{ToolAgentLoop}
[CITE:verl]; mask-zero tokens propagate correctly through the GRPO
loss.

**TIR vs CoT configuration.** Both conditions share a single system
prompt style ("You are an expert physics problem solver. … End with
your final answer as \textbackslash boxed\{…\}."); the TIR variant adds
one sentence describing tool availability and the permitted packages.
The CoT system prompt has no mention of tools. This parallel structure
is deliberate — prompt style is not a confound when comparing the two
conditions.

## 3.3 Training algorithm

We optimize with **Dr. GRPO** [CITE:drgrpo], which removes the
length- and standard-deviation-normalization terms from the original
group-relative policy optimization objective [CITE:grpo]. The empirical
improvements of DAPO [CITE:dapo] decompose into four techniques; we
adopt the three that are compatible with our asynchronous training loop:
**asymmetric clipping** ($\epsilon_\text{low} = 0.2$,
$\epsilon_\text{high} = 0.28$), **token-mean loss aggregation**, and
**overlong-response reward shaping**. We omit DAPO's dynamic sampling
because it requires synchronous group construction; with the KL term
off (standard under DAPO), groups where all rollouts succeed or all
fail contribute exactly zero to the policy gradient, reproducing the
filtering effect at the cost of extra rollout compute.

**Reward.** Binary correctness on the extracted boxed answer. A rule-based
verifier (symbolic match for expressions via SymPy; numerical match with
relative tolerance for numerical answers; canonicalization for MCQ and
interval types) handles the majority of cases; an xVerify-7B judge
[CITE:xverify] served on a dedicated GPU resolves cases the rule verifier
cannot adjudicate. The rule-plus-LLM combination is standard for
open-ended physics answers (cf. UGPhysics's MARJ pipeline
[CITE:ugphysics]). We do not add a length penalty in the main run;
an ablation with $\lambda > 0$ is reported in Appendix [TODO].

**Implementation details deferred to appendix.** Hermes vs
qwen3\_coder tool-call formats, Ray topology (1 + 6 + 1 A100 nodes),
FSDP sharding policy, and the exact VeRL patch for think-interrupt
are specified in Appendix [CITE:appendix_method].
