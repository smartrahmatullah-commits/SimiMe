# SimiMe

**Founder:** Mohammad Rahmatullah

---

## 🚀 Mission

SimiMe is building the smartest AI the world has ever seen — AI that thinks, learns, and adapts like never before.

---

## 🧠 What We Do

We are creating intelligent systems that push the boundaries of what AI can achieve.

---

## 📅 Founded

August 2026

---

## 👤 Founder

**Mohammad Rahmatullah** — Building the future of AI, one line of code at a time.

---

*More coming sHigh-level summary

SimiLoop is an orchestrator for solving a user request by splitting it into small "mini-problems", assigning each mini-problem to specialist chains, running those chains in parallel with a restricted slice of the full context, then fusing and verifying the outputs. It enforces strict separation of responsibilities (NiAK provides COMPLETE CONTEXT; Meta-Cognition only observes; A2A assigns chains; Router prepares work; Chains execute only with their chain_context; Fusion and Reflection finalize).
Main components

COMPLETE CONTEXT (external, produced by NiAK): contains user_question, knowledge (xDLM), memory, workspace, sql_results, external_sources, meta, etc.
MetaCognition: Observer-only. Decomposes the user request into a JSON list of mini-problems (id + description). Also refines Fusion requests into mini-problems.
A2ACouncil: An LLM component that maps each mini-problem id to a specialist chain name (MathChain, LogicChain, ProgrammingChain, VerificationChain, GeneralChain). Falls back to simple heuristics if LLM mapping is missing.
Router: Deterministic builder of task_specs. For each mini-problem it:
infers allowed resource labels (Knowledge, Memory, Workspace, SQL, External),
builds navigation paths into the COMPLETE CONTEXT (e.g., knowledge.xdlm.some_key, memory[0], workspace.current_proof),
sets priority/timeout and packs everything into a task_spec.
build_chain_context_for_router: Slices the COMPLETE CONTEXT according to allowed_resources + navigation and returns chain_context that the chain is allowed to see.
Chains (specialists): Autonomous workers. Each receives only (task_spec, chain_context) and runs using its own reasoning style. They must return a JSON object with status/task_id/result etc. Chains call the LLM via call_ollama.
Fusion (Neuro Analyzer): Receives all chain outputs + task_specs + COMPLETE CONTEXT and returns an action:
accept_final / combine_and_finalize -> produce combined_answer
request_more -> return high-level requested_work (Fusion can ask MetaCognition to expand those into new mini-problems)
Reflection: Verifies the combined answer strictly against xDLM rules in the COMPLETE CONTEXT and returns final_answer plus issues/passed flag.
SimiLoop orchestrator: coordinates the loop:
MetaCognition -> mini-problems
A2A assigns chains
Router builds task_specs
Execute task_specs in parallel threads (each chain executes with timeout)
Fusion analyzes chain outputs
If fusion accepts or finalizes -> Reflection verifies -> return final result
If fusion requests more -> MetaCognition expands -> loop (bounded by MAX_FUSION_CYCLES)
Else -> error
Important implementation details & constraints

LLM integration: call_ollama posts JSON to OLLAMA_URL with a strict system instruction that the model must reply with a single JSON object/array. Responses are parsed with safe_json_parse that strips fences and attempts to extract JSON.
Resource contract: the orchestrator expects NiAK to already provide the COMPLETE CONTEXT. SimiLoop is explicit: "No Skills, No loaders, No DB access, No xDLM loading inside SimiLoop."
Execution: chains run in threads with join and a timeout. If a thread times out, the result is marked as error: timeout (note: true cancellation is not implemented).
Fusion cycles: limited by MAX_FUSION_CYCLES (safety).
Developer checks: Chains are required to return JSON; the code tries to enforce/parse that but correctness depends on the LLM following the system instruction.
Available chains and heuristics: If A2A fails to produce valid mapping, simple keyword heuristics assign chains.
Demo: there's a demo example at the bottom that shows running the engine with a trig proof context.
Potential issues / observations I noticed

The Chains.base_prompt string near line 422 appears truncated in the file (it ends with "xdlm_rules[...]"). That likely needs fixing so the prompt is valid and complete.
The system requires an Ollama endpoint (OLLAMA_URL) and the specified MODEL to be running locally; otherwise call_ollama returns an error JSON.
Using threads for LLM work is OK for simple demos but for robust cancellation/isolated CPU/memory you’d want processes or an LLM client with cancellable requests.
The orchestrator assumes the LLM will always return parseable JSON; safe_json_parse is robust but may still fail on messy output.
How the runtime flow looks in practice (concise step-by-step)

NiAK produces COMPLETE CONTEXT (outside this module).
SimiLoop.meta.decompose_problem splits the user question into mini-problems.
A2A assigns each mini-problem to a specialist chain.
Router creates deterministic task_specs: allowed_resources, navigation slices, timeout/priority.
For each task_spec, Router/context builder slices only the needed parts of COMPLETE CONTEXT into chain_context.
Each chain executes (LLM prompt) using only its chain_context. Outputs are collected in parallel.
Fusion reviews outputs and decides: accept, combine, or request more.
If request_more, MetaCognition turns those requests into new mini-problems and the loop repeats (up to a cap).
On finalization, Reflection verifies final answer against xDLM and returns verified final_answer.

*If you want, next I can:oon...* 




            thanks for coming..............................!
