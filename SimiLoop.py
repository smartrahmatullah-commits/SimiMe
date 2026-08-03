"""
SimiLoop - Final Locked Architecture Implementation

Strict enforcement of responsibilities:
- NiAK builds COMPLETE CONTEXT (xDLM, Memory, Workspace, SQL results, metadata).
- Meta-Cognition (LLM) is ONLY an Observer: returns mini-problems (id + description).
- A2A (LLM) assigns specialist chains to mini-problems.
- Router builds task_specs and chain_contexts and dispatches work.
- Chains execute using only chain_context and choose their internal reasoning techniques.
- Fusion (Neuro Analyzer, LLM) analyzes chain outputs and may request more mini-problems.
- Meta-Cognition turns Fusion requests into new mini-problems.
- Reflection verifies final answer against xDLM inside COMPLETE CONTEXT.
- No Skills module. No loaders. No DB access. No xDLM loading inside SimiLoop.
"""

from typing import Dict, Any, List
import requests
import json
import ast
import threading
import time
from copy import deepcopy
import uuid

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:0.5b"
JSON_SYSTEM_INSTRUCTION = (
    "You must respond with ONLY a single valid JSON object or JSON array. "
    "No extra text, no markdown fences."
)

# Final resource labels (contract with NiAK) — Skills removed per instruction
RESOURCE_LABELS = ["Knowledge", "Memory", "Workspace", "SQL", "External"]
RESOURCE_KEY_MAP = {
    "Knowledge": "knowledge",
    "Memory": "memory",
    "Workspace": "workspace",
    "SQL": "sql_results",
    "External": "external_sources",
}

# Router's available specialist chains (Router does not choose; A2A assigns destination)
AVAILABLE_CHAINS = {
    "MathChain": {"capabilities": ["math", "algebra", "calculus", "trig"], "healthy": True},
    "LogicChain": {"capabilities": ["proof", "logic", "theorem"], "healthy": True},
    "ProgrammingChain": {"capabilities": ["code", "implement", "program"], "healthy": True},
    "VerificationChain": {"capabilities": ["verify", "check", "validate"], "healthy": True},
    "GeneralChain": {"capabilities": ["explain", "summarize", "general"], "healthy": True},
}

# Execution policy defaults
DEFAULT_TIMEOUT = 30  # seconds per chain
DEFAULT_PRIORITY = 5

# Fusion loop safety
MAX_FUSION_CYCLES = 3

# ---------------- LLM UTILITIES ----------------
def call_ollama(prompt: str, temperature: float = 0.7, max_retries: int = 2) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JSON_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    last_error = "unknown"
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries:
                time.sleep(1)
                continue
    return json.dumps({"error": last_error})


def safe_json_parse(text: str) -> Any:
    if not text:
        return {"error": "empty_response", "raw": ""}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(candidate)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except Exception:
                    pass
    return {"error": "could_not_parse_json", "raw": cleaned}

# ---------------- PATH RESOLUTION ----------------
def _resolve_dot_path(obj: Any, path: str) -> Any:
    """
    Resolve dotted path with optional indices, e.g. 'knowledge.xdlm.Trig.ids[0]'.
    Returns a deep copy or None if not found.
    """
    try:
        node = obj
        parts = []
        cur = ""
        i = 0
        while i < len(path):
            ch = path[i]
            if ch == ".":
                if cur:
                    parts.append(cur)
                    cur = ""
                i += 1
                continue
            if ch == "[":
                j = path.find("]", i)
                if j == -1:
                    return None
                cur += path[i : j + 1]
                i = j + 1
                continue
            cur += ch
            i += 1
        if cur:
            parts.append(cur)
        for p in parts:
            if "[" in p and p.endswith("]"):
                key, rest = p.split("[", 1)
                idx = int(rest[:-1])
                node = (node or {}).get(key)
                if node is None or not isinstance(node, (list, tuple)):
                    return None
                node = node[idx]
            else:
                if not isinstance(node, dict):
                    return None
                node = node.get(p)
            if node is None:
                return None
        return deepcopy(node)
    except Exception:
        return None

# ---------------- META-COGNITION (Observer) ----------------
class MetaCognition:
    """
    Observer only: reads COMPLETE CONTEXT and returns a list of mini-problems.
    Does NOT choose chains, create navigation, build task_specs, or reason about execution.
    """

    def decompose_problem(self, full_context: Dict[str, Any], temperature: float = 0.4) -> List[Dict[str, Any]]:
        """
        Return a list of mini-problems: [{'id':..., 'mini_problem': 'description string'}].
        """
        user_q = full_context.get("user_question") or full_context.get("question") or "<no question provided>"
        prompt = (
            "You are Meta-Cognition (Observer). You are given the COMPLETE Context prepared by NiAK and the user's request. "
            "Do NOT create task specifications, navigation maps, or assign chains. "
            "Your only job is to split the user's request into a list of small MINI-PROBLEMS. "
            "Return ONLY a JSON LIST of objects with fields: id (unique string) and mini_problem (short description).\n\n"
            f"User question: {user_q}\n\nContext: {json.dumps(full_context)}"
        )
        raw = call_ollama(prompt, temperature=temperature)
        parsed = safe_json_parse(raw)
        if isinstance(parsed, list):
            cleaned = []
            for item in parsed:
                if isinstance(item, dict) and "mini_problem" in item:
                    if "id" not in item:
                        item["id"] = str(uuid.uuid4())
                    cleaned.append(item)
            return cleaned
        return []

    def refine_from_fusion(self, full_context: Dict[str, Any], fusion_request: Dict[str, Any], temperature: float = 0.45) -> List[Dict[str, Any]]:
        """
        Convert Fusion's high-level requested_work into mini-problems (only).
        """
        prompt = (
            "You are Meta-Cognition. Fusion requested additional work described as:\n"
            f"{json.dumps(fusion_request)}\n\n"
            "Using ONLY the provided COMPLETE Context, return ONLY a JSON LIST of MINI-PROBLEMS (id and mini_problem keys). "
            "Do NOT create navigation or task_specs. Meta-Cognition only returns mini-problems.\n\n"
            f"Context: {json.dumps(full_context)}"
        )
        raw = call_ollama(prompt, temperature=temperature)
        parsed = safe_json_parse(raw)
        if isinstance(parsed, list):
            cleaned = []
            for item in parsed:
                if isinstance(item, dict) and "mini_problem" in item:
                    if "id" not in item:
                        item["id"] = str(uuid.uuid4())
                    cleaned.append(item)
            return cleaned
        return []

# ---------------- A2A (LLM) ----------------
class A2ACouncil:
    """
    A2A assigns the best specialist chain to every mini-problem.
    It receives full_context and the list of mini-problems and returns a mapping {mini_problem_id: chain_name}.
    """

    def assign_chains(self, full_context: Dict[str, Any], mini_problems: List[Dict[str, Any]], temperature: float = 0.35) -> Dict[str, str]:
        prompt = (
            "You are the A2A Council. You are given the COMPLETE Context and a list of MINI-PROBLEMS. "
            "For each mini-problem, select the BEST specialist chain to handle it. Available chains: "
            f"{json.dumps(list(AVAILABLE_CHAINS.keys()))}. Only choose from these names. "
            "Return ONLY a JSON OBJECT mapping mini_problem_id -> chain_name. Example: {\"mp-1\":\"MathChain\"}\n\n"
            f"Context: {json.dumps(full_context)}\n\nMini-problems: {json.dumps(mini_problems)}"
        )
        raw = call_ollama(prompt, temperature=temperature)
        parsed = safe_json_parse(raw)
        # Validate mapping
        mapping = {}
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(k, str) and isinstance(v, str) and v in AVAILABLE_CHAINS:
                    mapping[k] = v
        # Fallback: basic heuristic if mapping empty
        if not mapping:
            for mp in mini_problems:
                mpid = mp.get("id")
                text = mp.get("mini_problem", "").lower()
                # simple heuristics
                if any(tok in text for tok in ["prove", "derive", "theorem", "logic", "proof"]):
                    mapping[mpid] = "LogicChain"
                elif any(tok in text for tok in ["implement", "code", "program"]):
                    mapping[mpid] = "ProgrammingChain"
                elif any(tok in text for tok in ["calculate", "compute", "solve", "+", "-", "*", "/", "^", "sin", "cos", "tan"]):
                    mapping[mpid] = "MathChain"
                elif any(tok in text for tok in ["verify", "check", "validate"]):
                    mapping[mpid] = "VerificationChain"
                else:
                    mapping[mpid] = "GeneralChain"
        return mapping

# ---------------- ROUTER (deterministic) ----------------
class Router:
    """
    Router converts mini-problems + assigned chain -> task_specs.
    Router builds navigation by slicing the COMPLETE CONTEXT and decides execution policy.
    Router does NOT decide which chain to use. It only prepares the work.
    """

    def __init__(self, available_chains: Dict[str, Dict[str, Any]] = None):
        self.available_chains = available_chains or AVAILABLE_CHAINS

    def infer_allowed_resources(self, mini_problem: str) -> List[str]:
        """
        Heuristic to infer resource labels from the mini_problem text.
        Default to Knowledge; add Workspace if required.
        """
        text = mini_problem.lower()
        resources = set()
        # domain cues
        if any(tok in text for tok in ["memory", "experience", "recall"]):
            resources.add("Memory")
        if any(tok in text for tok in ["sql", "table", "database", "query", "rows"]):
            resources.add("SQL")
        if any(tok in text for tok in ["external", "web", "source", "article"]):
            resources.add("External")
        # procedural cues
        if any(tok in text for tok in ["prove", "derive", "calculate", "solve", "implement", "construct", "verify", "check"]):
            resources.add("Workspace")
        # default
        if not resources:
            resources.add("Knowledge")
        return [r for r in resources if r in RESOURCE_LABELS]

    def build_navigation(self, full_context: Dict[str, Any], mini_problem: str, allowed_resources: List[str]) -> Dict[str, List[str]]:
        """
        Conservative navigation builder:
        - For Knowledge: search knowledge.xdlm keys/titles/descs for tokens in mini_problem.
        - For Memory: include memory indices with matching text; fallback to a few memory entries.
        - For Workspace: include common workspace paths if present.
        - For SQL: include keys under sql_results.
        - For External: include up to first few external_sources indices.
        """
        nav = {}
        text_tokens = set(mini_problem.lower().split())
        # Knowledge
        if "Knowledge" in allowed_resources:
            kpaths = []
            knowledge = full_context.get("knowledge", {})
            xdlm = knowledge.get("xdlm") if isinstance(knowledge, dict) else None
            if isinstance(xdlm, dict):
                for key, val in xdlm.items():
                    key_l = key.lower()
                    if any(tok in key_l for tok in text_tokens):
                        kpaths.append(f"knowledge.xdlm.{key}")
                    if isinstance(val, dict):
                        for field in ("title", "desc", "id"):
                            fv = val.get(field)
                            if isinstance(fv, str) and any(tok in fv.lower() for tok in text_tokens):
                                kpaths.append(f"knowledge.xdlm.{key}")
            if not kpaths and "knowledge" in full_context:
                kpaths.append("knowledge")
            nav["knowledge_paths"] = list(dict.fromkeys(kpaths))
        # Memory
        if "Memory" in allowed_resources:
            memory = full_context.get("memory", [])
            mids = []
            for idx, mem in enumerate(memory):
                mem_text = json.dumps(mem).lower()
                if any(tok in mem_text for tok in text_tokens):
                    mids.append(f"memory[{idx}]")
            if not mids and memory:
                mids = [f"memory[{i}]" for i in range(min(3, len(memory)))]
            nav["memory_ids"] = mids
        # Workspace
        if "Workspace" in allowed_resources:
            wpaths = []
            candidates = ["workspace.current_proof", "workspace.scratch", "workspace.artifacts"]
            for p in candidates:
                if _resolve_dot_path(full_context, p) is not None:
                    wpaths.append(p)
            if not wpaths and "workspace" in full_context:
                wpaths.append("workspace")
            nav["workspace_paths"] = wpaths
        # SQL
        if "SQL" in allowed_resources:
            sql_keys = list(full_context.get("sql_results", {}).keys()) if isinstance(full_context.get("sql_results", {}), dict) else []
            nav["sql_refs"] = sql_keys[:5]
        # External
        if "External" in allowed_resources:
            externals = full_context.get("external_sources", [])
            nav["external_refs"] = [f"external_sources[{i}]" for i in range(min(len(externals), 5))]
        return nav

    def create_task_spec(self, full_context: Dict[str, Any], mini_problem: Dict[str, Any], assigned_chain: str) -> Dict[str, Any]:
        """
        Build the full task_spec for the assigned chain.
        Router decides allowed_resources, navigation, priority, timeout, retries.
        """
        mp_text = mini_problem.get("mini_problem", "")
        allowed = self.infer_allowed_resources(mp_text)
        navigation = self.build_navigation(full_context, mp_text, allowed)
        timeout = DEFAULT_TIMEOUT
        priority = DEFAULT_PRIORITY
        if any(tok in mp_text.lower() for tok in ["urgent", "critical", "verify", "prove"]):
            priority = 1
            timeout = max(10, DEFAULT_TIMEOUT // 2)
        ts = {
            "id": str(uuid.uuid4()),
            "origin_mini_problem_id": mini_problem.get("id"),
            "chain": assigned_chain if assigned_chain in AVAILABLE_CHAINS else "GeneralChain",
            "task": mp_text,
            "allowed_resources": allowed,
            "navigation": navigation,
            "priority": priority,
            "timeout": timeout,
            "retries": 1
        }
        return ts

# ---------------- BUILD CHAIN CONTEXT ----------------
def build_chain_context_for_router(full_context: Dict[str, Any], task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically slice the COMPLETE CONTEXT according to task_spec.allowed_resources and navigation.
    """
    ctx = {"user_question": full_context.get("user_question"), "meta": deepcopy(full_context.get("meta", {}))}
    missing = []
    for label in task_spec.get("allowed_resources", []):
        key = RESOURCE_KEY_MAP.get(label)
        if key is None:
            missing.append(label)
            continue
        if key in full_context:
            ctx[key] = deepcopy(full_context[key])
        else:
            missing.append(label)
    nav = task_spec.get("navigation", {}) or {}
    resolved = {}
    for kind, paths in nav.items():
        resolved[kind] = {}
        for p in (paths or []):
            v = _resolve_dot_path(full_context, p)
            if v is None:
                missing.append(p)
            else:
                resolved[kind][p] = v
    if resolved:
        ctx["_navigation_resolved"] = resolved
    if missing:
        ctx["_meta_missing"] = list(dict.fromkeys(missing))
    ctx["_meta_provenance"] = {"provided_by": "NiAK_via_Router", "task_spec_id": task_spec.get("id"), "origin_mini_problem_id": task_spec.get("origin_mini_problem_id")}
    return ctx

# ---------------- SPECIALIST CHAINS ----------------
class Chains:
    """
    Chains are autonomous and choose their internal reasoning methods.
    Each chain receives only (task_spec, chain_context).
    """

    @staticmethod
    def _base_prompt(chain_name: str, task_spec: Dict[str, Any], chain_context: Dict[str, Any]) -> str:
        return (
            f"You are the specialist '{chain_name}'. You received a single mini-task and a navigation-only chain_context. "
            "You MUST NOT fetch or access any data beyond the provided chain_context. You are free to pick your own internal reasoning style (CoT, ToT, GoT, reflection, self-critique, etc.).\n\n"
            f"Task spec: {json.dumps(task_spec)}\nChain Context: {json.dumps(chain_context)}\n\n"
            "Return ONLY a JSON object with fields: status ('ok'|'insufficient_context'|'error'), task_id, result (string or structured), artifacts (optional), used_navigation (list), xdlm_rules_applied (list), confidence (0-100)."
        )

    @staticmethod
    def execute(chain_name: str, task_spec: Dict[str, Any], chain_context: Dict[str, Any], temperature: float = 0.75) -> Dict[str, Any]:
        prompt = Chains._base_prompt(chain_name, task_spec, chain_context)
        raw = call_ollama(prompt, temperature=temperature)
        parsed = safe_json_parse(raw)
        if isinstance(parsed, dict) and "task_id" not in parsed:
            parsed["task_id"] = task_spec.get("id")
        return parsed

# ---------------- FUSION (Neuro Analyzer) ----------------
class Fusion:
    def analyze(self, full_context: Dict[str, Any], task_specs: List[Dict[str, Any]], chain_outputs: Dict[str, Any], temperature: float = 0.4) -> Dict[str, Any]:
        prompt = (
            "You are Fusion, the Neuro Analyzer. You receive the COMPLETE Context, the executed task_specs, and the outputs from all specialist chains. "
            "Analyze contradictions, missing information, confidence, and quality. Decide whether the overall result is sufficient.\n\n"
            "Return ONLY a JSON object:\n"
            "  action: 'accept_final'|'combine_and_finalize'|'request_more'\n"
            "  combined_answer: string (if action == accept_final or combine_and_finalize)\n"
            "  rationale: string\n"
            "  requested_work: list of high-level requests (if action == request_more)\n\n"
            f"Full Context: {json.dumps(full_context)}\nTask specs: {json.dumps(task_specs)}\nChain outputs: {json.dumps(chain_outputs)}"
        )
        raw = call_ollama(prompt, temperature=temperature)
        return safe_json_parse(raw)

# ---------------- REFLECTION ----------------
class Reflection:
    def verify(self, full_context: Dict[str, Any], combined_answer: str, temperature: float = 0.3) -> Dict[str, Any]:
        prompt = (
            "You are Reflection. Verify the candidate combined answer strictly against xDLM rules, methods, and concepts present in the COMPLETE Context (no external calls). "
            "Return ONLY JSON: {passed: bool, issues: [...], final_answer: '...'}\n\n"
            f"Context: {json.dumps(full_context)}\nCandidate answer: {combined_answer}"
        )
        raw = call_ollama(prompt, temperature=temperature)
        return safe_json_parse(raw)

# ---------------- SIMILOOP ORCHESTRATOR ----------------
class SimiLoop:
    def __init__(self, a2a: A2ACouncil, router: Router, max_fusion_cycles: int = MAX_FUSION_CYCLES):
        self.meta = MetaCognition()
        self.a2a = a2a
        self.router = router
        self.fusion = Fusion()
        self.reflection = Reflection()
        self.max_fusion_cycles = max_fusion_cycles

    def _execute_task_with_timeout(self, task_spec: Dict[str, Any], chain_context: Dict[str, Any], results: Dict[str, Any], lock: threading.Lock):
        chain_name = task_spec.get("chain")
        timeout = task_spec.get("timeout", DEFAULT_TIMEOUT)
        result_holder = {}

        def target():
            try:
                out = Chains.execute(chain_name, task_spec, chain_context)
            except Exception as e:
                out = {"status": "error", "error": str(e), "task_id": task_spec.get("id")}
            result_holder["out"] = out

        th = threading.Thread(target=target)
        th.start()
        th.join(timeout)
        if th.is_alive():
            out = {"status": "error", "error": "timeout", "task_id": task_spec.get("id")}
            # Note: in production use processes or LLM cancellation to truly stop work
        else:
            out = result_holder.get("out", {"status": "error", "error": "no_output", "task_id": task_spec.get("id")})
        with lock:
            results[task_spec.get("id")] = out

    def run(self, full_context: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Meta-Cognition -> mini-problems
        mini_problems = self.meta.decompose_problem(full_context)
        if not mini_problems:
            mini_problems = [{"id": str(uuid.uuid4()), "mini_problem": full_context.get("user_question", "<no question>")}]

        fusion_cycle = 0
        all_chain_outputs: Dict[str, Any] = {}
        executed_task_specs: List[Dict[str, Any]] = []

        while True:
            # 2) A2A assigns chains for mini-problems
            assignment = self.a2a.assign_chains(full_context, mini_problems)  # mapping mini_problem_id -> chain_name

            # 3) Router builds task_specs for each assigned mini-problem
            task_specs = []
            for mp in mini_problems:
                mpid = mp.get("id")
                assigned_chain = assignment.get(mpid, "GeneralChain")
                ts = self.router.create_task_spec(full_context, mp, assigned_chain)
                task_specs.append(ts)

            # 4) Execute task_specs in parallel (Router created chain_contexts)
            results: Dict[str, Any] = {}
            lock = threading.Lock()
            threads: List[threading.Thread] = []
            for ts in task_specs:
                chain_ctx = build_chain_context_for_router(full_context, ts)
                t = threading.Thread(target=self._execute_task_with_timeout, args=(ts, chain_ctx, results, lock))
                threads.append(t); t.start()
            for t in threads: t.join()

            all_chain_outputs.update(results)
            executed_task_specs.extend(task_specs)

            # 5) Fusion analyzes
            fusion_decision = self.fusion.analyze(full_context, executed_task_specs, all_chain_outputs)
            action = fusion_decision.get("action")

            if action in ("accept_final", "combine_and_finalize"):
                combined = fusion_decision.get("combined_answer") or ""
                verification = self.reflection.verify(full_context, combined)
                return {
                    "final_answer": verification.get("final_answer"),
                    "verification": verification,
                    "fusion_rationale": fusion_decision.get("rationale"),
                    "executed_task_specs": executed_task_specs,
                    "chain_outputs": all_chain_outputs
                }
            elif action == "request_more":
                if fusion_cycle >= self.max_fusion_cycles:
                    return {
                        "error": "max_fusion_cycles_reached",
                        "fusion_rationale": fusion_decision.get("rationale"),
                        "chain_outputs": all_chain_outputs
                    }
                requested_work = fusion_decision.get("requested_work", [])
                new_mini_problems = []
                for req in requested_work:
                    expanded = self.meta.refine_from_fusion(full_context, req)
                    if isinstance(expanded, list) and expanded:
                        new_mini_problems.extend(expanded)
                if not new_mini_problems:
                    return {
                        "error": "fusion_requested_more_but_meta_produced_nothing",
                        "fusion_decision": fusion_decision,
                        "chain_outputs": all_chain_outputs
                    }
                mini_problems = new_mini_problems
                fusion_cycle += 1
                continue
            else:
                return {
                    "error": "fusion_unknown_action",
                    "fusion_decision": fusion_decision,
                    "chain_outputs": all_chain_outputs
                }

# ---------------- DEMO (replace COMPLETE CONTEXT from NiAK in production) ----------------
if __name__ == "__main__":
    # Example COMPLETE CONTEXT (NiAK must produce this)
    example_context = {
        "user_question": "Provide a proof of sin^2(x) + cos^2(x) = 1 and an alternate approach.",
        "knowledge": {
            "xdlm": {
                "trig_pythagorean": {"id": "trig_pythagorean", "title": "Pythagorean identity", "desc": "sin^2+cos^2=1"},
                "unit_circle": {"id": "unit_circle", "title": "Unit circle approach"}
            }
        },
        "memory": [{"id": "mem_1", "text": "previous trig proof using unit circle"}],
        "workspace": {"current_proof": "starting notes"},
        "sql_results": {},
        "external_sources": [],
        "meta": {"request_id": "demo-001", "trust_level": "high"}
    }

    a2a = A2ACouncil()
    router = Router()
    engine = SimiLoop(a2a=a2a, router=router, max_fusion_cycles=2)
    result = engine.run(example_context)
    print(json.dumps(result, indent=2))
