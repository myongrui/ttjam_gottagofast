#!/usr/bin/env python3
"""Unattended candidate generator: OpenAI proposes, race.py measures.

Runs standalone (cron, nohup, launchd) with no Claude Code session, so it does
not consume a Claude usage window. One cycle is:

    read ledger + findings  ->  ask the model for a candidate  ->  static screen
    ->  write candidates/vNNN_*.py  ->  tools/race.py  ->  record  ->  repeat

Two things this deliberately does NOT trust:

1. The generated code. It is executed on a GPU box minutes later, so it passes a
   static screen first (see FORBIDDEN). The screen is not a sandbox -- it is a
   cheap filter for the specific failure modes the kernel-generation literature
   documents, and it complements the harness's runtime integrity checks rather
   than replacing them.
2. Its own budget. Unattended loops bill for API tokens AND pod time, so it stops
   on iteration count, wall-clock, or a run of non-improvements, whichever first.

Credentials come from the environment; this file never stores or logs a key.

    export OPENAI_API_KEY=sk-...
    export OPENAI_MODEL=<a model your account can use>
    python3 tools/autoloop.py --max-iters 20 --max-hours 6
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ledger  # noqa: E402
import routing  # noqa: E402

API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LOG = ROOT / "results" / "autoloop.log"
# Cloudflare rejects urllib's default "Python-urllib/x.y" agent with 403/1010 in
# front of api.runpod.io, so every request this file makes names itself.
USER_AGENT = "techjam-autoloop/1.0"
STATUS = ROOT / "results" / "STATUS.md"

# Patterns rejected before a generated candidate is ever executed. The first
# three map to the reward-hacking families in arXiv 2509.14279; the rest keep a
# candidate from reaching outside its own process.
FORBIDDEN = [
    (r"torch\.cuda\.(Stream|stream)\b",
     "creates a CUDA stream -- the documented timing-escape exploit"),
    (r"torch\.backends|set_float32_matmul_precision|allow_tf32",
     "mutates torch global state, which redefines the baseline"),
    (r"lru_cache|@cache\b|\bcached_property\b|\bglobal\s+\w*cache",
     "memoizes across calls -- the state-caching exploit"),
    (r"\bsubprocess\b|\bos\.system|\bshutil\b|\bsocket\b|\brequests\b|urllib",
     "performs I/O or process control; candidates are pure model code"),
    (r"open\s*\([^)]*['\"][wa]", "writes files"),
    (r"torch_transformer_benchmark|tools/harness|sys\.modules\[.bench.\]",
     "reaches into the benchmark or harness"),
]

SYSTEM = """You optimize PyTorch transformer inference. You write ONE candidate \
implementation per reply, as a single fenced ```python block and nothing else.

Hard contract:
  def build_model(config, bench) -> nn.Module      # required
  def load_from_baseline(model, baseline) -> None  # optional; needed when
                                                   # weights do not map 1:1
forward(x, valid_token_mask=None) -> [batch, seq, d_model].
The module docstring MUST state the hypothesis and its mechanism.

You must not: create CUDA streams, mutate torch global state (backends, tf32,
matmul precision), cache results across calls, perform file or network I/O, or
modify the benchmark. Correctness is verified against the reference on five
input distributions; a wrong answer scores zero regardless of speed."""


# Only these two are ever reachable. "terminate" is deliberately absent: it
# destroys /workspace, and no automated path should be able to reach it.
POD_ACTIONS = ("stop", "start")


def pod_action(action: str) -> bool:
    """Trigger a pod state transition via the Runpod REST v2 API.

    Endpoint and enum taken from https://api.runpod.io/v2/openapi.json
    (POST /v2/pods/{id}/action, PodAction = start|stop|restart|terminate).
    """
    if action not in POD_ACTIONS:
        raise ValueError(f"refusing action {action!r}; allowed: {POD_ACTIONS}")
    key = os.environ.get("RUNPOD_API_KEY")
    pod = os.environ.get("RUNPOD_POD_ID", "132yirw3l7rq7s")
    if not key:
        log(f"cannot {action} pod: RUNPOD_API_KEY not set — pod left running")
        return False
    url = f"https://api.runpod.io/v2/pods/{pod}/action"
    req = urllib.request.Request(
        url, data=json.dumps({"action": action}).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()
        log(f"pod {pod}: {action} requested")
        return True
    except urllib.error.HTTPError as e:
        body = e.read()[:200].decode(errors="replace")
        # 409 means the pod is already in the state we asked for. Starting an
        # already-running pod is success, not failure -- treating it as failure
        # made --start-pod abort a run whenever the pod was already up.
        if e.code == 409 and "not valid for status" in body:
            log(f"pod {pod}: already {'running' if action == 'start' else 'stopped'}")
            return True
        log(f"pod {action} failed: HTTP {e.code} {body}")
    except Exception as e:
        log(f"pod {action} failed: {type(e).__name__}: {e}")
    return False


def push_results(reason: str) -> bool:
    """Commit and push output so a scheduled reader sees current state.

    Called after EVERY iteration, not only at exit: a long run that pushed only
    on completion would be invisible to a check that fires while it is still
    running, which is precisely when someone wants to look. Rebases first,
    because a second session may have pushed in the meantime.
    """
    try:
        subprocess.run(["git", "add", "results", "candidates"], cwd=ROOT,
                       check=True, capture_output=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"],
                          cwd=ROOT).returncode == 0:
            return True                      # nothing new; stay quiet in-loop
        subprocess.run(["git", "commit", "-m", f"autoloop: {reason}"], cwd=ROOT,
                       check=True, capture_output=True)
        # Integrate anyone else's commits before pushing; --autostash keeps a
        # dirty tree from aborting the rebase.
        pull = subprocess.run(["git", "pull", "--rebase", "--autostash"],
                              cwd=ROOT, capture_output=True)
        if pull.returncode:
            log("pull --rebase failed; leaving the commit local for this cycle")
            subprocess.run(["git", "rebase", "--abort"], cwd=ROOT,
                           capture_output=True)
            return False
        subprocess.run(["git", "push"], cwd=ROOT, check=True, capture_output=True)
        log(f"pushed: {reason}")
        return True
    except subprocess.CalledProcessError as e:
        detail = e.stderr[:200].decode(errors="replace") if e.stderr else str(e)
        log(f"push failed: {detail}")
        return False


def wait_for_pod(attempts: int = 30, delay: int = 10) -> bool:
    """Poll until the pod answers over ssh, or give up.

    A started pod reports RUNNING well before sshd accepts connections, so
    polling the real transport is the only trustworthy readiness signal.
    """
    for i in range(1, attempts + 1):
        try:
            probe = subprocess.run([str(ROOT / "tools" / "podrun"), "echo ALIVE"],
                                   capture_output=True, text=True, cwd=ROOT,
                                   timeout=120)
            ready = "ALIVE" in probe.stdout
        except subprocess.TimeoutExpired:
            # A probe that hangs means the pod is not up yet, not that the
            # caller should crash -- this guard exists to prevent hangs.
            ready = False
        if ready:
            log(f"pod ready after {i} probe(s)")
            return True
        # Silence for ten minutes is indistinguishable from a hang, so report.
        log(f"waiting for pod ssh... probe {i}/{attempts}")
        time.sleep(delay)
    return False


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from a local .env, if present.

    An already-exported variable always wins, so an explicit `export` in the
    shell overrides the file rather than being silently ignored. Values are not
    logged anywhere.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def write_status(reason: str, best_score: float, iterations: int,
                 decision: "routing.Decision | None") -> None:
    """Layer 3. A detached script cannot start a Claude Code session, so the
    stop condition is recorded to a file for a human to pick up."""
    entries = ledger.load()
    bias = routing.screen_bias(entries)
    lines = [
        "# autoloop stopped", "",
        f"- **reason**: {reason}",
        f"- **iterations run**: {iterations}",
        f"- **best full-sweep score**: {best_score:.4f}",
        f"- **stopped at**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
    ]
    if decision is not None:
        lines += [
            f"- **last routing mode**: {decision.mode} — {decision.reason}",
            f"- **profile**: {decision.profile}   **parent**: {decision.parent}",
            f"- **beam/elite width**: {decision.beam_size}/{decision.elite_k}",
        ]
    lines += [
        f"- **screen-to-full contradiction rate**: {bias['rate']:.0%} "
        f"({bias['contradicted']} of {bias['confirmed'] + bias['contradicted']})"
        + (f" — offenders: {', '.join(bias['offenders'])}" if bias["offenders"] else ""),
        "", "## Recent attempts", "",
        "| candidate | score | decision | hypothesis |", "|---|---:|---|---|",
    ]
    for e in entries[-6:]:
        lines.append(f"| {e['candidate']} | {e.get('score', 0):.4f} | "
                     f"{e.get('decision', '-')} | {e.get('hypothesis', '')[:70]} |")
    lines += [
        "", "## Resume", "", "```",
        f"autoloop stopped: {reason}. Best {best_score:.4f} after {iterations} "
        f"iterations. Read results/STATUS.md, results/ledger.jsonl and LOOP.md, "
        f"diagnose the stall, and propose the next candidate.",
        "```", "",
    ]
    STATUS.write_text("\n".join(lines))
    log(f"wrote {STATUS.relative_to(ROOT)}")


def build_prompt(decision) -> str:
    entries = ledger.load()
    champ = ledger.champion(ledger.FULL_CASES, dtype="float32")
    parts = []

    findings = (ROOT / "LOOP.md")
    if findings.exists():
        text = findings.read_text()
        # Ship the findings and backlog, not the whole document.
        for header in ("## Established findings", "## Hypothesis backlog"):
            if header in text:
                chunk = text.split(header, 1)[1].split("\n## ", 1)[0]
                parts.append(f"{header}\n{chunk}")

    if champ:
        src = ROOT / "candidates" / champ["candidate"]
        parts.append(f"## Current champion ({champ['candidate']}, "
                     f"score {champ['score']})\n```python\n{src.read_text()}\n```")

    tried = []
    for e in entries:
        verdict = "FAILED" if e["score"] <= 0 else f"score {e['score']}"
        tried.append(f"- {e['candidate']}: {verdict} -- {e['hypothesis'][:120]}")
    if tried:
        parts.append("## Already tried (do not repeat a refuted idea)\n" + "\n".join(tried))

    parts.append(
        f"## Required move: {decision.mode}\n\n{decision.guidance}\n\n"
        + (f"Target cases: {', '.join(decision.target_cases)}.\n"
           if decision.target_cases else "")
        + "Propose ONE new candidate that beats the champion, of the kind "
          "required above. Change one thing, state the mechanism in the "
          "docstring. Reply with only the fenced python block.")
    return "\n\n".join(parts)


def _post(url: str, body: dict, key: str, timeout: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _text_from_responses(payload: dict) -> str:
    """Pull assistant text out of a Responses API payload.

    The output array interleaves reasoning items with message items, and only
    the latter carry output_text, so scan rather than indexing [0].
    """
    chunks = []
    for item in payload.get("output", []):
        for part in item.get("content") or []:
            if part.get("type") in ("output_text", "text") and part.get("text"):
                chunks.append(part["text"])
    if not chunks and payload.get("output_text"):
        chunks.append(payload["output_text"])
    return "\n".join(chunks)


def call_model(prompt: str, model: str, timeout: int = 600) -> str:
    """Call OpenAI, preferring the Responses API and falling back to Chat.

    Newer reasoning and Codex models are Responses-only; older ones are
    Chat-only. Rather than maintain a list of which is which, try the modern
    endpoint and fall back when the API says the model belongs on the other one.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (export it, or put it in .env)")

    attempts = [
        ("responses", f"{API_BASE}/responses",
         {"model": model, "instructions": SYSTEM, "input": prompt}),
        ("chat", f"{API_BASE}/chat/completions",
         {"model": model,
          "messages": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": prompt}]}),
    ]

    last = ""
    for name, url, body in attempts:
        try:
            payload = _post(url, body, key, timeout)
        except urllib.error.HTTPError as e:
            last = e.read()[:400].decode(errors="replace")
            # Only fall through when the API is telling us to use the other
            # endpoint; anything else is a real error worth surfacing now.
            if e.code == 404 and "endpoint" in last:
                log(f"{name} endpoint rejected {model}; trying the other one")
                continue
            raise SystemExit(f"OpenAI API error {e.code}: {last}")
        text = (_text_from_responses(payload) if name == "responses"
                else payload["choices"][0]["message"]["content"])
        if text.strip():
            return text
        last = f"{name} returned no text"
    raise SystemExit(f"OpenAI call failed: {last}")


def extract_code(reply: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, re.S)
    if not blocks:
        raise ValueError("reply contained no fenced code block")
    return max(blocks, key=len).strip() + "\n"


def screen(code: str) -> list[str]:
    problems = []
    for pattern, why in FORBIDDEN:
        if re.search(pattern, code):
            problems.append(why)
    if "def build_model" not in code:
        problems.append("missing required build_model(config, bench)")
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as e:
        problems.append(f"syntax error: {e}")
    return problems


def next_version() -> int:
    versions = [int(m.group(1))
                for p in (ROOT / "candidates").glob("v*.py")
                if (m := re.match(r"v(\d+)_", p.name))]
    return max(versions, default=0) + 1


def slug(code: str) -> str:
    first = (re.search(r'"""(.+)', code) or re.match(r"(.*)", code)).group(1)
    words = re.findall(r"[a-z0-9]+", first.lower())
    # Drop a leading version token so we do not emit v004_v004_...
    if words and re.fullmatch(r"v\d+", words[0]):
        words = words[1:]
    return "_".join(words[:4]) or "candidate"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--max-hours", type=float, default=6.0)
    p.add_argument("--stop-after-flat", type=int, default=5,
                   help="stop after this many consecutive non-improvements")
    p.add_argument("--profile", default="auto",
                   help="auto lets the router choose per cycle")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--model", default=None,
                   help="defaults to $OPENAI_MODEL (or .env); no built-in "
                        "fallback, since a guessed model id fails at the API")
    p.add_argument("--start-pod", action="store_true",
                   help="start the pod at launch and wait for ssh. With "
                        "--stop-pod this makes the run self-contained: no "
                        "manual RunPod step at either end")
    p.add_argument("--push-results", action="store_true",
                   help="commit and push results/ and candidates/ on exit. "
                        "Required for a scheduled cloud agent to see current "
                        "state: it clones the repo and cannot read this machine")
    p.add_argument("--stop-pod", action="store_true",
                   help="stop the pod when the loop exits, for any reason. "
                        "Strongly recommended for unattended runs: otherwise a "
                        "stall at 01:00 bills until you notice. Requires "
                        "RUNPOD_API_KEY. Never terminates, so /workspace is safe")
    p.add_argument("--eval-timeout", type=int, default=2400,
                   help="seconds before an evaluation is abandoned (default 40m). "
                        "Without this a hung ssh transport stalls the run forever")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="log routing guidance and each candidate's hypothesis")
    p.add_argument("--dry-run", action="store_true",
                   help="generate and screen, but do not spend pod time")
    args = p.parse_args()
    load_dotenv()
    args.model = args.model or os.environ.get("OPENAI_MODEL")
    if not args.model:
        raise SystemExit(
            "No model set. Put OPENAI_MODEL in .env or export it.\n"
            "List what your account can actually reach:\n"
            "  curl -s https://api.openai.com/v1/models \\\n"
            "    -H \"Authorization: Bearer $OPENAI_API_KEY\" \\\n"
            "    | python3 -c \"import json,sys; print('\\n'.join("
            "sorted(m['id'] for m in json.load(sys.stdin)['data'])))\"")

    if args.start_pod and not args.dry_run:
        if not pod_action("start"):
            raise SystemExit("could not start the pod; aborting before spending "
                             "anything on generation")
        if not wait_for_pod():
            log("pod started but never answered ssh; stopping it again")
            pod_action("stop")
            raise SystemExit("pod did not become reachable")

    deadline = time.time() + args.max_hours * 3600
    best = ledger.champion(ledger.FULL_CASES, dtype=args.dtype)
    best_score = best["score"] if best else 0.0
    flat = 0
    log(f"start model={args.model} profile={args.profile} "
        f"incumbent={best['candidate'] if best else 'none'} score={best_score}")

    stop_reason = "completed all iterations"
    done = 0
    decision = None
    for i in range(1, args.max_iters + 1):
        done = i - 1
        if time.time() > deadline:
            stop_reason = "wall-clock budget reached"; break
        if flat >= args.stop_after_flat:
            stop_reason = (f"{flat} consecutive non-improvements -- the search "
                           f"has stalled and needs a structural idea"); break

        # A dead pod must not be recorded as a failed candidate. Dry runs never
        # touch the pod, so they must not require one.
        alive = None if args.dry_run else subprocess.run([str(ROOT / "tools" / "podrun"), "echo ALIVE"],
                               capture_output=True, text=True, cwd=ROOT)
        if alive is not None and "ALIVE" not in alive.stdout:
            stop_reason = "pod unreachable (community cloud reclaim?)"; break

        decision = routing.decide(cycle=i, dtype=args.dtype)
        if decision.mode == "STOP":
            stop_reason = decision.reason; break
        profile = args.profile if args.profile != "auto" else decision.profile
        t_iter = time.time()
        elapsed_h = (time.time() - (deadline - args.max_hours * 3600)) / 3600
        log(f"--- iteration {i}/{args.max_iters} --- route={decision.mode} "
            f"profile={profile} parent={decision.parent} "
            f"beam={decision.beam_size} flat={flat} best={best_score:.4f} "
            f"elapsed={elapsed_h:.1f}h ({decision.reason})")
        if args.verbose and decision.guidance:
            log(f"  guidance: {decision.guidance[:300]}")
        log("  asking the model for a candidate...")
        try:
            reply = call_model(build_prompt(decision), args.model)
            code = extract_code(reply)
        except Exception as e:
            log(f"generation failed: {type(e).__name__}: {e}"); flat += 1; continue

        problems = screen(code)
        if problems:
            log(f"REJECTED before execution: {'; '.join(problems)}")
            flat += 1
            continue

        name = f"v{next_version():03d}_{slug(code)}.py"
        path = ROOT / "candidates" / name
        path.write_text(code)
        headline = next((l.strip().strip('"') for l in code.splitlines()
                         if l.strip().strip('"')), "")
        log(f"wrote {name} ({len(code.splitlines())} lines)")
        if args.verbose:
            log(f"  hypothesis: {headline[:200]}")

        if args.dry_run:
            log("dry-run: skipping evaluation"); continue

        log(f"evaluating {name} on profile={profile} (timeout {args.eval_timeout//60}m)")
        t_eval = time.time()
        try:
            rc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "race.py"), f"candidates/{name}",
                 "--profile", profile, "--dtype", args.dtype], cwd=ROOT,
                timeout=args.eval_timeout).returncode
        except subprocess.TimeoutExpired:
            # A hung evaluation used to stall the whole run indefinitely; the
            # ssh transport can block forever if the pod dies mid-command.
            log(f"evaluation of {name} exceeded {args.eval_timeout}s — abandoning "
                f"this candidate and continuing")
            flat += 1
            continue
        log(f"evaluation finished in {time.time() - t_eval:.0f}s (rc={rc})")
        if rc:
            log(f"race.py exited {rc}"); flat += 1; continue

        now = ledger.champion(ledger.FULL_CASES, dtype=args.dtype)
        score = now["score"] if now else 0.0
        if score > best_score + 1e-9:
            log(f"IMPROVED {best_score:.4f} -> {score:.4f} ({now['candidate']})")
            best_score, flat = score, 0
            outcome = f"{name} promoted, {score:.4f}"
        else:
            flat += 1
            log(f"no improvement (champion still {score:.4f}), flat={flat}")
            outcome = f"{name} rejected, champion {score:.4f}"
        log(f"iteration {i} took {time.time() - t_iter:.0f}s")
        if args.push_results:
            push_results(f"iteration {i} — {outcome}")

    log(f"stop: {stop_reason}")
    write_status(stop_reason, best_score, done, decision)
    if args.push_results and not args.dry_run:
        push_results(stop_reason)
    if args.stop_pod and not args.dry_run:
        pod_action("stop")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:      # includes KeyboardInterrupt
        # An unattended run that dies must not leave the GPU billing.
        if "--dry-run" not in sys.argv:
            log(f"aborting on {type(exc).__name__}")
            if "--push-results" in sys.argv:
                push_results(f"aborted on {type(exc).__name__}")
            if "--stop-pod" in sys.argv:
                pod_action("stop")
        raise
