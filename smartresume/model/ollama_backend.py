"""
Ollama backend: the local LLM served by Ollama (llama.cpp) instead of transformers.

Why it exists: on a laptop GPU the transformers + bitsandbytes path spends ~170 ms on
every generated token whatever the batch size (thousands of small kernel launches
per step), so one resume takes about a minute. The same Gemma weights as a 4-bit
GGUF generate several times faster, and the prompts of a resume run side by side
in the server's parallel slots.

The server is talked to over its HTTP API (/api/chat, streamed), so nothing but
`requests` is needed in this process: no torch model, no 5.6 GB embedding table in RAM.

Parallel slots are a *server* setting (OLLAMA_NUM_PARALLEL), and the Ollama tray app
runs with one slot. So this backend does not use the tray app's server (port 11434):
it starts its own `ollama serve` on port 11435 with the slots it needs and stops it
when Python exits. Both servers share the downloaded models. Point `host` at a
server you manage yourself to skip this; that server's own slot count then applies.

Measured on an RTX 4050 Laptop (6 GB), Gemma 4 E4B 4-bit: 39 tokens/s for one
stream, ~170 tokens/s summed over 8 parallel streams.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    ## Not Ollama's default port (11434): see the module docstring
    "host": "http://127.0.0.1:11435",
    ## 4-bit, quantization-aware-trained Gemma 4 E4B
    "model": "gemma4:e4b-it-qat",
    ## Prompts generated at the same time (server slots when the server is started here)
    "parallel": 8,
    ## Context window per slot, in tokens (prompt + answer). VRAM grows with parallel x num_ctx
    ## (about 0.9 GB at 8 x 4096).
    "num_ctx": 4096,
    ## Model layers on the GPU. 99 = all. Left to itself Ollama keeps ~2 GB of VRAM free
    ## for the image projector (unused here) and runs the last layers on the CPU on a
    ## 6 GB card, which costs a quarter of the speed. -1 = let Ollama decide.
    "num_gpu": 99,
    ## How long the server keeps the model in VRAM after the last request
    "keep_alive": "30m",
    ## Constrain the answer to the section's JSON shape (grammar-guided decoding)
    "structured_output": True,
    "auto_start": True,
    "auto_pull": True,
    "request_timeout": 600,
}

## Rough chars-per-token lower bound for English resume text with "[12]:" indexes;
## used only to keep a prompt inside num_ctx (the server would cut it silently)
_CHARS_PER_TOKEN = 2.8


def find_executable() -> Optional[str]:
    """Path of the ollama program, or None when it is not installed."""
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Ollama", "ollama.exe"),
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def pull_model(options: Optional[Dict[str, Any]] = None) -> str:
    """Download the configured model into Ollama (setup step); returns its tag."""
    backend = OllamaBackend(options)
    try:
        if not backend._alive():
            backend._start_server()
        backend._ensure_model()
    finally:
        backend.close()
    return backend.model


def _json_schema_for(template: Any) -> Any:
    """
    JSON schema with the shape of a prompt TEMPLATE ('' -> string, 0 -> integer,
    [] -> list of strings, or of integers for *_index_range keys).
    """
    if isinstance(template, dict):
        properties = {}
        for key, value in template.items():
            if isinstance(value, list) and not value:
                item_type = "integer" if key.endswith("index_range") else "string"
                properties[key] = {"type": "array", "items": {"type": item_type}}
            else:
                properties[key] = _json_schema_for(value)
        return {"type": "object", "properties": properties, "required": list(template)}
    if isinstance(template, list):
        return {"type": "array", "items": _json_schema_for(template[0]) if template else {"type": "string"}}
    if isinstance(template, bool):
        return {"type": "boolean"}
    if isinstance(template, int):
        return {"type": "integer"}
    return {"type": "string"}


def schema_from_template(template_json: str) -> Optional[Dict[str, Any]]:
    """The `format` schema for one section prompt; None if the template is not JSON."""
    try:
        return _json_schema_for(json.loads(template_json))
    except (json.JSONDecodeError, TypeError):
        return None


class OllamaBackend:
    """Chat completions from a local Ollama server, several at a time."""

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        settings = {**DEFAULTS, **(options or {})}
        self.host = str(os.environ.get("RESUME_OLLAMA_HOST") or settings["host"]).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.model = str(os.environ.get("RESUME_OLLAMA_MODEL") or settings["model"])
        self.parallel = max(1, int(settings["parallel"]))
        self.num_ctx = int(settings["num_ctx"])
        self.num_gpu = int(settings["num_gpu"])
        self.keep_alive = settings["keep_alive"]
        self.structured_output = bool(settings["structured_output"])
        self.auto_start = bool(settings["auto_start"])
        self.auto_pull = bool(settings["auto_pull"])
        self.timeout = float(settings["request_timeout"])
        self.session = requests.Session()
        ## Running totals for the log: prompt tokens sent, and how many the server
        ## already had cached (the shared system prompt of a slot's previous request)
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0
        self._stats_lock = threading.Lock()
        ## The server process, when it was started here (stopped again in close())
        self._server: Optional[subprocess.Popen] = None

    # -- Start-up -- #
    def start(self) -> None:
        """Make sure the server answers, the model is downloaded, and it is loaded."""
        if not self._alive():
            self._start_server()
        self._ensure_model()
        started = time.perf_counter()
        ## An empty request loads the model, so the first resume does not pay for it
        self._post("/api/generate", {"model": self.model, "keep_alive": self.keep_alive,
                                     "options": self._load_options()})
        logger.info("Ollama model %s loaded in %.1fs (%s, %d parallel, num_ctx %d)%s",
                    self.model, time.perf_counter() - started, self.host, self.parallel,
                    self.num_ctx, self._placement())

    def _load_options(self) -> Dict[str, Any]:
        """Options that decide how the model is loaded; the same in every request, or it reloads."""
        options: Dict[str, Any] = {"num_ctx": self.num_ctx}
        if self.num_gpu >= 0:
            options["num_gpu"] = self.num_gpu
        return options

    def _alive(self, host: Optional[str] = None) -> bool:
        try:
            return self.session.get((host or self.host) + "/api/version", timeout=3).ok
        except requests.RequestException:
            return False

    def _free_default_server(self) -> None:
        """Ask the tray app's server to drop this model, so it is not in VRAM twice."""
        default = "http://127.0.0.1:11434"
        if self.host.endswith(":11434") or not self._alive(default):
            return
        try:
            loaded = [m.get("name") for m in self.session.get(default + "/api/ps", timeout=5).json().get("models", [])]
            if self.model in loaded:
                self.session.post(default + "/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
        except (requests.RequestException, ValueError):
            pass

    def _start_server(self) -> None:
        executable = find_executable()
        if not self.auto_start or not executable:
            raise RuntimeError(
                f"No Ollama server answers at {self.host}. Install Ollama from https://ollama.com/download "
                "(or start it), or set llm_backend: transformers in configs/config.yaml.")
        environment = dict(os.environ)
        environment["OLLAMA_HOST"] = self.host.split("://", 1)[1]
        environment.setdefault("OLLAMA_NUM_PARALLEL", str(self.parallel))
        environment.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._free_default_server()
        self._server = subprocess.Popen([executable, "serve"], env=environment, creationflags=flags,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._alive():
                logger.info("Started Ollama server at %s (%d parallel slots)", self.host, self.parallel)
                return
            time.sleep(0.3)
        raise RuntimeError(f"Ollama was started but does not answer at {self.host}")

    def _ensure_model(self) -> None:
        names = [m.get("name", "") for m in self.session.get(self.host + "/api/tags", timeout=10).json().get("models", [])]
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if wanted in names:
            return
        if not self.auto_pull:
            raise RuntimeError(f"Ollama model '{self.model}' is not downloaded. Run: ollama pull {self.model}")
        logger.info("Downloading Ollama model %s (one time, several GB)...", self.model)
        logged = None
        with self.session.post(self.host + "/api/pull", json={"model": self.model}, stream=True, timeout=None) as reply:
            reply.raise_for_status()
            for line in reply.iter_lines():
                if not line:
                    continue
                status = json.loads(line)
                if status.get("error"):
                    raise RuntimeError(f"ollama pull {self.model} failed: {status['error']}")
                if status.get("total") and status.get("completed") is not None:
                    ## One log line per layer and 10%
                    step = (status.get("digest"), status["completed"] * 10 // status["total"])
                    if step != logged and status["total"] > 100 * 1024 ** 2:
                        logged = step
                        logger.info("  %.1f GB file: %d%%", status["total"] / 1024 ** 3, step[1] * 10)

    def _placement(self) -> str:
        """', 100% on GPU' or how much of the model spilled to system RAM."""
        try:
            for model in self.session.get(self.host + "/api/ps", timeout=5).json().get("models", []):
                size, vram = model.get("size") or 0, model.get("size_vram") or 0
                if model.get("name") == self.model and size:
                    share = 100 * vram // size
                    if share < 100:
                        logger.warning(
                            "Only %d%% of %s fits in VRAM (%.1f of %.1f GB); the rest runs on the CPU and "
                            "tokens are slower. Lower ollama.parallel or ollama.num_ctx, or close other GPU apps.",
                            share, self.model, vram / 1024 ** 3, size / 1024 ** 3)
                    return f", {share}% on GPU ({size / 1024 ** 3:.1f} GB)"
        except (requests.RequestException, ValueError):
            pass
        return ""

    def close(self) -> None:
        """Stop the server started here, with the model runner it spawned (frees the VRAM)."""
        server, self._server = self._server, None
        if server is None or server.poll() is not None:
            return
        if os.name == "nt":
            ## terminate() would leave the runner child process holding the model
            taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
            subprocess.run([taskkill, "/T", "/F", "/PID", str(server.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            server.terminate()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        reply = self.session.post(self.host + path, json=payload, timeout=self.timeout)
        reply.raise_for_status()
        return reply.json()

    # -- Generation -- #
    def fit_text(self, system_prompt: str, text: str, max_new_tokens: int) -> str:
        """Shorten resume text that would not fit in num_ctx next to its prompt and answer."""
        room = int((self.num_ctx - max_new_tokens - 64) * _CHARS_PER_TOKEN) - len(system_prompt)
        if len(text) <= room:
            return text
        keep = max(600, room)
        logger.warning("Prompt longer than num_ctx=%d; resume text cut from %d to %d chars.",
                       self.num_ctx, len(text), keep)
        return text[:keep]

    def chat(self, system_prompt: str, user_text: str, max_new_tokens: int,
             schema: Optional[Dict[str, Any]] = None,
             should_stop: Optional[Callable[[List[str]], bool]] = None) -> Tuple[str, str, int]:
        """
        One greedy completion, streamed so a looping answer can be cut early.

        should_stop gets the list of text pieces generated so far (about one per
        token) every few pieces; returning True ends the answer with reason "loop".

        Returns (text, stop_reason, new_tokens), stop_reason as in the transformers
        path: "eos", "loop" or "max_tokens".
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": True,
            ## No reasoning block: the token budget goes to the JSON
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                ## Greedy: the same resume gives the same answer on every run
                "temperature": 0,
                "seed": 0,
                "num_predict": int(max_new_tokens),
                **self._load_options(),
            },
        }
        if self.structured_output and schema:
            payload["format"] = schema

        pieces: List[str] = []
        stop_reason, new_tokens = "eos", 0
        with self.session.post(self.host + "/api/chat", json=payload, stream=True, timeout=self.timeout) as reply:
            if not reply.ok:
                raise RuntimeError(f"Ollama error {reply.status_code}: {reply.text[:300]}")
            for line in reply.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                piece = (chunk.get("message") or {}).get("content") or ""
                if piece:
                    pieces.append(piece)
                if chunk.get("done"):
                    new_tokens = int(chunk.get("eval_count") or len(pieces))
                    with self._stats_lock:
                        self.prompt_tokens += int(chunk.get("prompt_eval_count") or 0)
                        self.cached_prompt_tokens += int(chunk.get("prompt_eval_cached_count") or 0)
                    if chunk.get("done_reason") == "length":
                        stop_reason = "max_tokens"
                    break
                if should_stop and len(pieces) >= 30 and len(pieces) % 8 == 0 and should_stop(pieces):
                    ## Leaving the `with` block closes the connection, which stops the generation
                    stop_reason, new_tokens = "loop", len(pieces)
                    break
        return "".join(pieces), stop_reason, new_tokens

    def run(self, calls: List[Callable[[], Any]]) -> List[Any]:
        """Run the given chat calls, `parallel` at a time; an exception is returned in place."""
        def guarded(call: Callable[[], Any]) -> Any:
            try:
                return call()
            except Exception as error:  # reported per prompt by the caller
                return error

        if self.parallel == 1 or len(calls) < 2:
            return [guarded(call) for call in calls]
        with ThreadPoolExecutor(max_workers=self.parallel, thread_name_prefix="ollama") as pool:
            return list(pool.map(guarded, calls))
