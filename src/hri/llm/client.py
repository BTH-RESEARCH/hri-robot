"""OpenAI-compatible streaming client with provider fallback (Gemini -> Groq -> ...)."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator

log = logging.getLogger(__name__)

MIN_KEY_LEN = 20   # shorter values are placeholders ("xxxx"), not real keys


def _describe(e: Exception) -> str:
    """Error with the server's message (the SDK hides the body for some status codes)."""
    body = getattr(e, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        err = body.get("error", body)
        msg = err.get("message") if isinstance(err, dict) else None
        if msg:
            return f"{e.__class__.__name__} {getattr(e, 'status_code', '')}: {msg}"
    return str(e)


class LLMClient:
    def __init__(self, llm_cfg: dict):
        from openai import OpenAI

        self.cfg = llm_cfg
        self.providers: list[tuple[str, str, object, dict]] = []
        for p in llm_cfg.get("providers", []):
            key = os.environ.get(p["api_key_env"], "").strip()
            if len(key) < MIN_KEY_LEN:
                log.info("LLM provider %s skipped: %s not set", p["name"], p["api_key_env"])
                continue
            client = OpenAI(api_key=key, base_url=p["base_url"],
                            timeout=float(llm_cfg.get("timeout_sec", 15)), max_retries=0)
            self.providers.append((p["name"], p["model"], client, p.get("extra_body") or {}))
        if not self.providers:
            raise RuntimeError("No LLM provider available - put an API key in .env (see .env.example)")
        log.info("LLM providers: %s", ", ".join(f"{n}({m})" for n, m, _, _ in self.providers))

    def stream(self, messages: list[dict], info: dict | None = None) -> Iterator[str]:
        """Yield text deltas. Falls back to the next provider only if nothing was yielded yet.
        info["finish"] = the server's finish_reason; None means the stream was cut off."""
        info = {} if info is None else info
        errors: list[str] = []
        for name, model, client, extra in self.providers:
            t0 = time.perf_counter()
            first, info["finish"] = True, None
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, stream=True,
                    temperature=float(self.cfg.get("temperature", 0.7)),
                    max_tokens=int(self.cfg.get("max_tokens", 256)),
                    extra_body=extra or None,
                )
                for chunk in resp:
                    if not chunk.choices:
                        continue
                    info["finish"] = chunk.choices[0].finish_reason or info["finish"]
                    delta = (chunk.choices[0].delta.content if chunk.choices[0].delta else None) or ""
                    if delta:
                        if first:
                            log.info("LLM %s TTFT %.0f ms", name, (time.perf_counter() - t0) * 1000)
                            first = False
                        yield delta
                if first:
                    raise RuntimeError(f"empty reply (finish={info['finish']})")
                return
            except Exception as e:
                if not first:
                    raise
                msg = _describe(e)
                log.warning("LLM %s (%s) failed: %s", name, model, msg)
                errors.append(f"{name}: {msg}")
        raise RuntimeError("All LLM providers failed -> " + " | ".join(errors))

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))

    def list_models(self) -> dict[str, list[str]]:
        out = {}
        for name, _, client, _ in self.providers:
            try:
                out[name] = sorted(m.id.removeprefix("models/") for m in client.models.list())
            except Exception as e:  # noqa: BLE001
                out[name] = [f"<error: {_describe(e)}>"]
        return out
