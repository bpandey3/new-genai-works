# contract.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import base64, json

REQUIRED_ENDPOINTS = {
    ("POST", "/notable/getLLMResponse"),
    ("GET",  "/notable/getUserDetails"),
}
REQUIRED_CALLABLES = ("main_method",)

@dataclass
class Request:
    method: str
    path: str
    headers: Dict[str, str]
    query: Dict[str, Any]
    body: Any
    raw_event: Dict[str, Any]
    context: Any

def http_response(status: int, body: Any, cors: bool = True) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if cors:
        headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,PATCH,OPTIONS",
        })
    body_str = "" if body is None else (json.dumps(body) if isinstance(body, (dict, list)) else str(body))
    return {"statusCode": status, "headers": headers, "body": body_str, "isBase64Encoded": False}

class Registry:
    def __init__(self):
        self._routes: Dict[Tuple[str, str], Callable[[Request], Any]] = {}

    def endpoint(self, method: str, path: str):
        m = method.upper(); p = (path.rstrip("/") or "/")
        def deco(fn: Callable[[Request], Any]):
            self._routes[(m, p)] = fn
            return fn
        return deco

    def validate_endpoints(self, required = REQUIRED_ENDPOINTS):
        missing = [(m, p) for (m, p) in required if (m, p) not in self._routes]
        if missing:
            raise RuntimeError("Missing required endpoints: " + ", ".join(f"{m} {p}" for m, p in missing))

    def handle(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        rc = (event.get("requestContext") or {}); http = rc.get("http")
        if http:  # API GW v2
            method = (http.get("method") or "GET").upper()
            path = (event.get("rawPath") or event.get("path") or "/").rstrip("/") or "/"
            headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
            query = event.get("queryStringParameters") or {}
            body = _parse_body(event.get("body"), event.get("isBase64Encoded"))
        else:     # v1
            method = (event.get("httpMethod") or "GET").upper()
            path = (event.get("path") or "/").rstrip("/") or "/"
            headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
            query = event.get("queryStringParameters") or {}
            body = _parse_body(event.get("body"), event.get("isBase64Encoded"))

        if method == "OPTIONS":
            return http_response(200, "")

        handler = self._routes.get((method, path))
        if not handler:
            return http_response(404, {"error": "Not Found", "method": method, "path": path})
        try:
            req = Request(method, path, headers, query, body, event, context)
            result = handler(req)
            return http_response(200, result)
        except Exception:
            return http_response(500, {"error": "Internal Server Error"})

def _parse_body(body: Optional[str], is_b64: Optional[bool]):
    if body is None: return None
    if is_b64:
        try:
            import base64
            body = base64.b64decode(body).decode("utf-8", errors="ignore")
        except Exception:
            return None
    try:
        import json
        return json.loads(body)
    except Exception:
        return body  # allow plain text

def validate_module(mod,
                    required_endpoints = REQUIRED_ENDPOINTS,
                    required_callables  = REQUIRED_CALLABLES):
    # module must expose a Registry named `registry`
    if not hasattr(mod, "registry"):
        raise RuntimeError("Module must export `registry` (contract.Registry)")
    mod.registry.validate_endpoints(required_endpoints)
    # required callables (e.g., main_method)
    missing = [name for name in required_callables if not callable(getattr(mod, name, None))]
    if missing:
        raise RuntimeError("Missing required callables: " + ", ".join(missing))
