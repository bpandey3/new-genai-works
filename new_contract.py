# contract.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import json, base64

@dataclass
class Request:
    http_method: str          # "GET"/"POST"/...
    path: str                 # "/notable/getGPTResponse"
    headers: Dict[str, str]
    query: Dict[str, Any]
    body: Any                 # already JSON-decoded when possible
    raw_event: Dict[str, Any]
    context: Any

def _parse_body(body: Optional[str], is_b64: Optional[bool]):
    if body is None: return None
    if is_b64:
        try: body = base64.b64decode(body).decode("utf-8", errors="ignore")
        except Exception: return None
    try:
        return json.loads(body)
    except Exception:
        return body

def _extract_http(event: Dict[str, Any]):
    rc = event.get("requestContext") or {}
    http = rc.get("http")
    if http:  # API Gateway v2 / Lambda Function URL
        method = (http.get("method") or "GET").upper()
        path = (event.get("rawPath") or event.get("path") or "/")
        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        query = event.get("queryStringParameters") or {}
        body = _parse_body(event.get("body"), event.get("isBase64Encoded"))
        return method, path, headers, query, body
    # API Gateway v1
    method = (event.get("httpMethod") or "GET").upper()
    path = (event.get("path") or "/")
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    query = event.get("queryStringParameters") or {}
    body = _parse_body(event.get("body"), event.get("isBase64Encoded"))
    return method, path, headers, query, body

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

class Contract:
    """
    Usage in a notebook:
        from contract import Contract, Request
        api = Contract()
        api.set_endpoints("/notable/getGPTResponse", "/notable/getBedrockResonse", method="POST")

        @api.endpoint("POST", "/notable/getGPTResponse")
        def gpt(req: Request): ...

        @api.endpoint("POST", "/notable/getBedrockResonse")
        def bedrock(req: Request): ...

        api.validate()   # raises if missing
    """
    def __init__(self):
        self._required: set[Tuple[str, str]] = set()          # {(METHOD, PATH), ...}
        self._routes: Dict[Tuple[str, str], Callable[[Request], Any]] = {}

    # --- declare required endpoints (authors call this in the notebook) ---
    def set_endpoints(self, *paths: str, method: str = "POST"):
        """Declare required endpoints sharing one HTTP method (default POST)."""
        m = method.upper()
        for p in paths:
            self._required.add((m, self._norm_path(p)))

    def add_required(self, method: str, path: str):
        """Add one required endpoint with its method."""
        self._required.add((method.upper(), self._norm_path(path)))

    def set_required(self, endpoints: Iterable[Tuple[str, str]]):
        """Replace the required set entirely ([(method, path), ...])."""
        self._required = {(m.upper(), self._norm_path(p)) for m, p in endpoints}

    # --- register implementations via decorator ---
    def endpoint(self, method: str, path: str):
        m = method.upper(); p = self._norm_path(path)
        def deco(fn: Callable[[Request], Any]):
            self._routes[(m, p)] = fn
            return fn
        return deco

    # --- validation authors run inside the notebook ---
    def validate(self, raise_on_error: bool = True) -> List[Tuple[str, str]]:
        missing = [(m, p) for (m, p) in sorted(self._required) if (m, p) not in self._routes]
        if missing and raise_on_error:
            msg = "Missing required endpoints: " + ", ".join([f"{m} {p}" for m, p in missing])
            raise RuntimeError(msg)
        return missing

    # --- optional: simple dispatcher (handy for local testing or Lambda use) ---
    def dispatch(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        http_method, path, headers, query, body = _extract_http(event)
        if http_method == "OPTIONS":
            return http_response(200, "")
        key = (http_method, self._norm_path(path))
        fn = self._routes.get(key)
        if not fn:
            return http_response(404, {"error": "Not Found", "method": http_method, "path": path})
        try:
            req = Request(http_method, path, headers, query, body, event, context)
            result = fn(req)
            return http_response(200, result)
        except Exception:
            return http_response(500, {"error": "Internal Server Error"})

    @staticmethod
    def _norm_path(p: str) -> str:
        p = p or "/"
        p = "/" + p.lstrip("/")
        return p.rstrip("/") or "/"



#######usage########

# Cell 1: imports + create contract
from contract import Contract, Request
api = Contract()

# Cell 2: declare REQUIRED endpoints for THIS notebook (authors set them here)
api.set_endpoints("/notable/getGPTResponse", "/notable/getBedrockResonse", method="POST")
# If you want mixed methods, you can do:
# api.add_required("GET",  "/notable/getUserDetails")
# api.add_required("POST", "/notable/getGPTResponse")

# Cell 3: implement the endpoints using decorators (names can be anything)
@api.endpoint("POST", "/notable/getGPTResponse")
def gpt(req: Request):
    prompt = (req.body or {}).get("prompt", "")
    # ... call OpenAI/Anthropic/etc ...
    return {"model": "GPT", "answer": f"Processed: {prompt}"}

@api.endpoint("POST", "/notable/getBedrockResonse")
def bedrock(req: Request):
    prompt = (req.body or {}).get("prompt", "")
    # ... call Bedrock provider ...
    return {"model": "Bedrock", "answer": f"Processed: {prompt}"}

# Cell 4: validate INSIDE THE NOTEBOOK (hard fail here if missing)
api.validate()           # raises RuntimeError if either endpoint isn't implemented
print("Contract OK ✅")


######test cases#######

fake_gpt = {
    "requestContext": {"http": {"method": "POST"}},
    "rawPath": "/notable/getGPTResponse",
    "body": '{"prompt":"hello"}',
    "isBase64Encoded": False
}
resp = api.dispatch(fake_gpt)    # 200 with your handler's result
resp

#######lambda#########
# lambda_handler.py
from notable import main as nb   # the converted notebook module; must expose `api`
def main_handler(event, context):
    # no validate() call here — you chose to enforce in the notebook step
    return nb.api.dispatch(event, context)
