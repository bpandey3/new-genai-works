@registry.endpoint("POST", "/notable/getLLMResponse")
def get_llm_response(req: Request):
    """Endpoint 1: must exist as per contract"""
    result = main_method(req.body or {})
    return {"result": result, "source": "getLLMResponse"}

@registry.endpoint("GET", "/notable/getUserDetails")
def get_user_details(req: Request):
    """Endpoint 2: must exist as per contract"""
    user_id = req.query.get("user_id", "unknown")
    return {"user_id": user_id, "status": "active"}
