"""
src/api/webhook.py
Universal webhook endpoint
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.agent.pipeline import RAGAgent

router = APIRouter()
agent = RAGAgent()


@router.post("/api/webhook/universal")
async def universal_webhook(request: Request):
    content_type = request.headers.get("content-type", "")
    session_id = "default_user"
    message = ""

    if "application/x-www-form-urlencoded" in content_type:
        form_data = await request.form()
        session_id = form_data.get("From", "").replace("whatsapp:", "").strip()
        message = form_data.get("Body", "").strip()

    elif "application/json" in content_type:
        json_data = await request.json()
        if ("message" in json_data and
                isinstance(json_data["message"], dict) and
                "chat" in json_data["message"]):
            session_id = str(json_data["message"]["chat"].get("id", ""))
            message = json_data["message"].get("text", "").strip()

        elif "entry" in json_data and isinstance(json_data["entry"], list):
            entry = json_data["entry"][0]
            if "messaging" in entry:
                msg_event = entry["messaging"][0]
                session_id = msg_event.get("sender", {}).get("id", "")
                message = msg_event.get("message", {}).get("text", "").strip()
        else:
            session_id = json_data.get(
                "session_id",
                json_data.get("user_id", "default_user")
            )
            message = json_data.get(
                "message",
                json_data.get("text", json_data.get("question", ""))
            )

    if not message:
        return JSONResponse(
            content={"error": "No message found"},
            status_code=400
        )

    try:
        answer = agent.ask(message, session_id, uploaded_docs=False)
        return JSONResponse(content={"session_id": session_id, "reply": answer})
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)
