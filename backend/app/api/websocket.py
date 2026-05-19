from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import random
from datetime import datetime, UTC

router = APIRouter()


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()

    print("WebSocket connected")

    try:
        while True:

            # Generate random threat score
            score = round(random.uniform(1, 100), 2)

            # Dynamic severity based on score
            if score >= 80:
                severity = "high"
            elif score >= 50:
                severity = "medium"
            else:
                severity = "low"

            # Send live alert
            await websocket.send_json({
                "event": "threat",
                "timestamp": datetime.now(UTC).isoformat(),
                "source_ip": random.choice([
                    "192.168.1.10",
                    "10.0.0.5",
                    "172.16.0.12",
                    "203.45.67.89"
                ]),
                "request_method": random.choice([
                    "GET",
                    "POST",
                    "PUT"
                ]),
                "request_path": random.choice([
                    "/admin/login",
                    "/api/auth",
                    "/dashboard",
                    "/config",
                    "/user/profile",
                    "/database"
                ]),
                "threat_score": score,
                "severity": severity,
                "component_scores": {
                    "rules": round(random.uniform(40, 95), 2),
                    "ml": round(random.uniform(40, 95), 2)
                }
            })

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("WebSocket error:", e)