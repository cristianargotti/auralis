"""WebSocket handler for real-time progress updates.

Clients connect to /ws/{project_id} and receive live updates
as analysis/rendering/mastering jobs progress.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = structlog.get_logger()


class ConnectionManager:
    """Manages WebSocket connections per project."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, project_id: str, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if project_id not in self._connections:
            self._connections[project_id] = []
        self._connections[project_id].append(websocket)
        logger.info("WebSocket connected", project_id=project_id)

    def disconnect(self, project_id: str, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket."""
        if project_id in self._connections:
            self._connections[project_id] = [
                ws for ws in self._connections[project_id] if ws != websocket
            ]
            if not self._connections[project_id]:
                del self._connections[project_id]
        logger.info("WebSocket disconnected", project_id=project_id)

    async def send_progress(
        self,
        project_id: str,
        step: int,
        total: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Send progress update to all connections for a project."""
        payload = json.dumps(
            {
                "type": "progress",
                "step": step,
                "total": total,
                "percentage": round(step / total * 100) if total > 0 else 0,
                "message": message,
                "data": data or {},
            }
        )

        if project_id in self._connections:
            dead: list[WebSocket] = []
            for ws in self._connections[project_id]:
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(payload)
                except Exception:
                    dead.append(ws)

            for ws in dead:
                self.disconnect(project_id, ws)

    async def send_complete(
        self,
        project_id: str,
        result: dict[str, Any],
    ) -> None:
        """Send completion message with final result."""
        payload = json.dumps(
            {
                "type": "complete",
                "result": result,
            }
        )

        if project_id in self._connections:
            for ws in list(self._connections.get(project_id, [])):
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(payload)
                except Exception:
                    self.disconnect(project_id, ws)

    async def send_error(
        self,
        project_id: str,
        error: str,
    ) -> None:
        """Send error message."""
        payload = json.dumps(
            {
                "type": "error",
                "error": error,
            }
        )

        if project_id in self._connections:
            for ws in list(self._connections.get(project_id, [])):
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_text(payload)
                except Exception:
                    self.disconnect(project_id, ws)


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, project_id: str) -> None:
    """WebSocket endpoint for real-time project updates.

    Usage from frontend:
        const ws = new WebSocket('ws://host:8000/ws/project-id')
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            // data.type: 'progress' | 'complete' | 'error'
            // data.percentage: 0-100
            // data.message: human-readable status
        }
    """
    await manager.connect(project_id, websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            # Client can send commands like "cancel"
            msg = json.loads(data)
            if msg.get("action") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(project_id, websocket)
