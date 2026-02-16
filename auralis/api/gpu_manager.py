"""AURALIS GPU Manager — auto start/stop GPU instances via boto3.

Architecture:
    t3.small (always on) → manages → g5.xlarge (on-demand)

    The t3.small runs the web UI and API. When heavy processing is needed
    (stem separation, mastering, ML inference), the GPU manager auto-starts
    the g5.xlarge, sends the job via SSH, collects results, and auto-stops
    the GPU instance after an idle timeout.

Cost:
    - t3.small: ~$15/mo (always on)
    - g5.xlarge: ~$1.00/hr (on-demand, only when processing)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger()

# ── Configuration ───────────────────────────────────────────

GPU_INSTANCE_ID = os.environ.get("AURALIS_GPU_INSTANCE_ID", "")
GPU_REGION = os.environ.get("AURALIS_GPU_REGION", "us-east-1")
GPU_IDLE_TIMEOUT_SECONDS = int(
    os.environ.get("AURALIS_GPU_IDLE_TIMEOUT", "600")
)  # 10 min default


@dataclass
class GPUState:
    """Tracks GPU instance state and activity."""

    status: str = "unknown"  # stopped, pending, running, stopping
    instance_id: str = ""
    public_ip: str | None = None
    last_activity: datetime | None = None
    auto_stop_task: asyncio.Task[None] | None = field(
        default=None, repr=False
    )

    def touch(self) -> None:
        """Record activity to prevent auto-stop."""
        self.last_activity = datetime.now(UTC)


_state = GPUState(instance_id=GPU_INSTANCE_ID)


# ── Core Functions ──────────────────────────────────────────


def _get_ec2_client():
    """Get boto3 EC2 client."""
    try:
        import boto3  # type: ignore[import-untyped]
    except ImportError:
        msg = "boto3 not installed. Run: uv pip install boto3"
        raise RuntimeError(msg) from None
    return boto3.client("ec2", region_name=GPU_REGION)


async def gpu_status() -> dict[str, object]:
    """Get current GPU instance status."""
    if not _state.instance_id:
        return {
            "status": "not_configured",
            "message": "No GPU instance configured. Set AURALIS_GPU_INSTANCE_ID.",
        }

    try:
        ec2 = _get_ec2_client()
        response = await asyncio.to_thread(
            ec2.describe_instances, InstanceIds=[_state.instance_id]
        )
        instance = response["Reservations"][0]["Instances"][0]
        _state.status = instance["State"]["Name"]
        _state.public_ip = instance.get("PublicIpAddress")

        return {
            "status": _state.status,
            "instance_id": _state.instance_id,
            "instance_type": instance.get("InstanceType", "unknown"),
            "public_ip": _state.public_ip,
            "last_activity": (
                _state.last_activity.isoformat()
                if _state.last_activity
                else None
            ),
            "idle_timeout_seconds": GPU_IDLE_TIMEOUT_SECONDS,
        }
    except Exception as e:
        logger.error("gpu_status_error", error=str(e))
        return {"status": "error", "message": str(e)}


async def start_gpu() -> dict[str, object]:
    """Start the GPU instance."""
    if not _state.instance_id:
        return {"error": "No GPU instance configured"}

    current = await gpu_status()
    if current.get("status") == "running":
        return {"status": "already_running", **current}

    try:
        ec2 = _get_ec2_client()
        await asyncio.to_thread(
            ec2.start_instances, InstanceIds=[_state.instance_id]
        )
        _state.status = "pending"
        _state.touch()

        logger.info("gpu_starting", instance_id=_state.instance_id)

        # Wait for running state
        waiter = ec2.get_waiter("instance_running")
        await asyncio.to_thread(
            waiter.wait,
            InstanceIds=[_state.instance_id],
            WaiterConfig={"Delay": 5, "MaxAttempts": 60},
        )

        # Refresh status to get public IP
        result = await gpu_status()

        # Start auto-stop timer
        _schedule_auto_stop()

        return {"status": "started", **result}
    except Exception as e:
        logger.error("gpu_start_error", error=str(e))
        return {"status": "error", "message": str(e)}


async def stop_gpu() -> dict[str, str]:
    """Stop the GPU instance."""
    if not _state.instance_id:
        return {"error": "No GPU instance configured"}

    try:
        ec2 = _get_ec2_client()
        await asyncio.to_thread(
            ec2.stop_instances, InstanceIds=[_state.instance_id]
        )
        _state.status = "stopping"
        _state.public_ip = None

        # Cancel auto-stop if running
        if _state.auto_stop_task and not _state.auto_stop_task.done():
            _state.auto_stop_task.cancel()

        logger.info("gpu_stopping", instance_id=_state.instance_id)
        return {"status": "stopping", "instance_id": _state.instance_id}
    except Exception as e:
        logger.error("gpu_stop_error", error=str(e))
        return {"status": "error", "message": str(e)}


# ── Auto-Stop Logic ────────────────────────────────────────


def _schedule_auto_stop() -> None:
    """Schedule auto-stop after idle timeout."""
    if _state.auto_stop_task and not _state.auto_stop_task.done():
        _state.auto_stop_task.cancel()

    _state.auto_stop_task = asyncio.create_task(_auto_stop_loop())


async def _auto_stop_loop() -> None:
    """Background loop that stops GPU after idle timeout."""
    try:
        while True:
            await asyncio.sleep(60)  # Check every minute
            if _state.last_activity is None:
                continue

            elapsed = (
                datetime.now(UTC) - _state.last_activity
            ).total_seconds()
            if elapsed >= GPU_IDLE_TIMEOUT_SECONDS:
                logger.info(
                    "gpu_auto_stop",
                    idle_seconds=elapsed,
                    timeout=GPU_IDLE_TIMEOUT_SECONDS,
                )
                await stop_gpu()
                break
    except asyncio.CancelledError:
        pass


async def ensure_gpu_running() -> str | None:
    """Ensure GPU is running. Start it if needed. Returns public IP.

    Call this before sending jobs to the GPU instance.
    Returns None if GPU is not configured or failed to start.
    """
    if not _state.instance_id:
        return None

    status = await gpu_status()
    if status.get("status") != "running":
        result = await start_gpu()
        if result.get("status") == "error":
            return None

    _state.touch()
    return _state.public_ip
