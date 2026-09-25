"""Public JSON API for the eating detection project.

The service uses only Python's standard library.  It exposes a small local
HTTP API so the Huawei export fields can be integrated without importing the
internal detector modules directly.

Endpoints
----------
GET  /health
POST /v1/detect
POST /v1/train
POST /v1/evaluate
"""
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from .detect_nondominant import detect_nondominant_files
    from .evaluate_real_data import evaluate as evaluate_real_data
    from .train_model import train as train_model
except ImportError:
    from detect_nondominant import detect_nondominant_files
    from evaluate_real_data import evaluate as evaluate_real_data
    from train_model import train as train_model


API_VERSION = "1.0"


class ApiError(ValueError):
    """A client-facing request validation error."""


def _required(body: dict[str, Any], key: str) -> Any:
    value = body.get(key)
    if value is None or value == "":
        raise ApiError(f"missing required field: {key}")
    return value


def _paths(value: Any, field: str = "sensorData") -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return value
    raise ApiError(f"{field} must be a path string or a non-empty list of path strings")


def _hour_range(value: Any) -> tuple[float, float] | None:
    if value is None:
        return (5.0, 24.0)
    if isinstance(value, str):
        if value.lower() == "none":
            return None
        try:
            low, high = (float(part.strip()) for part in value.split(",", 1))
            return low, high
        except (TypeError, ValueError) as exc:
            raise ApiError("hourRange must be 'start,end' or 'none'") from exc
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError) as exc:
            raise ApiError("hourRange values must be numbers") from exc
    raise ApiError("hourRange must be 'start,end', [start, end], or 'none'")


def _event_payload(event: tuple[float, float], file_name: str, t0: float) -> dict[str, Any]:
    start, end = float(event[0]), float(event[1])
    return {
        "file": file_name,
        "start": start,
        "end": end,
        "duration_s": max(0.0, end - start),
        "relative_start": start - float(t0),
        "relative_end": end - float(t0),
    }


def detect(body: dict[str, Any]) -> dict[str, Any]:
    """Run inference from one ZIP, multiple ZIPs, or a ZIP directory."""
    sensor_data = body.get("sensorData", body.get("zip"))
    paths = _paths(_required({"sensorData": sensor_data}, "sensorData"))
    model = _required(body, "model")
    kwargs: dict[str, Any] = {
        "stitch": bool(body.get("stitch", body.get("stitchFiles", False))),
        "model": model,
        "min_duration_s": float(body.get("minDurationS", body.get("min-duration", 300.0))),
        "hour_range": _hour_range(body.get("hourRange", body.get("hour-range"))),
    }
    if body.get("threshold") is not None:
        kwargs["threshold"] = float(body["threshold"])
    results = detect_nondominant_files(paths, **kwargs)
    events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    files: list[str] = []
    for result in results:
        file_name = str(result.get("file", ""))
        t0 = float(result.get("t0", 0.0))
        files.extend(str(item) for item in result.get("files", []) or [])
        if not result.get("files") and file_name:
            files.append(file_name)
        events.extend(_event_payload(event, file_name, t0) for event in result.get("events", []))
        errors.extend(result.get("errors", []) or [])
    return {
        "ok": not errors,
        "api_version": API_VERSION,
        "count": len(events),
        "events": events,
        "files": list(dict.fromkeys(files)),
        "error_count": len(errors),
        "errors": errors,
    }


def train(body: dict[str, Any]) -> dict[str, Any]:
    """Train a model from sensor ZIPs and the attachment's CSV fields."""
    zip_dir = _required({"zipDir": body.get("zipDir", body.get("sensorDataDir"))}, "zipDir")
    meals = _required({"mealinfo": body.get("mealinfo", body.get("mealInfo"))}, "mealinfo")
    mapping = _required({"mapping": body.get("mapping", body.get("sensorMapping"))}, "mapping")
    output = body.get("model", body.get("output", "meal_random_forest.joblib"))
    result = train_model(zip_dir, meals, mapping, output,
                         random_state=int(body.get("randomState", 42)))
    return {"ok": True, "api_version": API_VERSION, **result}


def evaluate(body: dict[str, Any]) -> dict[str, Any]:
    """Evaluate predictions against beforeTime/afterTime meal annotations."""
    result = evaluate_real_data(
        _required({"zipDir": body.get("zipDir", body.get("sensorDataDir"))}, "zipDir"),
        _required({"mealinfo": body.get("mealinfo", body.get("mealInfo"))}, "mealinfo"),
        participant=body.get("externalid", body.get("participant")),
        mapping_csv=body.get("mapping", body.get("sensorMapping")),
        iou_threshold=float(body.get("iouThreshold", 0.25)),
        min_duration_s=float(body.get("minDurationS", 300.0)),
        hour_range=_hour_range(body.get("hourRange")),
        model_path=_required(body, "model"),
        plot_path=body.get("plot"),
        stitch_files=bool(body.get("stitch", body.get("stitchFiles", False))),
    )
    result["ok"] = result.get("error_count", 0) == 0
    result["api_version"] = API_VERSION
    return result


class _Handler(BaseHTTPRequestHandler):
    server_version = "EatingDetectionAPI/1.0"

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self._write(HTTPStatus.NOT_FOUND, {"ok": False, "error": "route not found"})
            return
        self._write(HTTPStatus.OK, {"ok": True, "api_version": API_VERSION, "service": "eating-detection"})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        routes = {"/v1/detect": detect, "/v1/train": train, "/v1/evaluate": evaluate}
        operation = routes.get(self.path)
        if operation is None:
            self._write(HTTPStatus.NOT_FOUND, {"ok": False, "error": "route not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ApiError("request body must be a JSON object")
            self._write(HTTPStatus.OK, operation(body))
        except ApiError as exc:
            self._write(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            self._write(HTTPStatus.UNPROCESSABLE_ENTITY, {"ok": False, "error": str(exc)})
        except Exception as exc:  # keep the service JSON-shaped for callers
            self._write(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": repr(exc)})

    def log_message(self, format, *args):
        return


def create_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """Create, but do not start, the public API server."""
    return ThreadingHTTPServer((host, int(port)), _Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the eating detection JSON API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"Eating detection API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
