"""
Run reports.

``build_report`` summarises a run (per-object looks, detections, first
detection, median SNR, the final track picture and the event log);
``write_report`` saves it as JSON, or as CSV (one summary table followed by the
event table) when the file name ends in ``.csv``.
"""

import csv
import json
from typing import Any, Dict

import numpy as np


def build_report(engine, scenario_name: str = "") -> Dict[str, Any]:
    looks: Dict[int, list] = {}
    for result in engine.log.detection_history:
        looks.setdefault(result.target_id, []).append(result)
    objects = []
    for target in engine.targets:
        history = looks.get(target.target_id, [])
        detections = [r for r in history if r.is_detected]
        objects.append(
            {
                "id": target.target_id,
                "name": target.name,
                "type": target.target_type,
                "affiliation": target.affiliation,
                "looks": len(history),
                "detections": len(detections),
                "pd_empirical": len(detections) / len(history) if history else None,
                "first_detection_s": detections[0].time if detections else None,
                "first_detection_range_km": detections[0].true_range_m / 1000 if detections else None,
                "median_snr_db": float(np.median([r.snr_db for r in history])) if history else None,
            }
        )
    return {
        "scenario": scenario_name,
        "simulated_time_s": float(engine.current_time),
        "seed": engine.seed,
        "objects": objects,
        "tracks": [
            {"id": int(t.id), "status": t.status.value, "hits": int(t.hits), "x_m": float(t.position[0]), "y_m": float(t.position[1])}
            for t in engine.tracks
        ],
        "events": [event.to_dict() for event in engine.event_log],
    }


def write_report(report: Dict[str, Any], path: str) -> None:
    if not path.lower().endswith(".csv"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"# {report['scenario']}", f"t = {report['simulated_time_s']:.1f} s"])
        if report["objects"]:
            fields = list(report["objects"][0].keys())
            writer.writerow(fields)
            for row in report["objects"]:
                writer.writerow([row[k] for k in fields])
        writer.writerow([])
        writer.writerow(["time_s", "event", "message", "target_id", "track_id"])
        for event in report["events"]:
            writer.writerow([f"{event['time']:.2f}", event["kind"], event["message"], event["target_id"], event["track_id"]])
