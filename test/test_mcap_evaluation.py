import json

from tools.evaluate_pv26_mcap import evaluate


def test_red_signal_report_keeps_detection_and_wrong_state_separate(tmp_path):
    identity = {"bag": str(tmp_path / "drive.mcap"), "camera": "left",
                "bag_time_ns": 100, "topic_sequence": 0}
    prediction = {**identity, "observation": {"image_hw": [600, 800],
        "roadmarks": [], "detections": [{
        "class_name": "vehicle_signal", "score": 0.9,
        "bbox_xyxy": [10, 10, 20, 30],
        "state": {"state_valid": True, "base_color": "green", "left_arrow": False},
    }]}}
    ground_truth = {**identity, "roadmarks": [], "signals": [{
        "class_name": "vehicle_signal", "bbox_xyxy": [10, 10, 20, 30],
        "state_valid": True, "base_color": "red", "left_arrow": False,
    }]}
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.jsonl"
    predictions.write_text(json.dumps(prediction) + "\n")
    labels.write_text(json.dumps(ground_truth) + "\n")
    result = evaluate(predictions, labels, iou_threshold=0.5, line_tolerance_px=8.0)
    assert result["signal_detection_total"]["f1"] == 1.0
    assert result["state"]["vehicle_signal"]["red"]["fn"] == 1
    assert result["state"]["vehicle_signal"]["green"]["fp"] == 1
    assert result["state_context"]["red_as_green"] == 1
