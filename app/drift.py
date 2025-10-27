import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def detect_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """
    Detect data drift using Eviently
    returns : (drift_score: Float)  
    """
    try:
        report = Report(metrics=[DataDriftPreset()])
        my_eval = report.run(reference_data=reference_data, current_data=current_data)
        result = my_eval.dict()

        drift_result = result["metrics"][0].get("result", {})
        drift_score = drift_result.get("share_of_drifted_columns", 0.0)
        drift_detected = drift_result.get("dataset_drift", False)

        # Save report if drift is detected
        if drift_detected:
            report.save_html("artifacts/drift_report.html")
            print("⚠️ Drift detected! Report saved to artifacts/drift_report.html")

        return drift_detected, drift_score

    
    except Exception as e:
        print(f"⚠️ Drift detection failed: {e}")
        return False, 0.0