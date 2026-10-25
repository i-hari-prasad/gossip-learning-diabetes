class ClinicalDecisionSupport:
    """
    Maps predicted glucose values to clinical zones.

    Clinical thresholds (configurable):
        Hypoglycemia : glucose < low_threshold   (default 70 mg/dL)
        Normal       : low_threshold <= glucose <= high_threshold
        Hyperglycemia: glucose > high_threshold  (default 180 mg/dL)

    Note: Thresholds exposed as configuration — not silently assumed.
    """

    ZONES = ('Hypoglycemia', 'Normal', 'Hyperglycemia')

    def __init__(self, low_threshold: float = 70.0, high_threshold: float = 180.0):
        self.low = low_threshold
        self.high = high_threshold

    def classify(self, glucose_value: float) -> str:
        if glucose_value < self.low:
            return 'Hypoglycemia'
        elif glucose_value <= self.high:
            return 'Normal'
        return 'Hyperglycemia'

    def analyze_predictions(self, y_true, y_pred) -> dict:
        true_classes = [self.classify(v) for v in y_true]
        pred_classes = [self.classify(v) for v in y_pred]
        n = len(y_true)
        correct = sum(t == p for t, p in zip(true_classes, pred_classes))
        hypo_missed = sum(
            1 for t, p in zip(true_classes, pred_classes)
            if t == 'Hypoglycemia' and p != 'Hypoglycemia'
        )
        return {
            'clinical_accuracy_pct': correct / n * 100 if n > 0 else 0.0,
            'hypo_missed': hypo_missed,
            'n_samples': n
        }
