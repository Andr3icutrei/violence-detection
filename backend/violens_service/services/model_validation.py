from dataclasses import dataclass


@dataclass
class ConfusionMatrixCounts:
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(self, is_violent_gt: bool, is_violent_pred: bool) -> None:
        if is_violent_gt and is_violent_pred:
            self.true_positive += 1
        elif not is_violent_gt and not is_violent_pred:
            self.true_negative += 1
        elif not is_violent_gt and is_violent_pred:
            self.false_positive += 1
        elif is_violent_gt and not is_violent_pred:
            self.false_negative += 1

    def accuracy(self) -> float:
        total = self.true_positive + self.true_negative + self.false_positive + self.false_negative
        return (self.true_positive + self.true_negative) / total if total > 0 else 0.0

