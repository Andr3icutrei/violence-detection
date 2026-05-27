from services.model_validation import ConfusionMatrixCounts


def test_confusion_matrix_update_all_paths():
    counts = ConfusionMatrixCounts()
    counts.update(True, True)
    counts.update(False, False)
    counts.update(False, True)
    counts.update(True, False)
    assert counts.true_positive == 1
    assert counts.true_negative == 1
    assert counts.false_positive == 1
    assert counts.false_negative == 1


def test_confusion_matrix_accuracy_zero_total():
    counts = ConfusionMatrixCounts()
    assert counts.accuracy() == 0.0


def test_confusion_matrix_accuracy_nonzero():
    counts = ConfusionMatrixCounts(true_positive=3, true_negative=1, false_positive=1, false_negative=1)
    assert counts.accuracy() == 4 / 6

