from sklearn.metrics import f1_score, accuracy_score


def get_f1_score(y_true, y_pred, average, **kwargs):
    return f1_score(y_true=y_true, y_pred=y_pred, average=average, **kwargs)


def get_accuracy(y_true, y_pred, **kwargs):
    return accuracy_score(y_true=y_true, y_pred=y_pred, **kwargs)
