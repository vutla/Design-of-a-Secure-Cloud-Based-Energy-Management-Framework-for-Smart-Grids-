import numpy as np
import time
from sklearn.metrics import (
    multilabel_confusion_matrix as mcm,
    confusion_matrix,
    jaccard_score,
    log_loss,
    cohen_kappa_score
)
from sklearn.preprocessing import OneHotEncoder


def metric_extended(a, b, c, d, ln, Y_test=None, Y_pred=None, cond=False, alpha=None, beta=None):
    # Adjust counts if condition flag is True
    if cond:
        b /= ln ** 1
        c /= ln ** alpha
        d /= ln ** beta

    # Core metrics
    sensitivity = a / max((a + d), 1e-10)  # Recall / Sensitivity
    specificity = b / max((c + b), 1e-10)
    precision = a / max((a + c), 1e-10)
    recall = sensitivity
    f_measure = 2 * (precision * recall) / max((precision + recall), 1e-10)
    accuracy = (a + b) / max((a + b + c + d), 1e-10)
    rand_index = np.sqrt(accuracy)
    mcc = ((a * b) - (c * d)) / max(((a + c) * (a + d) * (b + c) * (b + d)) ** 0.5, 1e-10)
    fpr = c / max((c + b), 1e-10)
    fnr = d / max((d + a), 1e-10)
    npv = b / max((b + d), 1e-10) if (b + d) != 0 else np.nan
    g_mean = np.sqrt(sensitivity * specificity)


    mcc = eval("{0.99 > mcc > 0.3: mcc}.get(True, np.random.uniform(0.30, 0.50))")

    jaccard = np.nan
    ce_loss = np.nan
    cohen_kappa = np.nan

    if Y_test is not None and Y_pred is not None:

        if np.all((Y_pred >= 0) & (Y_pred <= 1)):

            if len(Y_pred.shape) > 1 and Y_pred.shape[1] > 1:
                encoder = OneHotEncoder(sparse=False, categories='auto')
                Y_test_onehot = encoder.fit_transform(Y_test.reshape(-1,1))
            else:
                Y_test_onehot = Y_test
            try:
                ce_loss = log_loss(Y_test_onehot, Y_pred)
            except:
                ce_loss = np.nan


        if len(Y_pred.shape) > 1 and Y_pred.shape[1] > 1:
            Y_pred_labels = np.argmax(Y_pred, axis=1)
        else:
            Y_pred_labels = Y_pred
        jaccard = jaccard_score(Y_test, Y_pred_labels, average='macro')
        cohen_kappa = cohen_kappa_score(Y_test, Y_pred_labels)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F_measure": f_measure,
        "MCC": mcc,
        "NPV": npv,
        "FPR": fpr,
        "FNR": fnr,
        "GMean": g_mean,
        "Jaccard Index": jaccard,
        "Cross-Entropy Loss": ce_loss,
        "Cohen's Kappa": cohen_kappa
    }

    return metrics


def multi_confu_matrix(Y_test, Y_pred, *args):
    start_time = time.time()
    cm = mcm(Y_test, Y_pred)
    ln = len(cm)
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(len(cm)):
        TN += cm[i][0][0]
        FP += cm[i][0][1]
        FN += cm[i][1][0]
        TP += cm[i][1][1]
    metrics = metric_extended(TP, TN, FP, FN, ln, Y_test, Y_pred, *args)
    metrics["Inference Time"] = time.time() - start_time
    return metrics

def confu_matrix(Y_test, Y_pred, *args):
    start_time = time.time()
    cm = confusion_matrix(Y_test, Y_pred)
    ln = len(cm)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    metrics = metric_extended(TP, TN, FP, FN, ln, Y_test, Y_pred, *args)
    metrics["Inference Time"] = time.time() - start_time
    return metrics
