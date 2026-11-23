"""
src/metrics.py
评测指标: Accuracy, Macro-F1, ECE, Brier, NLL, AUROC, 风险-覆盖率
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)


# ============================================================
# 基础分类指标
# ============================================================
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average='macro', zero_division=0))


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    recalls = []
    for c in range(num_classes):
        mask = (y_true == c)
        if mask.sum() > 0:
            recalls.append(((y_pred[mask] == c)).mean())
        else:
            recalls.append(0.0)
    return np.array(recalls)


# ============================================================
# 校准指标
# ============================================================
def expected_calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE: 把置信度分到 n_bins 个等宽区间，加权平均 |acc - conf|
    
    Args:
        prob:    (N, K) 预测概率
        y:       (N,)   真值标签
        n_bins:  区间数
    """
    conf = prob.max(axis=1).astype(np.float64)
    pred = prob.argmax(axis=1)
    acc = (pred == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() > 0:
            w = mask.sum() / len(y)
            ece += w * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    """Brier 分数: (p - onehot(y))^2 的均值"""
    K = prob.shape[1]
    y_onehot = np.eye(K)[y]
    return float(((prob - y_onehot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(prob: np.ndarray, y: np.ndarray) -> float:
    """NLL: -log p(y)"""
    p_correct = prob[np.arange(len(y)), y].clip(1e-10, 1.0)
    return float(-np.log(p_correct).mean())


# ============================================================
# 错误检测 AUROC (用不确定性 u 检测错分样本)
# ============================================================
def error_detection_auroc(u: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    AUROC: 分数=u (越大越可能错), 标签=1[y_pred != y_true]
    """
    error_label = (y_pred != y_true).astype(int)
    if error_label.sum() == 0 or error_label.sum() == len(error_label):
        return 0.5  # 退化情况
    return float(roc_auc_score(error_label, u))


# ============================================================
# 风险-覆盖率曲线 (Risk-Coverage)
# ============================================================
def risk_coverage_curve(prob: np.ndarray, y: np.ndarray) -> tuple:
    """
    按置信度降序，取前 c% 样本，计算该子集的错误率
    
    Returns:
        coverages: (C,) 覆盖率列表
        risks:     (C,) 对应风险
    """
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y)

    # 按置信度升序排列 (最不确定的排前面)
    order = np.argsort(conf)
    correct_sorted = correct[order]

    N = len(y)
    coverages = np.linspace(0.1, 1.0, 20)
    risks = []
    for c in coverages:
        n = int(c * N)
        if n == 0:
            risks.append(0.0)
        else:
            risks.append(1.0 - correct_sorted[:n].mean())
    return coverages, risks


def area_under_risk_coverage(prob: np.ndarray, y: np.ndarray) -> float:
    coverages, risks = risk_coverage_curve(prob, y)
    # 梯形积分 (NumPy 2.x 用 trapezoid, 旧版用 trapz)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(risks, coverages))


# ============================================================
# 汇总所有指标
# ============================================================
def compute_all_metrics(prob: np.ndarray, u: np.ndarray, y_true: np.ndarray) -> dict:
    """
    一次调用计算全部指标
    
    Args:
        prob:    (N, K) 期望概率
        u:       (N,)   不确定性
        y_true:  (N,)   真值标签
    """
    y_pred = prob.argmax(axis=1)
    K = prob.shape[1]

    return {
        "Acc":        accuracy(y_true, y_pred),
        "Macro-F1":   macro_f1(y_true, y_pred),
        "ECE":        expected_calibration_error(prob, y_true),
        "Brier":      brier_score(prob, y_true),
        "NLL":        negative_log_likelihood(prob, y_true),
        "ErrDet-AUROC": error_detection_auroc(u, y_true, y_pred),
        "AURC":       area_under_risk_coverage(prob, y_true),
    }
