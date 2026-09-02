"""Success Model：使用 Logistic Regression 预测成功率"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .models import HistoricalData

SKLEARN_AVAILABLE = find_spec("sklearn") is not None


class SuccessClassifier:
    """全局成功率估计模型"""

    CAPABILITIES: ClassVar[tuple[str, ...]] = (
        "retrieval",
        "understanding",
        "planning",
        "editing",
        "diagnosis",
        "verification",
    )
    EFFORTS: ClassVar[tuple[str, ...]] = ("low", "medium", "high")

    def __init__(self) -> None:
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for SuccessClassifier; install nju-coding-agent[adaptive]"
            )

        linear_model = import_module("sklearn.linear_model")
        preprocessing = import_module("sklearn.preprocessing")
        self.encoder: Any = preprocessing.OneHotEncoder(
            categories=[self.EFFORTS] * len(self.CAPABILITIES),
            sparse_output=False,
        )
        self.model: Any = linear_model.LogisticRegression(max_iter=1000)
        self.is_fitted = False

    def fit(self, historical_data: HistoricalData) -> None:
        """使用全部历史数据训练"""
        if len(historical_data.samples) < 10:
            return  # 数据太少，不训练

        # 准备特征矩阵 X
        features: list[list[str]] = []
        outcomes: list[int] = []

        for sample in historical_data.samples:
            # 将 start_effort_vector 转换为 ordered list
            effort_list = [sample.start_effort_vector.get(c, "medium") for c in self.CAPABILITIES]
            features.append(effort_list)
            outcomes.append(int(sample.outcome))

        # Logistic regression needs both successful and failed observations.
        if len(set(outcomes)) < 2:
            return

        # One-hot 编码
        encoded = self.encoder.fit_transform(features)

        # 训练模型
        self.model.fit(encoded, outcomes)
        self.is_fitted = True

    def predict_performance(self, start_effort_vector: dict[str, str]) -> float:
        """预测成功概率"""
        if not self.is_fitted:
            return 0.5  # 未训练，返回默认值

        effort_list = [[start_effort_vector.get(c, "medium") for c in self.CAPABILITIES]]

        X_encoded = self.encoder.transform(effort_list)
        prob = self.model.predict_proba(X_encoded)[0][1]
        return float(prob)

    def counterfactual_evaluation(
        self,
        base_policy: dict[str, str],
        target_capability: str,
    ) -> dict[str, float]:
        """Counterfactual 评估：固定其他，变化目标 Capability"""

        if not self.is_fitted:
            # 未训练，返回默认估计
            return {"low": 0.5, "medium": 0.5, "high": 0.5}

        results = {}
        for effort in self.EFFORTS:
            config = base_policy.copy()
            config[target_capability] = effort
            prob = self.predict_performance(config)
            results[effort] = prob

        return results
