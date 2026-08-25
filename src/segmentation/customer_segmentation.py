"""
CustomerSegmentationEngine — 客户分群引擎.

基于 Purchase Probability (购买概率)、Churn Probability (流失概率) 和
Predicted LTV (预测生命周期价值) 三个维度，将客户划分为四个分群：

    VIP      — 高价值忠诚客户（高购买意愿 + 低流失 + 高LTV）
    Potential — 潜力客户（中等购买意愿 + 低/中流失 + 中/高LTV）
    At-Risk  — 流失风险客户（低购买意愿 + 高流失）
    Dormant  — 沉睡客户（极低购买意愿 + 极高流失 + 低LTV）

架构：
    客户特征 → [SegmentationEngine] → 分群标签 + 解释

规则优先级（从高到低）：
    1. VIP      — PP >= 0.7 且 CP < 0.3 且 LTV >= ltv_high
    2. Potential — PP >= 0.5 且 CP < 0.5 且 LTV >= ltv_medium
                   或 PP >= 0.6 且 CP >= 0.5（高购买意愿但有流失风险）
    3. At-Risk  — PP < 0.4 且 CP >= 0.5
    4. Dormant  — PP < 0.2 且 CP >= 0.7
                   或 LTV < ltv_low 且 PP < 0.3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEGMENT_VIP = "VIP"
SEGMENT_POTENTIAL = "Potential"
SEGMENT_AT_RISK = "At-Risk"
SEGMENT_DORMANT = "Dormant"


@dataclass
class SegmentationResult:
    """Single customer segmentation result.

    Attributes
    ----------
    segment : str
        Assigned segment label (VIP / Potential / At-Risk / Dormant).
    probability : float
        Purchase probability used for segmentation.
    churn_probability : float
        Churn probability used for segmentation.
    ltv : float
        Predicted LTV used for segmentation.
    reason : str
        Human-readable explanation for the assigned segment.
    """

    segment: str
    probability: float
    churn_probability: float
    ltv: float
    reason: str


class CustomerSegmentationEngine:
    """Rule-based customer segmentation engine.

    Parameters
    ----------
    vip_prob_threshold : float
        Minimum purchase probability for VIP segment (default 0.7).
    vip_churn_threshold : float
        Maximum churn probability for VIP segment (default 0.3).
    potential_prob_threshold : float
        Minimum purchase probability for Potential segment (default 0.5).
    potential_churn_threshold : float
        Maximum churn probability for Potential segment (default 0.5).
    at_risk_prob_threshold : float
        Maximum purchase probability for At-Risk segment (default 0.4).
    at_risk_churn_threshold : float
        Minimum churn probability for At-Risk segment (default 0.5).
    dormant_prob_threshold : float
        Maximum purchase probability for Dormant segment (default 0.2).
    dormant_churn_threshold : float
        Minimum churn probability for Dormant segment (default 0.7).
    ltv_percentile_low : float
        Percentile for low LTV threshold (default 0.2 = 20th percentile).
    ltv_percentile_medium : float
        Percentile for medium LTV threshold (default 0.5 = 50th percentile).
    ltv_percentile_high : float
        Percentile for high LTV threshold (default 0.8 = 80th percentile).
    ltv_low_override : float or None
        Fixed low LTV threshold (overrides percentile-based if set).
    ltv_medium_override : float or None
        Fixed medium LTV threshold.
    ltv_high_override : float or None
        Fixed high LTV threshold.
    """

    def __init__(
        self,
        vip_prob_threshold: float = 0.7,
        vip_churn_threshold: float = 0.3,
        potential_prob_threshold: float = 0.5,
        potential_churn_threshold: float = 0.5,
        at_risk_prob_threshold: float = 0.4,
        at_risk_churn_threshold: float = 0.5,
        dormant_prob_threshold: float = 0.2,
        dormant_churn_threshold: float = 0.7,
        ltv_percentile_low: float = 0.2,
        ltv_percentile_medium: float = 0.5,
        ltv_percentile_high: float = 0.8,
        ltv_low_override: Optional[float] = None,
        ltv_medium_override: Optional[float] = None,
        ltv_high_override: Optional[float] = None,
    ) -> None:
        self.vip_prob_threshold = vip_prob_threshold
        self.vip_churn_threshold = vip_churn_threshold
        self.potential_prob_threshold = potential_prob_threshold
        self.potential_churn_threshold = potential_churn_threshold
        self.at_risk_prob_threshold = at_risk_prob_threshold
        self.at_risk_churn_threshold = at_risk_churn_threshold
        self.dormant_prob_threshold = dormant_prob_threshold
        self.dormant_churn_threshold = dormant_churn_threshold

        self.ltv_percentile_low = ltv_percentile_low
        self.ltv_percentile_medium = ltv_percentile_medium
        self.ltv_percentile_high = ltv_percentile_high

        self.ltv_low_override = ltv_low_override
        self.ltv_medium_override = ltv_medium_override
        self.ltv_high_override = ltv_high_override

        self._ltv_low: Optional[float] = None
        self._ltv_medium: Optional[float] = None
        self._ltv_high: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "CustomerSegmentationEngine":
        """Learn LTV thresholds from customer data.

        Computes percentile-based LTV thresholds. Override values
        take precedence when set.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a ``predicted_ltv`` column.

        Returns
        -------
        self : CustomerSegmentationEngine
        """
        if "predicted_ltv" not in df.columns:
            raise ValueError("Column 'predicted_ltv' not found in input DataFrame.")

        ltv_series = df["predicted_ltv"].dropna()

        if self.ltv_low_override is not None:
            self._ltv_low = self.ltv_low_override
        else:
            self._ltv_low = float(ltv_series.quantile(self.ltv_percentile_low))

        if self.ltv_medium_override is not None:
            self._ltv_medium = self.ltv_medium_override
        else:
            self._ltv_medium = float(ltv_series.quantile(self.ltv_percentile_medium))

        if self.ltv_high_override is not None:
            self._ltv_high = self.ltv_high_override
        else:
            self._ltv_high = float(ltv_series.quantile(self.ltv_percentile_high))

        logger.info(
            "LTV thresholds learned — low=%.2f, medium=%.2f, high=%.2f",
            self._ltv_low,
            self._ltv_medium,
            self._ltv_high,
        )
        return self

    def segment(
        self,
        probability: float,
        churn_probability: float,
        ltv: float,
    ) -> SegmentationResult:
        """Segment a single customer.

        Parameters
        ----------
        probability : float
            Purchase probability [0, 1].
        churn_probability : float
            Churn probability [0, 1].
        ltv : float
            Predicted lifetime value.

        Returns
        -------
        SegmentationResult
        """
        self._ensure_fitted()

        prob = float(probability)
        churn = float(churn_probability)
        ltv_val = float(ltv)

        if prob >= self.vip_prob_threshold and churn < self.vip_churn_threshold and ltv_val >= self._ltv_high:
            return SegmentationResult(
                segment=SEGMENT_VIP,
                probability=prob,
                churn_probability=churn,
                ltv=ltv_val,
                reason=(
                    f"High purchase probability ({prob:.2f} >= {self.vip_prob_threshold}), "
                    f"low churn risk ({churn:.2f} < {self.vip_churn_threshold}), "
                    f"high LTV ({ltv_val:.2f} >= {self._ltv_high:.2f})"
                ),
            )

        if (prob >= self.potential_prob_threshold and churn < self.potential_churn_threshold and ltv_val >= self._ltv_medium) or \
           (prob >= 0.6 and churn >= self.at_risk_churn_threshold):
            if prob >= self.potential_prob_threshold and churn < self.potential_churn_threshold and ltv_val >= self._ltv_medium:
                reason = (
                    f"Moderate-high purchase probability ({prob:.2f} >= {self.potential_prob_threshold}), "
                    f"manageable churn risk ({churn:.2f} < {self.potential_churn_threshold}), "
                    f"moderate-high LTV ({ltv_val:.2f} >= {self._ltv_medium:.2f})"
                )
            else:
                reason = (
                    f"High purchase intent ({prob:.2f} >= 0.6) "
                    f"but elevated churn risk ({churn:.2f} >= {self.at_risk_churn_threshold}) — "
                    f"retention opportunity"
                )
            return SegmentationResult(
                segment=SEGMENT_POTENTIAL,
                probability=prob,
                churn_probability=churn,
                ltv=ltv_val,
                reason=reason,
            )

        if prob < self.at_risk_prob_threshold and churn >= self.at_risk_churn_threshold:
            return SegmentationResult(
                segment=SEGMENT_AT_RISK,
                probability=prob,
                churn_probability=churn,
                ltv=ltv_val,
                reason=(
                    f"Low purchase probability ({prob:.2f} < {self.at_risk_prob_threshold}), "
                    f"high churn risk ({churn:.2f} >= {self.at_risk_churn_threshold})"
                ),
            )

        if prob < self.dormant_prob_threshold and churn >= self.dormant_churn_threshold:
            return SegmentationResult(
                segment=SEGMENT_DORMANT,
                probability=prob,
                churn_probability=churn,
                ltv=ltv_val,
                reason=(
                    f"Very low purchase probability ({prob:.2f} < {self.dormant_prob_threshold}), "
                    f"very high churn risk ({churn:.2f} >= {self.dormant_churn_threshold})"
                ),
            )

        if ltv_val < self._ltv_low and prob < self.at_risk_prob_threshold:
            return SegmentationResult(
                segment=SEGMENT_DORMANT,
                probability=prob,
                churn_probability=churn,
                ltv=ltv_val,
                reason=(
                    f"Very low LTV ({ltv_val:.2f} < {self._ltv_low:.2f}) "
                    f"combined with low purchase probability ({prob:.2f} < {self.at_risk_prob_threshold})"
                ),
            )

        return SegmentationResult(
            segment=SEGMENT_POTENTIAL,
            probability=prob,
            churn_probability=churn,
            ltv=ltv_val,
            reason=(
                f"Borderline metrics — purchase={prob:.2f}, churn={churn:.2f}, LTV={ltv_val:.2f}. "
                f"Defaulted to Potential for nurturing."
            ),
        )

    def segment_batch(
        self,
        df: pd.DataFrame,
        probability_col: str = "purchase_probability",
        churn_col: str = "churn_probability",
        ltv_col: str = "predicted_ltv",
        return_details: bool = False,
    ) -> pd.DataFrame:
        """Segment multiple customers in batch.

        Parameters
        ----------
        df : pd.DataFrame
            Customer-level DataFrame.
        probability_col : str
            Column name for purchase probability.
        churn_col : str
            Column name for churn probability.
        ltv_col : str
            Column name for predicted LTV.
        return_details : bool
            If True, returns a DataFrame with reason and all input columns.
            If False, returns only customer_id and segment.

        Returns
        -------
        pd.DataFrame
        """
        self._ensure_fitted()

        for col in [probability_col, churn_col, ltv_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")

        segments: List[str] = []
        reasons: List[str] = []

        for _, row in df.iterrows():
            result = self.segment(
                probability=row[probability_col],
                churn_probability=row[churn_col],
                ltv=row[ltv_col],
            )
            segments.append(result.segment)
            reasons.append(result.reason)

        result_df = df.copy()
        result_df["segment"] = segments

        if return_details:
            result_df["segment_reason"] = reasons
            return result_df

        if "customer_id" in result_df.columns:
            return result_df[["customer_id", "segment"]]
        return result_df[["segment"]]

    def get_segment_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Return segment distribution as percentages.

        Parameters
        ----------
        df : pd.DataFrame
            Customer-level DataFrame.

        Returns
        -------
        dict
            Segment name → percentage (0-100).
        """
        result_df = self.segment_batch(df)
        counts = result_df["segment"].value_counts()
        total = len(result_df)
        distribution = {
            seg: float(counts.get(seg, 0)) / total * 100.0
            for seg in [SEGMENT_VIP, SEGMENT_POTENTIAL, SEGMENT_AT_RISK, SEGMENT_DORMANT]
        }
        return distribution

    def get_thresholds(self) -> Dict[str, float]:
        """Return current threshold values."""
        return {
            "vip_prob": self.vip_prob_threshold,
            "vip_churn": self.vip_churn_threshold,
            "potential_prob": self.potential_prob_threshold,
            "potential_churn": self.potential_churn_threshold,
            "at_risk_prob": self.at_risk_prob_threshold,
            "at_risk_churn": self.at_risk_churn_threshold,
            "dormant_prob": self.dormant_prob_threshold,
            "dormant_churn": self.dormant_churn_threshold,
            "ltv_low": self._ltv_low,
            "ltv_medium": self._ltv_medium,
            "ltv_high": self._ltv_high,
        }

    def summary(self) -> str:
        """Return human-readable summary of segmentation rules."""
        lines = [
            "Customer Segmentation Engine",
            "=" * 60,
            f"  VIP:       PP >= {self.vip_prob_threshold} AND CP < {self.vip_churn_threshold} AND LTV >= {self._ltv_high:.2f}",
            f"  Potential: PP >= {self.potential_prob_threshold} AND CP < {self.potential_churn_threshold} AND LTV >= {self._ltv_medium:.2f}",
            f"             OR PP >= 0.6 AND CP >= {self.at_risk_churn_threshold}",
            f"  At-Risk:   PP < {self.at_risk_prob_threshold} AND CP >= {self.at_risk_churn_threshold}",
            f"  Dormant:   PP < {self.dormant_prob_threshold} AND CP >= {self.dormant_churn_threshold}",
            f"             OR LTV < {self._ltv_low:.2f} AND PP < {self.at_risk_prob_threshold}",
            "-" * 60,
            f"  LTV thresholds: low={self._ltv_low:.2f}, medium={self._ltv_medium:.2f}, high={self._ltv_high:.2f}",
        ]
        return "\n".join(lines)

    def _ensure_fitted(self) -> None:
        if self._ltv_low is None or self._ltv_medium is None or self._ltv_high is None:
            raise RuntimeError(
                "CustomerSegmentationEngine not fitted. Call fit() first to learn LTV thresholds, "
                "or set ltv_low_override / ltv_medium_override / ltv_high_override explicitly."
            )

    def __repr__(self) -> str:
        return (
            f"CustomerSegmentationEngine("
            f"vip_pp>={self.vip_prob_threshold}, "
            f"vip_cp<{self.vip_churn_threshold}, "
            f"potential_pp>={self.potential_prob_threshold}, "
            f"at_risk_pp<{self.at_risk_prob_threshold}, "
            f"dormant_pp<{self.dormant_prob_threshold})"
        )


def create_default_segmentation_engine(
    ltv_low: Optional[float] = None,
    ltv_medium: Optional[float] = None,
    ltv_high: Optional[float] = None,
) -> CustomerSegmentationEngine:
    """Create a segmentation engine with default thresholds.

    Parameters
    ----------
    ltv_low : float or None
        Fixed low LTV threshold. If None, uses 20th percentile.
    ltv_medium : float or None
        Fixed medium LTV threshold. If None, uses 50th percentile.
    ltv_high : float or None
        Fixed high LTV threshold. If None, uses 80th percentile.

    Returns
    -------
    CustomerSegmentationEngine
    """
    return CustomerSegmentationEngine(
        vip_prob_threshold=0.7,
        vip_churn_threshold=0.3,
        potential_prob_threshold=0.5,
        potential_churn_threshold=0.5,
        at_risk_prob_threshold=0.4,
        at_risk_churn_threshold=0.5,
        dormant_prob_threshold=0.2,
        dormant_churn_threshold=0.7,
        ltv_percentile_low=0.2,
        ltv_percentile_medium=0.5,
        ltv_percentile_high=0.8,
        ltv_low_override=ltv_low,
        ltv_medium_override=ltv_medium,
        ltv_high_override=ltv_high,
    )