"""
RuleEngine — 前置规则引擎层.

真实风控系统的第一道防线：在 ML 模型打分之前，先用
确定性规则拦截明显欺诈和高风险交易。

架构：
    交易 → [规则引擎] → 命中 → 直接 block/challenge（不走模型）
                      → 未命中 → 交给 ML 模型打分

规则类型：
    - BlacklistRule:  黑名单匹配（卡号/IP/邮箱域名）
    - VelocityRule:   速度规则（同一 card 在 N 秒内交易次数超限）
    - AmountRule:     硬阈值（单笔金额超限）

生产环境中，黑名单和速度计数器应使用 Redis：
    - 黑名单: Redis SET，O(1) 查询
    - 速度: Redis SORTED SET（score=timestamp），ZRANGEBYSCORE 统计窗口内数量
本实现用内存数据结构（单进程 demo），接口设计兼容 Redis 替换。
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """单条规则的评估结果.

    Attributes
    ----------
    action : str
        "block" — 直接拦截，不进模型
        "challenge" — 触发 step-up 验证（短信/3D Secure）
        "pass" — 未命中，交给下一道规则或模型
    matched_rule : str | None
        命中的规则名（如 "blacklist", "velocity_3600s"）
    reason : str | None
        人类可读的命中原因
    """

    action: str = "pass"
    matched_rule: Optional[str] = None
    reason: Optional[str] = None


class RuleBase(ABC):
    """规则抽象基类."""

    name: str = "BaseRule"

    @abstractmethod
    def check(self, transaction: Dict[str, Any]) -> RuleResult:
        """评估单笔交易，返回 RuleResult."""
        ...


class BlacklistRule(RuleBase):
    """黑名单匹配规则.

    检查指定字段（如 card1、IP、邮箱域名）是否在黑名单中。
    命中 → block。

    生产环境: Redis SET 存储，SISMEMBER O(1) 查询。
    """

    name = "blacklist"

    def __init__(
        self,
        field: str = "card1",
        blacklist: Optional[Set[Any]] = None,
    ) -> None:
        self.field = field
        self._blacklist: Set[str] = {str(v) for v in (blacklist or set())}
        self._lock = threading.Lock()

    def add(self, value: Any) -> None:
        with self._lock:
            self._blacklist.add(str(value))

    def remove(self, value: Any) -> None:
        with self._lock:
            self._blacklist.discard(str(value))

    def check(self, transaction: Dict[str, Any]) -> RuleResult:
        val = transaction.get(self.field)
        if val is None:
            return RuleResult()
        if str(val) in self._blacklist:
            return RuleResult(
                action="block",
                matched_rule=f"blacklist_{self.field}",
                reason=f"{self.field}={val} is blacklisted",
            )
        return RuleResult()


class VelocityRule(RuleBase):
    """速度规则: 同一 group_col 在 window 秒内交易次数超限.

    命中 → challenge（触发 step-up 验证）。

    生产环境: Redis SORTED SET（member=tx_id, score=timestamp），
    ZRANGEBYSCORE 统计窗口内数量，ZREMRANGEBYSCORE 清理过期。
    """

    name = "velocity"

    def __init__(
        self,
        group_col: str = "card1",
        window_seconds: float = 3600.0,
        max_count: int = 5,
    ) -> None:
        self.group_col = group_col
        self.window = window_seconds
        self.max_count = max_count
        self._history: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, transaction: Dict[str, Any]) -> RuleResult:
        key = str(transaction.get(self.group_col, ""))
        if not key or key in ("None", "nan", ""):
            return RuleResult()

        now = int(transaction.get("TransactionDT", 0))

        with self._lock:
            dq = self._history[key]
            while dq and dq[0] < now - self.window:
                dq.popleft()
            dq.append(now)
            count = len(dq)

        if count > self.max_count:
            return RuleResult(
                action="challenge",
                matched_rule=f"velocity_{self.group_col}_{int(self.window)}s",
                reason=f"{count} txs by {self.group_col}={key} in {int(self.window)}s (limit={self.max_count})",
            )
        return RuleResult()


class AmountRule(RuleBase):
    """硬阈值规则: 单笔金额超限 → challenge."""

    name = "amount_threshold"

    def __init__(self, threshold: float = 10000.0) -> None:
        self.threshold = threshold

    def check(self, transaction: Dict[str, Any]) -> RuleResult:
        amt = transaction.get("TransactionAmt")
        if amt is not None and float(amt) > self.threshold:
            return RuleResult(
                action="challenge",
                matched_rule="amount_threshold",
                reason=f"TransactionAmt={amt} exceeds {self.threshold}",
            )
        return RuleResult()


class RuleEngine:
    """规则引擎: 按顺序执行规则，命中即返回.

    规则未命中 → 交给 ML 模型打分。
    规则命中 block → 直接拦截。
    规则命中 challenge → 触发 step-up 验证。
    """

    def __init__(self, rules: Optional[List[RuleBase]] = None) -> None:
        self.rules: List[RuleBase] = rules or []

    def add_rule(self, rule: RuleBase) -> "RuleEngine":
        self.rules.append(rule)
        return self

    def evaluate(self, transaction: Dict[str, Any]) -> RuleResult:
        """按顺序评估所有规则，返回第一个命中的结果.

        Returns
        -------
        RuleResult
            action="pass" 表示所有规则都未命中，应交给模型打分。
        """
        for rule in self.rules:
            try:
                result = rule.check(transaction)
                if result.action != "pass":
                    logger.info(
                        "Rule hit: %s → %s (%s)",
                        result.matched_rule,
                        result.action,
                        result.reason,
                    )
                    return result
            except Exception as e:
                logger.warning("Rule %s error: %s", rule.name, e)
        return RuleResult(action="pass")

    def list_rules(self) -> List[Dict[str, Any]]:
        """返回所有规则的信息（用于 /rules 端点）."""
        info = []
        for rule in self.rules:
            entry: Dict[str, Any] = {"name": rule.name, "type": type(rule).__name__}
            if isinstance(rule, BlacklistRule):
                entry["field"] = rule.field
                entry["blacklist_size"] = len(rule._blacklist)
            elif isinstance(rule, VelocityRule):
                entry["group_col"] = rule.group_col
                entry["window_seconds"] = rule.window
                entry["max_count"] = rule.max_count
            elif isinstance(rule, AmountRule):
                entry["threshold"] = rule.threshold
            info.append(entry)
        return info


def create_default_engine() -> RuleEngine:
    """创建默认规则引擎实例.

    规则执行顺序（短路评估）：
        1. 黑名单 → block（最高优先级，直接拦截）
        2. 速度规则 → challenge（1 小时内同卡 > 5 笔）
        3. 硬阈值 → challenge（单笔 > $10,000）
    未命中 → 交给 ML 模型
    """
    return RuleEngine(
        rules=[
            BlacklistRule(field="card1"),
            VelocityRule(group_col="card1", window_seconds=3600, max_count=5),
            AmountRule(threshold=10000.0),
        ]
    )
