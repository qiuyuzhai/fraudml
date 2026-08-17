"""
EmailFeature — Email domain features .

Fraud often uses disposable / temporary email domains.
This feature extracts signals from P_emaildomain.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import FeatureBase

DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "tempmail.com",
    "throwaway.email", "yopmail.com", "maildrop.cc",
    "sharklasers.com", "guerrillamailblock.com",
    "dispostable.com", "trashmail.com", "temp-mail.org",
    "10minuteemail.com", "tempmailaddress.com",
}

POPULARITY_EMAIL_DOMAINS = {
    "qq.com", "foxmail.com",
    "163.com", "126.com", "yeah.net",
    "sina.com", "sohu.com",
    "139.com", "189.cn", "wo.cn",
    "aliyun.com", "tom.com",
    "gmail.com",
    "outlook.com", "hotmail.com", "live.com",
    "yahoo.com", "yahoo.co.uk", "yahoo.co.jp",
    "icloud.com",
    "mail.ru", "gmx.com"
}


class EmailFeature(FeatureBase):
    """Email-domain derived features.

    **Stateless** — fit() is a no-op.  No parameters learned.

    Features:
        - email_suffix_domain: extracted domain part (e.g. 'gmail.com')
        - is_disposable_email: 1 if domain is a known disposable provider
        - is_trusted_email_domain: 1 if domain is a known popular/mainstream provider
        - is_unknown_email_domain: 1 if non‑empty domain neither disposable nor trusted
        - email_domain_missing: 1 if P_emaildomain is null
    """

    def __init__(self, name: str = "EmailFeature") -> None:
        super().__init__(name=name)

    @property
    def is_stateful(self) -> bool:
        return False

    def fit(self, df: pd.DataFrame) -> "EmailFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")

        df = df.copy()
        col = "P_emaildomain"

        if col not in df.columns:
            df["email_domain_missing"] = 1
            df["is_disposable_email"] = 0
            df["is_trusted_email_domain"] = 0
            df["is_unknown_email_domain"] = 0
            df["email_suffix_domain"] = "unknown"
            return df

        domain = df[col].astype(str).str.strip().str.lower()
        domain = domain.replace({"nan": "", "": ""})

        df["email_domain_missing"] = df[col].isnull().astype(np.int8)
        df["is_disposable_email"] = domain.isin(DISPOSABLE_DOMAINS).astype(np.int8)
        df["is_trusted_email_domain"] = domain.isin(POPULARITY_EMAIL_DOMAINS).astype(np.int8)

        not_disposable = ~domain.isin(DISPOSABLE_DOMAINS)
        not_trusted = ~domain.isin(POPULARITY_EMAIL_DOMAINS)
        domain_not_empty = domain != ""
        df["is_unknown_email_domain"] = (not_disposable & not_trusted & domain_not_empty).astype(np.int8)
        df["email_suffix_domain"] = domain.replace("", "unknown")

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "EmailFeature",
            "layer": "fraud-domain",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "EmailFeature",
                    "description": "Instance name.",
                },
            ],
            "example": "- EmailFeature",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        return {
            "feature_names": [
                "email_suffix_domain",
                "is_disposable_email",
                "is_trusted_email_domain",
                "is_unknown_email_domain",
                "email_domain_missing",
            ],
            "physical_meaning": "Email-domain derived fraud signals",
            "unit": "domain / flag",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_email_basic():
        df = pd.DataFrame({
            "P_emaildomain": [
                "Gmail.com",
                "Mailinator.com",
                np.nan,
                "Yahoo.co.uk",
            ],
        })
        feat = EmailFeature()
        feat.fit(df)
        result = feat.transform(df)

        assert "email_suffix_domain" in result.columns
        assert "is_disposable_email" in result.columns
        assert "is_trusted_email_domain" in result.columns
        assert "is_unknown_email_domain" in result.columns
        assert "email_domain_missing" in result.columns

        assert result.iloc[0]["email_domain_missing"] == 0
        assert result.iloc[1]["is_disposable_email"] == 1
        assert result.iloc[2]["email_domain_missing"] == 1
        assert result.iloc[3]["email_suffix_domain"] == "yahoo.co.uk"

        df_extra = pd.DataFrame({"P_emaildomain": ["mycorp‑biz.com"]})
        res_extra = feat.transform(df_extra)
        assert res_extra.iloc[0]["is_unknown_email_domain"] == 1

        df_nan = pd.DataFrame({"P_emaildomain": [np.nan]})
        res_nan = feat.transform(df_nan)
        assert res_nan.iloc[0]["is_unknown_email_domain"] == 0

    test_email_basic()
    print("All EmailFeature tests passed!")