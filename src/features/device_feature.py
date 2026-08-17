"""
DeviceFeature — Device info parsing.

Extracts OS category flag、consistency conflict flags、version tier、high‑cardinality derived features
from DeviceInfo / id_30 / id_31 / DeviceType columns.
Stateless — NO groupby / window / target‑dependent operations.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import FeatureBase


def _extract_major_version(s: str | None) -> float | None:
    """
    Stateless helper: extract major version number from raw os/browser string.
    e.g. "Android 7.0" →7.0; "chrome 62.0" →62.0; None/nan → None
    """
    if pd.isna(s):
        return None
    s_text = str(s)
    match = re.search(r"(\d+)", s_text)
    if match:
        return float(match.group(1))
    return None


def _hash_mod_bucket(text: str | None, bucket_size: int = 64) -> int:
    """Stateless hash bucket for high cardinality DeviceInfo, no training stat needed."""
    if pd.isna(text):
        return -1
    h = hashlib.md5(str(text).encode("utf‑8")).hexdigest()
    val = int(h, 16)
    return val % bucket_size


class DeviceFeature(FeatureBase):
    """Parse device‑related features.

    **Stateless** — fit() is a no-op.  No parameters learned.

    Features list:
        ## OS flags
        - device_os_android: 1 if android
        - device_os_ios: 1 if ios
        - device_os_windows: 1 if windows
        - device_os_macos: 1 if macos
        - device_os_unknown: 1 if os cannot be determined

        ## DeviceType flags
        - device_type_mobile: 1 if DeviceType == mobile
        - device_type_desktop: 1 if DeviceType == desktop

        ## Consistency / conflict flags (fraud signal)
        - dev_multi_os_hit: 1 when more than one OS matched
        - dev_os_devinfo_mismatch: 1 when id_30 os != DeviceInfo os (both non‑null)
        - dev_os_devicetype_conflict: 1 when OS contradicts DeviceType
        - dev_browser_os_mismatch: 1 when browser contradicts OS
        - id30_missing_but_devinfo_ok: 1 id_30 null but DeviceInfo has value

        ## Browser flags from id_31
        - browser_chrome
        - browser_safari
        - browser_samsung
        - browser_firefox
        - browser_edge
        - browser_unknown

        ## Version numeric & tier (hard‑coded stateless threshold)
        - os_android_ver_major
        - os_ios_ver_major
        - browser_chrome_ver_major
        - os_android_ver_tier: low/mid/high/none
        - os_ios_ver_tier: low/mid/high/none
        - browser_chrome_ver_tier: low/mid/high/none
        - os_is_very_old: 1 for extremely old os version

        ## High‑cardinality DeviceInfo derived
        - devinfo_str_len: length of DeviceInfo string
        - devinfo_has_build_tag: whether contains "Build/"
        - devinfo_hash_bucket: md5 hash mod 64 bucket, -1 for missing
        - devinfo_has_samsung
        - devinfo_has_iphone

        ## Completeness flag
        - device_info_missing: 1 if id_* all null AND DeviceInfo null
    """

    @property
    def is_stateful(self) -> bool:
        return False

    def __init__(self, name: str = "DeviceFeature", hash_bucket_size: int = 64) -> None:
        super().__init__(name=name)
        self.hash_bucket_size = hash_bucket_size

    def fit(self, df: pd.DataFrame) -> "DeviceFeature":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: not fitted.")
        df = df.copy()

        # --------------------------
        # Step 0: Clean common placeholder values to np.nan
        # --------------------------
        def clean_dash(series: pd.Series) -> pd.Series:
            return series.replace({
                "—": np.nan,
                "-": np.nan,
                "--": np.nan,
                "unknown": np.nan,
            })

        # --------------------------
        # Step1: OS parse: id_30 and DeviceInfo, parse separately for conflict detection
        # --------------------------
        id30_raw = clean_dash(df["id_30"]) if "id_30" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        devinfo_raw = clean_dash(df["DeviceInfo"]) if "DeviceInfo" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        id30_str = id30_raw.astype(str).str.lower()
        devinfo_str = devinfo_raw.astype(str).str.lower()

        # parse OS flag from id_30 alone
        id30_android = id30_str.str.contains("android", na=False)
        id30_ios = id30_str.str.contains("ios", na=False)
        id30_windows = id30_str.str.contains("windows", na=False)
        id30_macos = id30_str.str.contains(r"\bmac(?:\s|os|_)", na=False, regex=True)

        # parse OS flag from DeviceInfo alone
        di_android = devinfo_str.str.contains(r"android|build/", na=False)
        di_ios = devinfo_str.str.contains("ios", na=False)
        di_windows = devinfo_str.str.contains("windows", na=False)
        di_macos = devinfo_str.str.contains(r"\bmac(?:\s|os|_)", na=False, regex=True)

        # combined os source for main os flags
        combined_os = id30_raw.where(id30_raw.notna(), devinfo_raw)
        combined_os_str = combined_os.astype(str).str.lower()
        df["device_os_android"] = combined_os_str.str.contains(r"android|build/", na=False).astype(np.int8)
        df["device_os_ios"] = combined_os_str.str.contains("ios", na=False).astype(np.int8)
        df["device_os_windows"] = combined_os_str.str.contains("windows", na=False).astype(np.int8)
        df["device_os_macos"] = combined_os_str.str.contains(r"\bmac(?:\s|os|_)", na=False, regex=True).astype(np.int8)

        known_os = (df["device_os_android"] | df["device_os_ios"] | df["device_os_windows"] | df["device_os_macos"]).astype(bool)
        df["device_os_unknown"] = (~known_os).astype(np.int8)

        # --------------------------
        # Step2: DeviceType features
        # --------------------------
        if "DeviceType" in df.columns:
            dt_series = clean_dash(df["DeviceType"]).astype(str).str.lower()
            df["device_type_mobile"] = (dt_series == "mobile").astype(np.int8)
            df["device_type_desktop"] = (dt_series == "desktop").astype(np.int8)
        else:
            df["device_type_mobile"] = 0
            df["device_type_desktop"] = 0

        # --------------------------
        # Step3: Consistency & conflict features
        # --------------------------
        # dev_multi_os_hit: multiple os flag triggered
        sum_os = df["device_os_android"] + df["device_os_ios"] + df["device_os_windows"] + df["device_os_macos"]
        df["dev_multi_os_hit"] = (sum_os > 1).astype(np.int8)

        # id30_missing_but_devinfo_ok
        df["id30_missing_but_devinfo_ok"] = ((id30_raw.isna()) & (devinfo_raw.notna())).astype(np.int8)

        # dev_os_devinfo_mismatch: both non‑null, os conclusion conflict
        id30_has_os = id30_android | id30_ios | id30_windows | id30_macos
        di_has_os = di_android | di_ios | di_windows | di_macos
        both_has_os = id30_has_os & di_has_os

        id30_os_idx = np.argmax([id30_android.values, id30_ios.values, id30_windows.values, id30_macos.values], axis=0)
        di_os_idx = np.argmax([di_android.values, di_ios.values, di_windows.values, di_macos.values], axis=0)
        mismatch_arr = (id30_os_idx != di_os_idx) & both_has_os.values
        df["dev_os_devinfo_mismatch"] = mismatch_arr.astype(np.int8)

        # dev_os_devicetype_conflict: os vs devicetype business rule conflict
        is_mobile_os = df["device_os_android"] | df["device_os_ios"]
        is_desktop_os = df["device_os_windows"] | df["device_os_macos"]
        dt_mobile = df["device_type_mobile"].astype(bool)
        dt_desktop = df["device_type_desktop"].astype(bool)

        conflict = ((is_mobile_os & dt_desktop) | (is_desktop_os & dt_mobile))
        # only mark conflict when both side is valid(not unknown)
        valid_both = ((is_mobile_os | is_desktop_os) & (dt_mobile | dt_desktop))
        df["dev_os_devicetype_conflict"] = (conflict & valid_both).astype(np.int8)

        # --------------------------
        # Step4: Browser(id_31) parse & browser‑os mismatch
        # --------------------------
        if "id_31" in df.columns:
            br_raw = clean_dash(df["id_31"])
            br_str = br_raw.astype(str).str.lower()
            df["browser_chrome"] = br_str.str.contains("chrome", na=False).astype(np.int8)
            df["browser_safari"] = br_str.str.contains("safari", na=False).astype(np.int8)
            df["browser_samsung"] = br_str.str.contains("samsung browser", na=False).astype(np.int8)
            df["browser_firefox"] = br_str.str.contains("firefox", na=False).astype(np.int8)
            df["browser_edge"] = br_str.str.contains("edge", na=False).astype(np.int8)
            # browser_unknown: 全部已知浏览器flag都为0，则unknown=1
            known_browser_sum = (
                df["browser_chrome"]
                + df["browser_safari"]
                + df["browser_samsung"]
                + df["browser_firefox"]
                + df["browser_edge"]
            )
            df["browser_unknown"] = (known_browser_sum == 0).astype(np.int8)


            # dev_browser_os_mismatch
            # samsung browser should only on android; mobile safari only on ios
            br_samsung = df["browser_samsung"].astype(bool)
            br_mobile_safari = br_str.str.contains("mobile safari", na=False)
            os_not_android = ~df["device_os_android"].astype(bool)
            os_not_ios = ~df["device_os_ios"].astype(bool)

            br_conflict = (br_samsung & os_not_android) | (br_mobile_safari & os_not_ios)
            valid_br_os = (br_samsung | br_mobile_safari) & (~df["device_os_unknown"].astype(bool))
            df["dev_browser_os_mismatch"] = (br_conflict & valid_br_os).astype(np.int8)
        else:
            df["browser_chrome"] = 0
            df["browser_safari"] = 0
            df["browser_samsung"] = 0
            df["browser_firefox"] = 0
            df["browser_edge"] = 0
            df["browser_unknown"] = 1   # id_31列完全不存在，浏览器直接unknown=1
            df["dev_browser_os_mismatch"] = 0

        # --------------------------
        # Step5: Version extract & tier (hard‑coded stateless threshold)
        # --------------------------
        def get_tier(major_ver: float | None, low: float, mid: float) -> str:
            if pd.isna(major_ver):
                return "none"
            if major_ver < low:
                return "low"
            elif major_ver < mid:
                return "mid"
            else:
                return "high"

        # android version
        android_ver_series = combined_os.apply(_extract_major_version)
        df["os_android_ver_major"] = android_ver_series
        df["os_android_ver_tier"] = android_ver_series.apply(lambda x: get_tier(x, low=8, mid=13))

        # ios version
        ios_mask = combined_os_str.str.contains("ios", na=False)
        ios_ver_series = combined_os.where(ios_mask, np.nan).apply(_extract_major_version)
        df["os_ios_ver_major"] = ios_ver_series
        df["os_ios_ver_tier"] = ios_ver_series.apply(lambda x: get_tier(x, low=12, mid=16))

        # chrome browser version
        if "id_31" in df.columns:
            chrome_mask = br_str.str.contains("chrome", na=False)
            chrome_ver_series = br_raw.where(chrome_mask, np.nan).apply(_extract_major_version)
            df["browser_chrome_ver_major"] = chrome_ver_series
            df["browser_chrome_ver_tier"] = chrome_ver_series.apply(lambda x: get_tier(x, low=80, mid=100))
        else:
            df["browser_chrome_ver_major"] = np.nan
            df["browser_chrome_ver_tier"] = "none"

        # os_is_very_old flag
        very_old_android = (~android_ver_series.isna()) & (android_ver_series <= 7)
        very_old_ios = (~ios_ver_series.isna()) & (ios_ver_series <= 11)
        df["os_is_very_old"] = (very_old_android | very_old_ios).astype(np.int8)

        # --------------------------
        # Step6: High‑cardinality DeviceInfo derived features
        # --------------------------
        df["devinfo_str_len"] = devinfo_raw.apply(lambda s: len(str(s)) if not pd.isna(s) else np.nan)
        df["devinfo_has_build_tag"] = devinfo_str.str.contains(r"build/", na=False).astype(np.int8)
        df["devinfo_hash_bucket"] = devinfo_raw.apply(lambda x: _hash_mod_bucket(x, self.hash_bucket_size))
        df["devinfo_has_samsung"] = devinfo_str.str.contains("samsung", na=False).astype(np.int8)
        df["devinfo_has_iphone"] = devinfo_str.str.contains("iphone", na=False).astype(np.int8)

        # --------------------------
        # Step7: device_info_missing (revised rule: id_* all null AND DeviceInfo null)
        # --------------------------
        id_cols = [c for c in df.columns if c.startswith("id_")]
        id_all_null = df[id_cols].isnull().all(axis=1) if id_cols else True
        devinfo_null = devinfo_raw.isna()
        df["device_info_missing"] = (id_all_null & devinfo_null).astype(np.int8)

        return df

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "class_name": "DeviceFeature",
            "layer": "fraud-domain",
            "is_stateful": False,
            "parameters": [
                {
                    "name": "name",
                    "type": "str",
                    "default": "DeviceFeature",
                    "description": "Instance name.",
                },
            ],
            "example": "- DeviceFeature",
        }

    def get_feature_metadata(self) -> Dict[str, Any]:
        feature_names = [
            "device_os_android",
            "device_os_ios",
            "device_os_windows",
            "device_os_macos",
            "device_os_unknown",
            "device_type_mobile",
            "device_type_desktop",
            "dev_multi_os_hit",
            "dev_os_devinfo_mismatch",
            "dev_os_devicetype_conflict",
            "dev_browser_os_mismatch",
            "id30_missing_but_devinfo_ok",
            "browser_chrome",
            "browser_safari",
            "browser_samsung",
            "browser_firefox",
            "browser_edge",
            "browser_unknown",
            "os_android_ver_major",
            "os_ios_ver_major",
            "browser_chrome_ver_major",
            "os_android_ver_tier",
            "os_ios_ver_tier",
            "browser_chrome_ver_tier",
            "os_is_very_old",
            "devinfo_str_len",
            "devinfo_has_build_tag",
            "devinfo_hash_bucket",
            "devinfo_has_samsung",
            "devinfo_has_iphone",
            "device_info_missing",

            
        ]
        return {
            "feature_names": feature_names,
            "physical_meaning": "Device OS、device type、consistency conflict、version tier、high‑cardinality derived features",
            "unit": "flag/numeric/category",
            "depends_on_target": False,
        }


if __name__ == "__main__":
    def test_device_basic():
        df = pd.DataFrame({
            "id_30": ["Android 7.0", "iOS 11.1.2", "Windows 10", "Mac OS X 10_11_6", np.nan, "—", "Linux"],
            "DeviceInfo": [np.nan, np.nan, np.nan, np.nan, "iOS Device", "SAMSUNG SM‑G892A Build/NRD90M", np.nan],
            "DeviceType": ["mobile", "mobile", "desktop", "desktop", "mobile", "mobile", "desktop"],
            "id_31": ["samsung browser 6.2", "mobile safari 11.0", "chrome 62.0", "chrome 62.0", np.nan, "chrome 62.0", np.nan],
            "id_01": [1.0, np.nan, 3.0, 4.0, np.nan, 5.0, np.nan],
            "id_02": [np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        })
        feat = DeviceFeature(hash_bucket_size=64)
        feat.fit(df)
        result = feat.transform(df)

        # basic os flag
        assert result.iloc[0]["device_os_android"] == 1
        assert result.iloc[1]["device_os_ios"] == 1
        assert result.iloc[2]["device_os_windows"] == 1
        assert result.iloc[3]["device_os_macos"] == 1
        assert result.iloc[4]["device_os_ios"] == 1
        assert result.iloc[5]["device_os_android"] == 1
        assert result.iloc[6]["device_os_unknown"] == 1

        # version tier
        assert result.iloc[0]["os_android_ver_tier"] == "low"
        assert result.iloc[0]["os_is_very_old"] == 1

        # id30 missing but devinfo ok
        assert result.iloc[4]["id30_missing_but_devinfo_ok"] == 1
        assert result.iloc[4]["browser_chrome"] == 0
        assert result.iloc[4]["browser_safari"] == 0
        assert result.iloc[4]["browser_samsung"] == 0
        assert result.iloc[4]["browser_firefox"] == 0
        assert result.iloc[4]["browser_edge"] == 0
        assert result.iloc[4]["browser_unknown"] == 1

        # devinfo build tag
        assert result.iloc[5]["devinfo_has_build_tag"] == 1

        # device_info_missing: row4 id_* all null but DeviceInfo exists → not missing
        assert result.iloc[4]["device_info_missing"] == 0

        # conflict case: android + desktop
        df_conflict = pd.DataFrame({
            "id_30": ["Android 10"],
            "DeviceInfo": ["Android"],
            "DeviceType": ["desktop"],
            "id_01": [1.0],
        })
        res_conflict = feat.transform(df_conflict)
        assert res_conflict.iloc[0]["dev_os_devicetype_conflict"] == 1

        print("All DeviceFeature unit‑tests passed!")

    test_device_basic()