import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results = []
output_file = "d:/fraudml/reports/test_results.txt"
out_fh = open(output_file, "w", encoding="utf-8")

def log(msg):
    print(msg)
    out_fh.write(msg + "\n")

def run_test(name, fn):
    try:
        fn()
        results.append(f"  PASS: {name}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        results.append(f"  FAIL: {name} — {e}\n{tb}")

log("=" * 50)
log("Feature Module Tests")
log("=" * 50)

# --- MissingPatternFeature ---
print("\n[MissingPatternFeature]")
from src.features.missing_pattern_feature import MissingPatternFeature
import pandas as pd, numpy as np

def test_missing_basic():
    df = pd.DataFrame({"id_01": [1.0, np.nan, 3.0], "id_02": [np.nan, np.nan, 6.0], "V1": [7.0, 8.0, 9.0]})
    feat = MissingPatternFeature(); feat.fit(df); result = feat.transform(df)
    assert result.iloc[0]["total_missing_count"] == 1
    assert result.iloc[1]["total_missing_count"] == 2
run_test("missing_basic", test_missing_basic)

def test_missing_all_nonnull():
    df = pd.DataFrame({"id_01": [1,2,3], "V1": [4,5,6]})
    feat = MissingPatternFeature(); feat.fit(df); result = feat.transform(df)
    assert result["total_missing_count"].sum() == 0
run_test("missing_all_nonnull", test_missing_all_nonnull)

# --- TimeFeature ---
print("\n[TimeFeature]")
from src.features.time_feature import TimeFeature

def test_time():
    df = pd.DataFrame({"TransactionDT": [3600, 43200, 75600]})
    feat = TimeFeature(); feat.fit(df); result = feat.transform(df)
    assert result.iloc[0]["TransactionDT_hour"] == 1
    assert result.iloc[1]["TransactionDT_hour"] == 12
    assert result.iloc[2]["TransactionDT_hour"] == 21
run_test("time_basic", test_time)

# --- DeviceFeature ---
print("\n[DeviceFeature]")
from src.features.device_feature import DeviceFeature

def test_device():
    df = pd.DataFrame({"id_30": ["android 10", "iOS 14", "Windows 10", np.nan], "id_01": [1.0, np.nan, 3.0, np.nan], "id_02": [np.nan, 2.0, np.nan, np.nan]})
    feat = DeviceFeature(); feat.fit(df); result = feat.transform(df)
    assert result.iloc[0]["device_os_android"] == 1
    assert result.iloc[1]["device_os_ios"] == 1
    assert result.iloc[2]["device_os_windows"] == 1
    assert result.iloc[3]["device_os_unknown"] == 1
    assert result.iloc[3]["device_info_missing"] == 1
run_test("device_basic", test_device)

# --- EmailFeature ---
print("\n[EmailFeature]")
from src.features.email_feature import EmailFeature

def test_email():
    df = pd.DataFrame({"P_emaildomain": ["Gmail.com", "Mailinator.com", np.nan, "Yahoo.co.uk"]})
    feat = EmailFeature(); feat.fit(df); result = feat.transform(df)
    assert result.iloc[0]["email_domain_missing"] == 0
    assert result.iloc[1]["is_disposable_email"] == 1
    assert result.iloc[2]["email_domain_missing"] == 1
run_test("email_basic", test_email)

# --- CardFeature ---
print("\n[CardFeature]")
from src.features.card_feature import CardFeature

def test_card():
    df = pd.DataFrame({"card1": [100, 200, 300], "card2": [1, np.nan, 3], "card3": [np.nan, 2, 3], "card5": [5, 5, np.nan]})
    feat = CardFeature(); feat.fit(df); result = feat.transform(df)
    assert result.iloc[1]["card2_missing_flag"] == 1
    assert result.iloc[0]["card1_card2"] == "100@1"
run_test("card_basic", test_card)

# --- AddrFeature ---
print("\n[AddrFeature]")
from src.features.addr_feature import AddrFeature

def test_addr():
    df = pd.DataFrame({"addr1": [100, np.nan, 300], "addr2": [np.nan, 200, 300]})
    feat = AddrFeature(); feat.fit(df); result = feat.transform(df)
    assert result.iloc[0]["addr1_missing_flag"] == 0
    assert result.iloc[0]["addr2_missing_flag"] == 1
    assert result.iloc[2]["addr1_addr2"] == "300@300"
run_test("addr_basic", test_addr)

# --- CrossFeature ---
print("\n[CrossFeature]")
from src.features.cross_feature import CrossFeature

def test_cross():
    df = pd.DataFrame({"card1": [100,200], "addr1": [1,2], "card2": [10,20], "P_emaildomain": ["gmail.com","yahoo.com"]})
    feat = CrossFeature(); feat.fit(df); result = feat.transform(df)
    assert result.iloc[0]["card1@addr1"] == "100@1"
run_test("cross_basic", test_cross)

# --- AmountFeature ---
print("\n[AmountFeature]")
from src.features.amount_feature import AmountFeature

def test_amount():
    df = pd.DataFrame({"card1": ["A","A","B"], "TransactionAmt": [10.5, 20.0, 5.0]})
    feat = AmountFeature(); feat.fit(df); result = feat.transform(df)
    assert "log_TransactionAmt" in result.columns
    assert "card1_amount_ratio" in result.columns
    assert "card1_amount_delta" in result.columns
run_test("amount_basic", test_amount)

def test_amount_leakage():
    df1 = pd.DataFrame({"card1": ["A","A","A"], "TransactionAmt": [10.0, 20.0, 99999.0]})
    df2 = pd.DataFrame({"card1": ["A","A","A"], "TransactionAmt": [10.0, 20.0, 30.0]})
    feat = AmountFeature(); feat.fit(df1)
    r1 = feat.transform(df1); r2 = feat.transform(df2)
    assert r1.iloc[0]["card1_amount_delta"] == r2.iloc[0]["card1_amount_delta"]
    assert r1.iloc[1]["card1_amount_delta"] == r2.iloc[1]["card1_amount_delta"]
run_test("amount_leakage", test_amount_leakage)

# --- HistoryFeature ---
print("\n[HistoryFeature]")
from src.features.history_feature import HistoryFeature

def test_history():
    df = pd.DataFrame({"card1": ["A","A","A","B","B"], "TransactionDT": [1000,2000,3000,1500,2500], "TransactionAmt": [10.0,20.0,30.0,5.0,15.0]})
    feat = HistoryFeature(); feat.fit(df); result = feat.transform(df)
    assert "time_since_last_transaction" in result.columns
    assert result.iloc[0]["time_since_last_transaction"] == 0.0
    assert result.iloc[1]["time_since_last_transaction"] == 1000.0
    assert result.iloc[0]["cumulative_spend"] == 0.0
    assert result.iloc[2]["cumulative_spend"] == 30.0
run_test("history_basic", test_history)

def test_history_leakage():
    df1 = pd.DataFrame({"card1": ["A","A","A"], "TransactionDT": [1000,2000,3000], "TransactionAmt": [10.0,20.0,99999.0]})
    df2 = pd.DataFrame({"card1": ["A","A","A"], "TransactionDT": [1000,2000,3000], "TransactionAmt": [10.0,20.0,30.0]})
    feat = HistoryFeature(); feat.fit(df1)
    r1 = feat.transform(df1); r2 = feat.transform(df2)
    assert r1.iloc[2]["cumulative_spend"] == 30.0
    assert r2.iloc[2]["cumulative_spend"] == 30.0
run_test("history_leakage", test_history_leakage)

# --- AggregationFeature ---
print("\n[AggregationFeature]")
from src.features.aggregation_feature import AggregationFeature

def test_agg():
    df = pd.DataFrame({"card1": ["A","A","A","B","B"], "addr1": [1,1,2,3,3], "TransactionAmt": [10.0,20.0,30.0,5.0,15.0]})
    feat = AggregationFeature(agg_cols=["TransactionAmt"], group_keys=[("card1",)], stats=["count","sum"])
    feat.fit(df); result = feat.transform(df)
    assert result.iloc[0]["card1_TransactionAmt_count"] == 0
    assert result.iloc[1]["card1_TransactionAmt_count"] == 1
    assert result.iloc[2]["card1_TransactionAmt_sum"] == 30
run_test("agg_basic", test_agg)

def test_agg_leakage():
    df1 = pd.DataFrame({"card1": ["A","A","A"], "TransactionAmt": [10.0,20.0,99999.0]})
    df2 = pd.DataFrame({"card1": ["A","A","A"], "TransactionAmt": [10.0,20.0,30.0]})
    feat = AggregationFeature(agg_cols=["TransactionAmt"], group_keys=[("card1",)], stats=["sum"])
    feat.fit(df1); r1 = feat.transform(df1); r2 = feat.transform(df2)
    assert r1.iloc[2]["card1_TransactionAmt_sum"] == 30.0
    assert r2.iloc[2]["card1_TransactionAmt_sum"] == 30.0
run_test("agg_leakage", test_agg_leakage)

# --- TargetEncoderFeature ---
print("\n[TargetEncoderFeature]")
from src.features.encoding import TargetEncoderFeature

def test_te_basic():
    df = pd.DataFrame({"card1": ["A","A","B","B","C"], "isFraud": [1,0,1,1,0]})
    feat = TargetEncoderFeature(target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0)
    feat.fit(df); result = feat.transform(df)
    assert "card1_target_enc" in result.columns
    assert abs(result.iloc[0]["card1_target_enc"] - 0.5) < 0.01
run_test("te_basic", test_te_basic)

def test_te_unseen():
    train = pd.DataFrame({"card1": ["A","A","B"], "isFraud": [1,0,1]})
    val = pd.DataFrame({"card1": ["C","A"], "isFraud": [0,0]})
    feat = TargetEncoderFeature(target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0)
    feat.fit(train); result = feat.transform(val)
    gm = train["isFraud"].mean()
    assert abs(result.iloc[0]["card1_target_enc"] - gm) < 0.01
run_test("te_unseen", test_te_unseen)

def test_te_no_leakage():
    train = pd.DataFrame({"card1": ["A","A","B"], "isFraud": [1,0,1]})
    feat = TargetEncoderFeature(target_col="isFraud", encode_cols=["card1"], smoothing=0, min_samples=0)
    feat.fit(train)
    val = pd.DataFrame({"card1": ["A","A","B"], "isFraud": [99,99,99]})
    result = feat.transform(val)
    assert abs(result.iloc[0]["card1_target_enc"] - 0.5) < 0.01
    assert abs(result.iloc[2]["card1_target_enc"] - 1.0) < 0.01
run_test("te_no_leakage", test_te_no_leakage)

# --- FeatureRegistry generate ---
print("\n[FeatureRegistry]")
from src.features import FeatureRegistry

def test_registry_generate():
    df = pd.DataFrame({
        "card1": ["A","A","B","B"], "card2": [1,np.nan,3,3],
        "addr1": [10,20,30,40], "TransactionAmt": [100.0,200.0,300.0,400.0],
        "TransactionDT": [3600,7200,10800,14400],
        "isFraud": [0,1,0,1], "id_30": ["android","iOS","Windows",np.nan],
        "P_emaildomain": ["gmail.com","yahoo.com",np.nan,"hotmail.com"],
    })
    import yaml
    config = {"feature_steps": [
        "MissingPatternFeature", "TimeFeature", "DeviceFeature",
        "EmailFeature", "CardFeature", "AddrFeature",
        "CrossFeature", "AmountFeature", "HistoryFeature",
        "AggregationFeature", "TargetEncoderFeature",
    ]}
    with open("d:/fraudml/config.yaml", "w") as f:
        yaml.dump(config, f)

    registry = FeatureRegistry(config_path="d:/fraudml/config.yaml")
    registry.auto_discover("src.features")
    result = registry.generate(df)
    n_cols = len(result.columns)
    log(f"    Generated {n_cols} features")
    assert n_cols > len(df.columns)
run_test("registry_generate", test_registry_generate)

# --- Summary ---
log("\n" + "=" * 50)
log("RESULTS:")
log("=" * 50)
for r in results:
    log(r)
passed = sum(1 for r in results if r.startswith("  PASS"))
failed = sum(1 for r in results if r.startswith("  FAIL"))
log(f"\nTotal: {passed} passed, {failed} failed")

out_fh.close()

if failed > 0:
    sys.exit(1)
else:
    log("All tests passed!")
    sys.exit(0)