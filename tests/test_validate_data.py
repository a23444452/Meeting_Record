from scripts.validate_data import (
    find_simplified_chars, check_summary_structure,
    parse_action_items, check_record,
)

GOOD_SUMMARY = """## 會議主旨
討論第三季產品上線時程。

## 討論要點
- 前端進度落後兩週，需要增援。

## 決議事項
- 上線日期延後至十月十五日。

## 行動項目
| 事項 | 負責人 | 期限 |
|------|--------|------|
| 提出增援名單 | 小美 | 未定 |
"""


def make_record(**overrides):
    rec = {
        "id": "s001",
        "seed": {"id": "s001", "industry": "科技軟體", "meeting_type": "部門週會",
                 "num_participants": 4, "length": "short", "noise_features": ["口語贅詞偏多"]},
        "transcript": "主管：我們開始開會。小美：好的，那個前端的部分我說明一下。" + "大家討論了很久。" * 60,
        "summary": GOOD_SUMMARY,
    }
    rec.update(overrides)
    return rec


def test_simplified_chars_detected():
    assert find_simplified_chars("这个会议有问题") >= {"这", "会", "议", "问", "题"}
    assert find_simplified_chars("這個會議沒有問題") == set()


def test_summary_structure_ok():
    assert check_summary_structure(GOOD_SUMMARY) == []


def test_summary_structure_missing_section():
    errors = check_summary_structure(GOOD_SUMMARY.replace("## 決議事項", "## 結論"))
    assert any("決議事項" in e for e in errors)


def test_parse_action_items():
    items = parse_action_items(GOOD_SUMMARY)
    assert items == [{"事項": "提出增援名單", "負責人": "小美", "期限": "未定"}]


def test_good_record_passes():
    assert check_record(make_record()) == []


def test_owner_missing_from_transcript_fails():
    bad = GOOD_SUMMARY.replace("小美", "阿宏")
    errors = check_record(make_record(summary=bad))
    assert any("負責人" in e for e in errors)


def test_transcript_too_short_fails():
    errors = check_record(make_record(transcript="主管：散會。小美：好。"))
    assert any("長度" in e for e in errors)


def test_parse_action_items_stops_at_next_section():
    summary_with_extra = GOOD_SUMMARY + """
## 風險評估
| 風險 | 影響 | 對策 |
|------|------|------|
| 人力不足 | 高 | 外包支援 |
"""
    items = parse_action_items(summary_with_extra)
    assert items == [{"事項": "提出增援名單", "負責人": "小美", "期限": "未定"}]


def test_empty_owner_fails():
    bad = GOOD_SUMMARY.replace("| 提出增援名單 | 小美 | 未定 |",
                               "| 提出增援名單 |  | 未定 |")
    errors = check_record(make_record(summary=bad))
    assert any("缺少負責人" in e for e in errors)


def test_no_false_positive_on_legit_traditional_business_sentence():
    legit = (
        "會議紀錄：本次會議由財務部門召集，討論下一季度預算編列與資源配置事宜。"
        "與會人員針對專案進度、風險評估及決策流程進行了詳細說明，"
        "並針對客戶需求變更提出因應對策，最終決議由專案經理負責追蹤後續進度，"
        "並於下次會議前完成相關文件的整理與歸檔工作。"
    )
    assert find_simplified_chars(legit) == set()


def test_simplified_chars_extended_set():
    # 2026-07-08 評估發現的漏網字必須被偵測
    assert find_simplified_chars("晓琪拟定选择方案") >= {"晓", "拟", "择"}
    # 這些正體字不得誤判
    assert find_simplified_chars("曉琪擬定選擇方案，蘭花繼續確認") == set()
