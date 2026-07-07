from scripts.evaluate import score_output

GOOD = """## 會議主旨
測試會議。

## 討論要點
- 要點一。

## 決議事項
- 本次會議無正式決議

## 行動項目
| 事項 | 負責人 | 期限 |
|------|--------|------|
| 追蹤進度 | 待指派 | 未定 |
"""


def test_score_good_output():
    s = score_output(GOOD)
    assert s == {"sections_ok": True, "table_ok": True, "no_simplified": True}


def test_score_bad_output():
    s = score_output("这是一段没有结构的简体输出")
    assert s == {"sections_ok": False, "table_ok": False, "no_simplified": False}
