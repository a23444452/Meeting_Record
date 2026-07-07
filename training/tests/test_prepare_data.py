import json
from scripts.prepare_data import build_message, split_records

SYSTEM = "測試用 system prompt"


def test_build_message_chat_format():
    rec = {"id": "s001", "transcript": "主管：開會了。", "summary": "## 會議主旨\n測試。"}
    msg = build_message(rec, SYSTEM)
    roles = [m["role"] for m in msg["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert msg["messages"][1]["content"] == "主管：開會了。"
    assert msg["messages"][2]["content"].startswith("## 會議主旨")


def test_split_sizes_and_determinism():
    records = [{"id": f"s{i:03d}"} for i in range(300)]
    a = split_records(records, seed=42)
    b = split_records(records, seed=42)
    assert [len(a["train"]), len(a["valid"]), len(a["test"])] == [270, 20, 10]
    assert a == b
    all_ids = [r["id"] for part in a.values() for r in part]
    assert len(set(all_ids)) == 300
