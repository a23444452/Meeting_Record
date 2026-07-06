"""確定性生成 300 筆會議情境種子。"""
import json
import random
from pathlib import Path

INDUSTRIES = ["科技軟體", "電子製造", "電商零售", "醫療院所", "金融保險",
              "教育培訓", "餐飲連鎖", "行銷代理", "新創團隊", "非營利組織"]
MEETING_TYPES = ["部門週會", "專案進度檢討", "客戶需求訪談", "跨部門協調會",
                 "緊急事故處理", "產品規劃會議", "業績檢討會", "供應商洽談"]
NOISE_FEATURES = ["口語贅詞偏多", "有人離題閒聊", "發言互相打斷",
                  "中英夾雜（技術詞彙）", "有人中途加入", "有人提早離席"]
LENGTH_WEIGHTS = [("short", 0.4), ("medium", 0.4), ("long", 0.2)]


def generate_seeds(n: int = 300, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    lengths = [name for name, w in LENGTH_WEIGHTS for _ in range(int(n * w))]
    lengths += ["medium"] * (n - len(lengths))
    rng.shuffle(lengths)
    seeds = []
    for i in range(n):
        industry = INDUSTRIES[i % len(INDUSTRIES)]
        meeting_type = MEETING_TYPES[(i // len(INDUSTRIES)) % len(MEETING_TYPES)]
        seeds.append({
            "id": f"s{i + 1:03d}",
            "industry": industry,
            "meeting_type": meeting_type,
            "num_participants": rng.randint(3, 8),
            "length": lengths[i],
            "noise_features": sorted(rng.sample(NOISE_FEATURES, rng.randint(1, 3))),
        })
    return seeds


def main() -> None:
    out = Path("data/seeds/scenarios.json")
    out.write_text(json.dumps(generate_seeds(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已寫入 {out}（300 筆）")


if __name__ == "__main__":
    main()
