from scripts.gen_seeds import generate_seeds

def test_generates_300_unique_seeds():
    seeds = generate_seeds()
    assert len(seeds) == 300
    assert len({s["id"] for s in seeds}) == 300

def test_seed_schema():
    s = generate_seeds()[0]
    assert set(s) == {"id", "industry", "meeting_type", "num_participants", "length", "noise_features"}
    assert s["length"] in {"short", "medium", "long"}
    assert 3 <= s["num_participants"] <= 8
    assert 1 <= len(s["noise_features"]) <= 3

def test_deterministic():
    assert generate_seeds() == generate_seeds()

def test_coverage():
    seeds = generate_seeds()
    assert len({s["industry"] for s in seeds}) >= 10
    assert len({s["meeting_type"] for s in seeds}) >= 8
    lengths = [s["length"] for s in seeds]
    assert lengths.count("long") >= 40  # 長逐字稿佔比不可太低
