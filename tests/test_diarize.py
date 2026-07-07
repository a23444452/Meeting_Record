"""講者指派純邏輯測試（不需要 pyannote 安裝）。"""

from meeting_record.diarize import SpeakerTurn, assign_speakers
from meeting_record.models import Segment


def seg(start: float, end: float, text: str = "內容") -> Segment:
    return Segment(start=start, end=end, text=text)


class TestAssignSpeakers:
    def test_assigns_by_max_overlap(self):
        segments = [seg(0, 4), seg(4, 8)]
        turns = [
            SpeakerTurn(start=0, end=4.5, speaker="SPEAKER_00"),
            SpeakerTurn(start=4.5, end=8, speaker="SPEAKER_01"),
        ]
        result = assign_speakers(segments, turns)
        # 第二段與 SPEAKER_00 重疊 0.5 秒、與 SPEAKER_01 重疊 3.5 秒 → 取後者
        assert [s.speaker for s in result] == ["講者1", "講者2"]

    def test_friendly_labels_follow_first_appearance(self):
        segments = [seg(0, 2), seg(2, 4), seg(4, 6)]
        turns = [
            SpeakerTurn(start=0, end=2, speaker="SPEAKER_07"),
            SpeakerTurn(start=2, end=4, speaker="SPEAKER_02"),
            SpeakerTurn(start=4, end=6, speaker="SPEAKER_07"),
        ]
        result = assign_speakers(segments, turns)
        assert [s.speaker for s in result] == ["講者1", "講者2", "講者1"]

    def test_no_overlap_keeps_none(self):
        segments = [seg(0, 2), seg(10, 12)]
        turns = [SpeakerTurn(start=0, end=2, speaker="SPEAKER_00")]
        result = assign_speakers(segments, turns)
        assert result[0].speaker == "講者1"
        assert result[1].speaker is None

    def test_empty_turns_returns_segments_unchanged(self):
        segments = [seg(0, 2)]
        result = assign_speakers(segments, [])
        assert result == segments

    def test_original_segments_not_mutated(self):
        segments = [seg(0, 2)]
        turns = [SpeakerTurn(start=0, end=2, speaker="SPEAKER_00")]
        assign_speakers(segments, turns)
        assert segments[0].speaker is None

    def test_turn_starting_after_segment_end_ignored(self):
        segments = [seg(0, 2)]
        turns = [
            SpeakerTurn(start=1, end=2, speaker="SPEAKER_00"),
            SpeakerTurn(start=2, end=5, speaker="SPEAKER_01"),  # start == seg.end → 不算重疊
        ]
        result = assign_speakers(segments, turns)
        assert result[0].speaker == "講者1"
