import unittest

from backend.modules.audio_processor import LogicValidator


class LogicValidatorTests(unittest.TestCase):
    def test_untyped_transcript_numbers_do_not_create_contradiction_warnings(self):
        validator = LogicValidator()

        validator.add_statement(
            "SPEAKER_00",
            "第一季度完成了1个版本，会议持续20分钟。",
            [],
            0.0,
        )
        flags = validator.add_statement(
            "SPEAKER_00",
            "第二季度计划完成5个版本，会议持续90分钟。",
            [],
            10.0,
        )

        self.assertFalse(
            [flag for flag in flags if flag.get("type") == "self_contradiction"],
            "untyped numbers from different contexts are not reliable contradictions",
        )

    def test_structured_data_conflict_is_still_reported(self):
        validator = LogicValidator()

        validator.add_statement(
            "SPEAKER_00",
            "当前结果是100。",
            [{"type": "result", "value": "100"}],
            0.0,
        )
        flags = validator.add_statement(
            "SPEAKER_01",
            "当前结果是150。",
            [{"type": "result", "value": "150"}],
            10.0,
        )

        self.assertTrue(
            [flag for flag in flags if flag.get("type") == "data_conflict"],
            "structured checks should remain available",
        )


if __name__ == "__main__":
    unittest.main()
