from mkv_episode_matcher.misalignment import MisalignmentPolicy


def test_offsets_are_deterministic():
    policy = MisalignmentPolicy("random", 1.0, 3.0, 42, per_segment=True)
    first = policy.offset_for("video_a", 7)
    second = policy.offset_for("video_a", 7)
    assert first == second


def test_left_profile_offsets_are_negative_and_in_range():
    policy = MisalignmentPolicy("left", 1.0, 3.0, 123, per_segment=False)
    offset = policy.offset_for("video_a", 2)
    assert -3.0 <= offset <= -1.0


def test_right_profile_offsets_are_positive_and_in_range():
    policy = MisalignmentPolicy("right", 1.0, 3.0, 123, per_segment=False)
    offset = policy.offset_for("video_b", 2)
    assert 1.0 <= offset <= 3.0


def test_random_profile_offsets_respect_absolute_range():
    policy = MisalignmentPolicy("random", 1.0, 3.0, 11, per_segment=True)
    offset = policy.offset_for("video_c", 2)
    assert 1.0 <= abs(offset) <= 3.0
