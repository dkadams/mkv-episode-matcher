import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class MisalignmentPolicy:
    profile: str
    min_seconds: float
    max_seconds: float
    seed: int
    per_segment: bool

    def offset_for(self, video_id: str, segment_index: int) -> float:
        key = f"{self.profile}|{self.seed}|{video_id}"
        if self.per_segment:
            key = f"{key}|{segment_index}"

        magnitude = self._sample_uniform_0_1(key)
        seconds = self.min_seconds + (self.max_seconds - self.min_seconds) * magnitude

        match self.profile:
            case "left":
                return -seconds
            case "right":
                return seconds
            case "random":
                sign_key = f"{key}|sign"
                sign = -1.0 if self._sample_uniform_0_1(sign_key) < 0.5 else 1.0
                return sign * seconds
            case _:
                raise ValueError(f"Unknown misalignment profile: {self.profile}")

    @staticmethod
    def _sample_uniform_0_1(key: str) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        integer = int.from_bytes(digest[:8], "big")
        return integer / float(2**64 - 1)
