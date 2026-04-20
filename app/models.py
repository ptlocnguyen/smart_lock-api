from pydantic import BaseModel
from typing import Optional


class FingerprintSync(BaseModel):
    user_id: str
    fingerprint_id: int
    device_id: str


class FaceRecognizeResponse(BaseModel):
    user_id: Optional[str]
    score: float
    result: str