"""Cấu trúc ngoại lệ (Exceptions) rõ ràng cho hệ thống ghép ảnh toàn cảnh."""

from __future__ import annotations
from typing import Any, Dict, Optional


class PanoramaError(Exception):
    """Lớp ngoại lệ cơ sở cho toàn bộ pipeline ghép ảnh."""

    code: str = "PANORAMA_ERROR"

    def __init__(
        self,
        message: str,
        suggestion: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "suggestion": self.suggestion,
            "details": self.details,
        }


class InsufficientImagesError(PanoramaError):
    code = "INSUFFICIENT_IMAGES"


class ImageLoadError(PanoramaError):
    code = "IMAGE_LOAD_ERROR"


class InsufficientFeaturesError(PanoramaError):
    code = "INSUFFICIENT_FEATURES"


class InsufficientMatchesError(PanoramaError):
    code = "INSUFFICIENT_MATCHES"


class HomographyEstimationError(PanoramaError):
    code = "HOMOGRAPHY_ESTIMATION_ERROR"


class DisconnectedImagesError(PanoramaError):
    code = "DISCONNECTED_IMAGES"


class CanvasOverflowError(PanoramaError):
    code = "CANVAS_OVERFLOW"
