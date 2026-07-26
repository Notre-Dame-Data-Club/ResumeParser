"""
OCR provider manager
Select OCR implementations based on configuration
"""
import numpy as np
from typing import List, Any, Optional


class OCRProviderManager:
    """OCR provider manager"""

    def __init__(self) -> None:
        self._easy_ocr: Optional[Any] = None

    def get_ocr_provider(self) -> Any:
        """Get OCR provider based on configuration"""
        return self._get_easy_ocr_provider()

    def _get_easy_ocr_provider(self) -> Any:
        """Get EasyOCR provider"""
        if self._easy_ocr is None:
            try:
                import easyocr
                # Initialize EasyOCR with English support
                self._easy_ocr = easyocr.Reader(['en'], gpu=True)
            except Exception:
                raise
        return self._easy_ocr

    def ocr_extract(self, image: np.ndarray) -> List[Any]:
        """
        Perform OCR on image.

        Args:
            image: Image array

        Returns:
            List: OCR results
        """
        return self._ocr_with_easy_ocr(image)

    def _ocr_with_easy_ocr(self, image: np.ndarray) -> List[Any]:
        """Run OCR using EasyOCR"""
        try:
            easy_ocr = self._get_easy_ocr_provider()

            # EasyOCR expects RGB images, convert if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already RGB
                result = easy_ocr.readtext(image)
            else:
                # Convert to RGB if needed
                if len(image.shape) == 2:
                    # Grayscale to RGB
                    image_rgb = np.stack([image] * 3, axis=-1)
                else:
                    image_rgb = image
                result = easy_ocr.readtext(image_rgb)

            if not result:
                return []

            formatted_result = []
            for item in result:
                # EasyOCR format: (bbox, text, confidence)
                bbox, text, confidence = item
                formatted_result.append([
                    [bbox],
                    [text, confidence]
                ])

            return formatted_result

        except Exception:
            return []

    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from a file (for document files).

        Args:
            file_path: File path

        Returns:
            Extracted text
        """
        return ""

    def ocr(self, image_data: bytes) -> List[Any]:
        """
        Perform OCR on image data (bytes).

        Args:
            image_data: Image data as bytes

        Returns:
            List: OCR results
        """
        try:
            import cv2
            import numpy as np

            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return []

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return self._ocr_with_easy_ocr(image_rgb)

        except Exception:
            return []


# Create global instance
ocr_provider_manager = OCRProviderManager()
