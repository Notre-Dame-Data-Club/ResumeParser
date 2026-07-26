"""
File processor module
"""
import os
import cv2
import base64
import io
import string
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from smartresume.data.text_extractor import TextExtractor
from smartresume.utils.config import config
from smartresume.data.layout_detector import LayoutDetector


class FileProcessor:
    """File processor responsible for handling different file formats"""

    def __init__(self, text_extractor: TextExtractor):
        self.text_extractor = text_extractor

        self.layout_detector = None
        if config.layout_detection.enabled:
            # Let LayoutDetector handle model path automatically (use downloaded model)
            self.layout_detector = LayoutDetector()

    @staticmethod
    def _garbled_ratio(text: str) -> float:
        """Compute garbled ratio in text supporting major language charsets"""
        if not text:
            return 1.0

        def is_valid(c: str) -> bool:
            return (
                c in string.printable or
                '\u4e00' <= c <= '\u9fff' or
                '\u0400' <= c <= '\u04FF' or
                '\u00C0' <= c <= '\u024F' or
                '\u1EA0' <= c <= '\u1EFF' or
                '\u0600' <= c <= '\u06FF' or
                '\u0900' <= c <= '\u097F' or
                '\u0E00' <= c <= '\u0E7F' or
                '\u3040' <= c <= '\u309F' or
                '\u30A0' <= c <= '\u30FF' or
                '\uAC00' <= c <= '\uD7AF'
            )

        valid_chars = sum(1 for c in text if is_valid(c))
        ratio = 1.0 - valid_chars / len(text)
        return ratio

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a file, choosing the appropriate method based on type"""
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self._process_image(file_path)
        elif file_ext == '.pdf':
            return self._process_pdf(file_path)
        elif file_ext in ['.txt', '.md']:
            return self._process_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported: PDF, images (.jpg/.jpeg/.png/.tiff/.bmp), text (.txt/.md)"
            )

    def _process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Process image file"""
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        scale = max(1080 / min(h, w), 1)
        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(image, (new_w, new_h))

        ocr_results = self.text_extractor.ocr_extract(image)
        ocr_results = self.restore_ocr_coordinates(ocr_results, scale)

        page_data = {
            'page_number': 1,
            'text': [],
            'source': 'image_ocr'
        }

        page_data = self.text_extractor.add_ocr_to_page_text(page_data, ocr_results)
        if config.layout_detection.enabled:
            layout_location = self.layout_detector.detect(image.copy())
            sorted_results = self.text_extractor.resort_page_text_with_layout(page_data['text'], 0, layout_location)
        else:
            sorted_results = self.text_extractor.resort_page_text_with_center_location(page_data['text'], 0)

        image_base64 = self._image_to_base64(image)

        return [{'text': sorted_results, 'image': image_base64}]

    def _process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF file"""
        if config.processing.use_force_ocr:
            return self._process_pdf_ocr_only(pdf_path)

        text = self.text_extractor.extract_from_pdf_string(pdf_path)
        garbled_ratio = self._garbled_ratio(text)
        if garbled_ratio > 0.15:
            return self._process_pdf_ocr_only(pdf_path)

        if config.processing.use_pdf_raw_text:
            if text.strip() == "":
                return self._process_pdf_with_ocr(pdf_path)
            text_lines = text.split("\n")
            result = [{'text': [{'text': line} for line in text_lines]}]
            return result
        else:
            return self._process_pdf_with_ocr(pdf_path)

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert image to base64 string"""
        buffered = io.BytesIO()
        Image.fromarray(image).save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _process_pdf_with_ocr(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF combining text extraction and OCR"""
        page_texts, images = self.text_extractor.extract_with_positions(pdf_path, extract_text=True, extract_render_img=True)
        # images = self._pdf_to_images(pdf_path)
        results = []

        for page_num, img in enumerate(images):
            blacked_out_image = self._blackout_text(img.copy(), page_texts[page_num])

            h, w, _ = blacked_out_image.shape

            new_size = 960 / min(h, w)
            new_w = int(w * new_size)
            new_h = int(h * new_size)

            blacked_out_image = cv2.resize(blacked_out_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ocr_results = self.text_extractor.ocr_extract(blacked_out_image)
            ocr_results = self.restore_ocr_coordinates(ocr_results, new_size)

            page_data = {
                'page_number': page_num + 1,
                'text': page_texts[page_num],
                'source': 'pdf_text_with_ocr'
            }

            combined_page_data = self.text_extractor.add_ocr_to_page_text(page_data, ocr_results)

            if config.layout_detection.enabled:
                layout_location = self.layout_detector.detect(img.copy())
                sorted_texts = self.text_extractor.resort_page_text_with_layout(combined_page_data['text'], page_num, layout_location)
            else:
                sorted_texts = self.text_extractor.resort_page_text_with_center_location(combined_page_data['text'], page_num)

            image_base64 = self._image_to_base64(img)

            page_result = {'text': sorted_texts, 'image': image_base64}

            results.append(page_result)

        return results

    def restore_ocr_coordinates(self, ocr_results: List[Any], scale: float) -> List[Any]:
        """
        Restore coordinates for nested OCR result structures by applying inverse scale.

        Args:
            ocr_results: OCR engine result list.
            scale: Scale used during OCR (i.e., new_size).

        Returns:
            OCR results with coordinates mapped back to original size, preserving structure.
        """
        if not ocr_results:
            return []

        if scale == 0:
            return []

        inverse_scale = 1.0 / scale
        restored_results = []

        for result in ocr_results:
            try:
                box_points_on_resized = result[0][0]
                text_info = result[1]

                restored_box_points = []
                for point in box_points_on_resized:
                    restored_x = int(round(point[0] * inverse_scale))
                    restored_y = int(round(point[1] * inverse_scale))
                    restored_box_points.append([restored_x, restored_y])

                restored_box_data = [restored_box_points]

                restored_results.append([restored_box_data, text_info])

            except (TypeError, IndexError):
                continue

        return restored_results

    def _process_pdf_ocr_only(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF using OCR only"""
        _, images = self.text_extractor.extract_with_positions(pdf_path, extract_text=False, extract_render_img=True)
        results = []

        for page_num, img in enumerate(images):

            h, w, _ = img.shape
            new_size = 1080 / min(h, w)
            new_w = int(w * new_size)
            new_h = int(h * new_size)

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ocr_results = self.text_extractor.ocr_extract(img)
            ocr_results = self.restore_ocr_coordinates(ocr_results, new_size)

            page_data = {
                'page_number': page_num + 1,
                'text': [],
                'source': 'pdf_ocr'
            }

            page_data = self.text_extractor.add_ocr_to_page_text(page_data, ocr_results)
            if config.layout_detection.enabled:
                layout_location = self.layout_detector.detect(img.copy())
                sorted_texts = self.text_extractor.resort_page_text_with_layout(page_data['text'], page_num, layout_location)
            else:
                sorted_texts = self.text_extractor.resort_page_text_with_center_location(page_data['text'], page_num)

            image_base64 = self._image_to_base64(img)

            page_result = {'text': sorted_texts, 'image': image_base64}

            results.append(page_result)

        return results

    def _process_text(self, text_path: str) -> List[Dict[str, Any]]:
        """Process text file"""
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return [{'text': text}]

    def _blackout_text(self, image: np.ndarray, page_text: List[Dict], color=(0, 0, 0)) -> np.ndarray:
        """Black out text regions in image"""

        for item in page_text:
            bbox = item['bbox']
            x0, y0, x1, y1 = [int(coord) for coord in bbox]
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
        return image
