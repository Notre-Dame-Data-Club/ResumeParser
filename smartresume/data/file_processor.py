"""
File processor module
"""
import os
import cv2
import string
import threading
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from smartresume.data.text_extractor import TextExtractor, pdfplumber
from smartresume.utils.config import config
from smartresume.data.layout_detector import LayoutDetector

## A page with fewer characters in its text layer is treated as scanned
_MIN_TEXT_CHARS = 50
## An embedded picture at least this share of the page may hold text worth OCR
_MIN_IMAGE_AREA_SHARE = 0.01
## This many vector curves on a page suggests text exported as outlines
_MIN_OUTLINE_CURVES = 150


class OcrRequired(Exception):
    """The file needs OCR and the caller asked not to run it (process_file allow_ocr=False)."""


class FileProcessor:
    """File processor responsible for handling different file formats"""

    def __init__(self, text_extractor: TextExtractor):
        self.text_extractor = text_extractor

        self.layout_detector: Optional[LayoutDetector] = None
        self._layout_pid = os.getpid()
        self._layout_lock = threading.Lock()
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

    def process_file(self, file_path: str, allow_ocr: bool = True) -> List[Dict[str, Any]]:
        """
        Process a file, choosing the appropriate method based on type.

        allow_ocr=False raises OcrRequired instead of running OCR, so a caller without
        a GPU at hand (the CPU half of a ZeroGPU Space) can hand the file to one.
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            if not allow_ocr:
                raise OcrRequired(file_path)
            return self._process_image(file_path)
        elif file_ext == '.pdf':
            return self._process_pdf(file_path, allow_ocr)
        elif file_ext in ['.txt', '.md']:
            return self._process_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported: PDF, images (.jpg/.jpeg/.png/.tiff/.bmp), text (.txt/.md)"
            )

    def _layout(self) -> Optional[LayoutDetector]:
        """
        The layout detector of this process, or None when layout detection is off.

        An ONNX Runtime session does not survive a fork (its worker threads stay in
        the parent and the first run() in the child hangs), so a forked worker, e.g.
        a ZeroGPU one, builds its own.
        """
        if not config.layout_detection.enabled:
            return None
        if self.layout_detector is None or self._layout_pid != os.getpid():
            with self._layout_lock:
                if self.layout_detector is None or self._layout_pid != os.getpid():
                    self.layout_detector = LayoutDetector()
                    self._layout_pid = os.getpid()
        return self.layout_detector

    def _sort_page_text(self, texts: List[Dict[str, Any]], page_num: int, image: np.ndarray) -> List[Dict[str, Any]]:
        """Reading order: by layout block when layout detection is on, else top to bottom."""
        detector = self._layout()
        if detector is not None:
            layout_location = detector.detect(image)
            return self.text_extractor.resort_page_text_with_layout(texts, page_num, layout_location)
        return self.text_extractor.resort_page_text_with_center_location(texts, page_num)

    def _process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Process image file"""
        original = cv2.imread(image_path)
        h, w, _ = original.shape

        scale = max(1080 / min(h, w), 1)
        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(original, (new_w, new_h))

        ocr_results = self.text_extractor.ocr_extract(image)
        ocr_results = self.restore_ocr_coordinates(ocr_results, scale)

        page_data = {
            'page_number': 1,
            'text': [],
            'source': 'image_ocr'
        }

        page_data = self.text_extractor.add_ocr_to_page_text(page_data, ocr_results)
        ## OCR boxes are mapped back to the original size, so the layout is read there too
        sorted_results = self._sort_page_text(page_data['text'], 0, original)

        return [{'text': sorted_results, 'image': None}]

    # -- PDF -- #
    def _read_pdf(self, pdf_path: str, allow_ocr: bool) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Read a PDF in one pass: per page the words with positions, the plain text,
        whether OCR could add anything, and the rendered page when a later step needs it.

        Returns (mode, pages); mode is "raw_text", "ocr_only" or "text_with_ocr".
        The file is opened once: text, words and image boxes all come from the same
        parsed page (pdfplumber caches it), and pages are only rendered for layout
        detection or OCR.
        """
        if pdfplumber is None:
            raise ImportError("PDFplumber is not installed; cannot process PDF files")

        processing = config.processing
        pages: List[Dict[str, Any]] = []
        with pdfplumber.open(pdf_path) as pdf:
            pdf_pages = pdf.pages[:max(1, processing.page_limit)]

            if processing.use_force_ocr:
                mode = "ocr_only"
            else:
                for page in pdf_pages:
                    words = [
                        {"text": word['text'].strip(),
                         "bbox": [word['x0'], word['top'], word['x1'], word['bottom']],
                         "confidence": 1.0,
                         "source": "pdf_text"}
                        for word in page.extract_words() if word['text'].strip()
                    ]
                    pages.append({
                        "words": words,
                        "plain": (page.extract_text() or "").strip(),
                        "ocr_target": self._has_ocr_target(page, words),
                    })
                text = "\n\n".join(p["plain"] for p in pages if p["plain"])
                if self._garbled_ratio(text) > 0.15:
                    mode = "ocr_only"   # scanned (no text layer) or a broken font map
                elif processing.use_pdf_raw_text and text.strip():
                    mode = "raw_text"
                else:
                    mode = "text_with_ocr"

            if mode == "ocr_only":
                pages = [{"words": [], "plain": "", "ocr_target": True} for _ in pdf_pages]
            elif mode == "text_with_ocr" and not processing.skip_ocr_without_images:
                for page_info in pages:
                    page_info["ocr_target"] = True

            if mode != "raw_text":
                if not allow_ocr and any(p["ocr_target"] for p in pages):
                    raise OcrRequired(pdf_path)
                needs_layout = config.layout_detection.enabled
                for page, page_info in zip(pdf_pages, pages):
                    if needs_layout or page_info["ocr_target"]:
                        page_info["image"] = np.array(page.to_image().original)
        return mode, pages

    @staticmethod
    def _has_ocr_target(page: Any, words: List[Dict[str, Any]]) -> bool:
        """
        True when OCR could find text the PDF text layer does not have.

        OCR runs on the page with its text layer blacked out, so on an ordinary
        text PDF it has nothing to read and its detection pass is pure cost. It is
        only worth running when the page embeds a picture big enough to hold text
        (a scanned block, a banner, a screenshot), draws a lot of vector outlines
        (text exported as curves), or has almost no text layer at all.
        """
        if sum(len(word["text"]) for word in words) < _MIN_TEXT_CHARS:
            return True
        page_area = float(page.width) * float(page.height) or 1.0
        for image in page.images:
            width = float(image['x1']) - float(image['x0'])
            height = float(image['bottom']) - float(image['top'])
            ## Icons (phone, email, LinkedIn glyphs) are small squares; skip those
            if width * height >= _MIN_IMAGE_AREA_SHARE * page_area or (width >= 100 and height >= 12):
                return True
        return len(page.curves) >= _MIN_OUTLINE_CURVES

    def _process_pdf(self, pdf_path: str, allow_ocr: bool = True) -> List[Dict[str, Any]]:
        """Process PDF file"""
        mode, pages = self._read_pdf(pdf_path, allow_ocr)
        if mode == "raw_text":
            lines = "\n\n".join(p["plain"] for p in pages if p["plain"]).split("\n")
            return [{'text': [{'text': line} for line in lines]}]
        if mode == "ocr_only":
            return self._process_pdf_ocr_only(pages)
        return self._process_pdf_with_ocr(pages)

    def _process_pdf_with_ocr(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process PDF combining text extraction and OCR (OCR only where it can add text)"""
        results = []

        for page_num, page_info in enumerate(pages):
            img = page_info.get("image")
            page_data = {
                'page_number': page_num + 1,
                'text': page_info["words"],
                'source': 'pdf_text'
            }

            if page_info["ocr_target"]:
                blacked_out_image = self._blackout_text(img.copy(), page_info["words"])

                h, w, _ = blacked_out_image.shape

                new_size = 960 / min(h, w)
                new_w = int(w * new_size)
                new_h = int(h * new_size)

                blacked_out_image = cv2.resize(blacked_out_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

                ocr_results = self.text_extractor.ocr_extract(blacked_out_image)
                ocr_results = self.restore_ocr_coordinates(ocr_results, new_size)
                page_data['source'] = 'pdf_text_with_ocr'
                page_data = self.text_extractor.add_ocr_to_page_text(page_data, ocr_results)

            sorted_texts = self._sort_page_text(page_data['text'], page_num, img)
            ## 'image' only marks that the page was rendered; the pixels are not kept
            results.append({'text': sorted_texts, 'image': None})

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

    def _process_pdf_ocr_only(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process PDF using OCR only"""
        results = []

        for page_num, page_info in enumerate(pages):
            original = page_info["image"]

            h, w, _ = original.shape
            new_size = 1080 / min(h, w)
            new_w = int(w * new_size)
            new_h = int(h * new_size)

            img = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ocr_results = self.text_extractor.ocr_extract(img)
            ocr_results = self.restore_ocr_coordinates(ocr_results, new_size)

            page_data = {
                'page_number': page_num + 1,
                'text': [],
                'source': 'pdf_ocr'
            }

            page_data = self.text_extractor.add_ocr_to_page_text(page_data, ocr_results)
            ## OCR boxes are mapped back to the original size, so the layout is read there too
            sorted_texts = self._sort_page_text(page_data['text'], page_num, original)

            results.append({'text': sorted_texts, 'image': None})

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
