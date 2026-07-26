#!/usr/bin/env python3
"""
Text extractor
Supports extracting text from PDF, DOC, DOCX, images, etc.
"""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from smartresume.data.ocr_provider import ocr_provider_manager


class TextExtractor:
    """Text extractor class"""

    def __init__(self, init_ocr: bool = True):
        """
        Initialize text extractor

        Args:
            init_ocr: Whether to initialize OCR
        """
        self.ocr = None

        if init_ocr:
            try:
                self.ocr = ocr_provider_manager

            except Exception:
                pass

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from file.

        Args:
            file_path: File path

        Returns:
            List[Dict]: Per-page text data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        try:
            if file_ext == '.pdf':
                return self._extract_from_pdf(str(file_path))
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return self._extract_from_image(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

        except Exception:
            raise

    def extract_with_positions(self, pdf_path: str, extract_text: bool = True,
                               extract_render_img: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Extract text and positions from PDF.

        Args:
            pdf_path: PDF file path

        Returns:
            List[List[Dict]]: Per-page text blocks with positions
        """
        if pdfplumber is None:
            raise ImportError("PDFplumber is not installed; cannot process PDF files")

        pages_data = []
        images = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                from smartresume.utils.config import config
                total_pages = len(pdf.pages)
                max_pages = min(total_pages, config.processing.page_limit)

                if extract_text:
                    for page_num in range(max_pages):
                        page = pdf.pages[page_num]

                        page_texts = []
                        page_images = []

                        # Extract text with positions
                        words = page.extract_words()
                        for word in words:
                            if word['text'].strip():
                                page_texts.append({
                                    "text": word['text'].strip(),
                                    "bbox": [word['x0'], word['top'], word['x1'], word['bottom']],
                                    "confidence": 1.0,
                                    "source": "pdf_text"
                                })

                        # Extract images (if any)
                        if hasattr(page, 'images'):
                            for img in page.images:
                                page_images.append([img['x0'], img['top'], img['x1'], img['bottom']])

                        pages_data.append(page_texts)

                if extract_render_img:
                    for page_num in range(max_pages):
                        page = pdf.pages[page_num]
                        # Convert page to image using pdfplumber's built-in method
                        img = page.to_image()
                        if img:
                            images.append(np.array(img.original))

        except Exception:
            raise

        return pages_data, images

    def extract_from_pdf_string(self, pdf_path: str) -> str:
        """
        Extract plain text string from PDF.

        Args:
            pdf_path: PDF file path

        Returns:
            str: Extracted text
        """
        if pdfplumber is None:
            raise ImportError("PDFplumber is not installed; cannot process PDF files")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []

                from smartresume.utils.config import config
                total_pages = len(pdf.pages)
                max_pages = min(total_pages, config.processing.page_limit)

                for page_num in range(max_pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text and text.strip():
                        text_parts.append(text.strip())

                return '\n\n'.join(text_parts)

        except Exception:
            return ""

    def ocr_extract(self, image: np.ndarray) -> List[Any]:
        """
        Perform OCR on image.

        Args:
            image: Image array

        Returns:
            List: OCR results
        """
        if not self.ocr:
            return []

        try:
            result = self.ocr.ocr_extract(image)
            return result if result else []

        except Exception:
            return []

    def resort_page_text_with_location(self, page_texts: List[Dict[str, Any]], page_num: int) -> List[Dict[str, Any]]:
        """
        Resort page texts.

        Args:
            page_texts: Page text list
            page_num: Page index

        Returns:
            List[Dict]: Sorted texts
        """
        if not page_texts:
            return []

        try:
            sorted_texts = sorted(page_texts, key=lambda x: (
                x.get('bbox', [0, 0, 0, 0])[1],
                x.get('bbox', [0, 0, 0, 0])[0]
            ))
            return sorted_texts
        except Exception:
            return page_texts

    def resort_page_text_with_center_location(self, page_texts: List[Dict[str, Any]],
                                              page_num: int) -> List[Dict[str, Any]]:
        """
        Resort page texts by center coordinates.

        Args:
            page_texts: Page text list
            page_num: Page index

        Returns:
            List[Dict]: Sorted texts
        """
        if not page_texts:
            return []

        try:
            sorted_texts = sorted(
                page_texts,
                key=lambda x: (
                    (x.get('bbox', [0, 0, 0, 0])[1] +
                     x.get('bbox', [0, 0, 0, 0])[3]) / 2,
                    (x.get('bbox', [0, 0, 0, 0])[0] +
                     x.get('bbox', [0, 0, 0, 0])[2]) / 2
                )
            )
            return sorted_texts
        except Exception:
            return page_texts

    def resort_page_text_with_layout(self, page_texts: List[Dict[str, Any]],
                                     page_num: int,
                                     layout_location: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not layout_location:
            return self.resort_page_text_with_center_location(page_texts, page_num)

        if not page_texts:
            return []

        try:
            for text in page_texts:
                tx1, ty1, tx2, ty2 = text.get('bbox', [0, 0, 0, 0])
                tx_center = (tx1 + tx2) / 2
                ty_center = (ty1 + ty2) / 2
                assigned = False

                for idx, layout in enumerate(layout_location):
                    lx1, ly1, lx2, ly2 = layout['x1'], layout['y1'], layout['x2'], layout['y2']
                    if lx1 <= tx_center <= lx2 and ly1 <= ty_center <= ly2:
                        text["x_center"] = tx_center
                        text["y_center"] = ty_center
                        text["lx_center"] = (lx1 + lx2) / 2
                        text["ly_center"] = (ly1 + ly2) / 2
                        text["layout_idx"] = idx
                        assigned = True
                        break

                if not assigned:
                    text["x_center"] = tx_center
                    text["y_center"] = ty_center
                    text["lx_center"] = tx_center
                    text["ly_center"] = ty_center
                    text["layout_idx"] = -1

            for idx, layout in enumerate(layout_location):
                layout_texts = [text for text in page_texts if text.get("layout_idx") == idx]
                if not layout_texts:
                    continue

                lx1, ly1, lx2, ly2 = layout['x1'], layout['y1'], layout['x2'], layout['y2']
                layout_area = (lx2 - lx1) * (ly2 - ly1)

                text_total_area = sum(
                    (text['bbox'][2] - text['bbox'][0]) * (text['bbox'][3] - text['bbox'][1])
                    for text in layout_texts
                )
                ratio = text_total_area / layout_area

                if ratio < 0.075:
                    for text in layout_texts:
                        text["lx_center"] = text["x_center"]
                        text["ly_center"] = text["y_center"]
                        text["layout_idx"] = -1

            active_layouts = []
            for idx, layout in enumerate(layout_location):
                if any(text.get("layout_idx") == idx for text in page_texts):
                    active_layouts.append((idx, layout))

            unassigned_texts = [text for text in page_texts if text.get("layout_idx") == -1]
            for text in unassigned_texts:
                tx_center = text["x_center"]
                ty_center = text["y_center"]
                min_distance = float('inf')
                closest_idx = -1
                closest_center = (tx_center, ty_center)

                for layout_idx, layout in active_layouts:
                    lx1, ly1, lx2, ly2 = layout['x1'], layout['y1'], layout['x2'], layout['y2']
                    if ty_center < ly2:
                        dy = ly2 - ty_center
                    elif ty_center > ly1:
                        dy = ty_center - ly1
                    else:
                        dy = min(abs(ty_center - ly1), abs(ty_center - ly2))

                    if tx_center < lx1:
                        dx = lx1 - tx_center
                    elif tx_center > lx2:
                        dx = tx_center - lx2
                    else:
                        dx = min(abs(tx_center - lx1), abs(tx_center - lx2))

                    dist = min(dx, dy)
                    if dist < min_distance:
                        min_distance = dist
                        closest_idx = layout_idx
                        closest_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)

                if closest_idx != -1:
                    text["layout_idx"] = closest_idx
                    text["lx_center"] = closest_center[0]
                    text["ly_center"] = closest_center[1]

            sorted_texts = sorted(
                page_texts,
                key=lambda x: (
                    x.get("ly_center", 0),
                    x.get("lx_center", 0),
                    x.get("y_center", 0),
                    x.get("x_center", 0),
                )
            )
            return sorted_texts
        except Exception:
            return page_texts

    def _extract_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF file"""
        if pdfplumber is None:
            raise ImportError("PDFplumber is not installed; cannot process PDF files")

        pages_data = []

        try:
            with pdfplumber.open(file_path) as pdf:
                from smartresume.utils.config import config
                total_pages = len(pdf.pages)
                max_pages = min(total_pages, config.processing.page_limit)

                for page_num in range(max_pages):
                    page = pdf.pages[page_num]

                    text = page.extract_text()

                    page_data = {
                        'page_number': page_num + 1,
                        'text': [{'text': text, 'confidence': 1.0}] if text and text.strip() else [],
                        'source': 'pdf_text'
                    }

                    # If text is empty or very short, try OCR
                    if len(text.strip()) < 50 and self.ocr:
                        try:
                            # Convert page to image for OCR
                            img = page.to_image()
                            if img:
                                img_array = np.array(img.original)
                                ocr_result = self.ocr.ocr_extract(img_array)
                                if ocr_result:
                                    page_data = self.add_ocr_to_page_text(
                                        page_data, ocr_result)

                        except Exception:
                            pass

                    pages_data.append(page_data)

        except Exception:
            raise

        return pages_data

    def _extract_from_image(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from image file"""
        if not self.ocr:
            raise RuntimeError("OCR is not initialized; cannot process image files")

        try:
            with open(file_path, 'rb') as f:
                img_data = f.read()

            ocr_result = self.ocr.ocr(img_data)

            page_data = {
                'page_number': 1,
                'text': [],
                'source': 'image_ocr'
            }

            if ocr_result:
                page_data = self.add_ocr_to_page_text(page_data, ocr_result)

            return [page_data]

        except Exception:
            raise

    def add_ocr_to_page_text(self, page_data: Dict[str, Any], ocr_result: List[Any]) -> Dict[str, Any]:
        """
        Add OCR results into page text data.

        Args:
            page_data: Page data
            ocr_result: OCR results

        Returns:
            Dict: Updated page data
        """
        try:
            ocr_texts = []

            for item in ocr_result:
                try:
                    if (len(item) >= 2 and isinstance(item[1], list) and len(item[1]) >= 2):

                        xs = [p[0] for p in item[0][0]]
                        ys = [p[1] for p in item[0][0]]

                        x_min = min(xs)
                        x_max = max(xs)
                        y_min = min(ys)
                        y_max = max(ys)

                        bbox = [x_min, y_min, x_max, y_max]

                        text = item[1][0]
                        confidence = float(item[1][1])

                        if text and text.strip():
                            ocr_texts.append({
                                'bbox': bbox,
                                'text': text.strip(),
                                'confidence': confidence
                            })
                except (IndexError, ValueError, TypeError):
                    continue

            if ocr_texts:
                # If there are existing texts, extend; otherwise replace
                if page_data['text']:
                    page_data['text'].extend(ocr_texts)
                else:
                    page_data['text'] = ocr_texts

                page_data['source'] = f"{page_data['source']}_with_ocr"

            return page_data

        except Exception:
            return page_data

    def get_text_content(self, pages_data: List[Dict[str, Any]]) -> str:
        """
        Extract plain text from page data.

        Args:
            pages_data: List of page data

        Returns:
            str: Concatenated text content
        """
        text_parts = []

        for page_data in pages_data:
            page_texts = []

            if 'text' in page_data and isinstance(page_data['text'], list):
                for text_item in page_data['text']:
                    if isinstance(text_item, dict) and 'text' in text_item:
                        text = text_item['text']
                        if text and text.strip():
                            page_texts.append(text.strip())
                    elif isinstance(text_item, str) and text_item.strip():
                        page_texts.append(text_item.strip())

            if page_texts:
                text_parts.append('\n'.join(page_texts))

        return '\n\n'.join(text_parts)
