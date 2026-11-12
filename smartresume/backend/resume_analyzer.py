"""
Main analyzer class
"""
from typing import List, Dict, Any, Optional, Tuple
from smartresume.data import DataProcessor, FileProcessor
from smartresume.data.file_processor import OcrRequired
from smartresume.data.text_extractor import TextExtractor
from smartresume.model.llm_client import LLMClient
from smartresume.utils.config import config


class ResumeAnalyzer:
    """Resume analyzer that integrates all modules"""

    def __init__(self, init_ocr: bool = True, init_llm: bool = True) -> None:
        """
        Initialize analyzer

        Args:
            init_ocr: Whether to initialize OCR functionality
            init_llm: Whether to initialize LLM
        """
        self.text_extractor = TextExtractor(init_ocr)
        self.file_processor = FileProcessor(self.text_extractor)
        self.llm_client = LLMClient() if init_llm else None
        self.data_processor = DataProcessor()

    def pipeline(self,
                 cv_path: str,
                 resume_id: str,
                 extract_types: List[str] = None) -> Dict[str, Any]:
        """
        Complete resume analysis pipeline.

        Args:
            cv_path: Resume file path
            resume_id: Resume ID
            extract_types: Extraction types

        Returns:
            Analysis result
        """
        trace = self.pipeline_with_trace(
            cv_path=cv_path,
            resume_id=resume_id,
            extract_types=extract_types,
        )
        return trace.get("final_result", {})

    @staticmethod
    def _strip_page_images(processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return processed page data without embedded base64 images."""
        # Only the "image" key is dropped and the trace is only serialized, so a
        # shallow copy is enough (a deepcopy re-copies every text box on every page).
        return [
            {key: value for key, value in page_data.items() if key != "image"}
            for page_data in processed_data
        ]

    @staticmethod
    def _error_result(error: Exception, resume_id: str, cv_path: str) -> Dict[str, Any]:
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "resume_id": resume_id,
            "file_path": cv_path
        }
        return {"error": str(error), "error_details": error_details}

    def prepare_with_trace(self,
                           cv_path: str,
                           resume_id: str,
                           extract_types: List[str] = None,
                           allow_ocr: bool = True) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
        """
        Pipeline stage 1: read the file (PDF text / OCR / layout) and build the LLM input.

        Args:
            allow_ocr: False reads only files that need no OCR. For one that does,
                processed_data is None and trace["needs_ocr"] is True (no error is
                set); call again with allow_ocr=True where a GPU is available.

        Returns:
            (trace, processed_data). On failure processed_data is None and
            trace["final_result"] holds the error, as in pipeline_with_trace.
        """
        if extract_types is None:
            extract_types = ["basic_info"]

        trace: Dict[str, Any] = {
            "cv_path": cv_path,
            "resume_id": resume_id,
            "extract_types": extract_types,
        }

        try:
            processed_data = self.file_processor.process_file(cv_path, allow_ocr=allow_ocr)
            trace["processed_pages"] = self._strip_page_images(processed_data)

            text_lines, text_content, indexed_text_content = (
                self.data_processor.build_text_content(processed_data))
            trace["text_lines"] = text_lines
            trace["text_content"] = text_content
            trace["indexed_text_content"] = indexed_text_content
            return trace, processed_data

        except OcrRequired:
            trace["needs_ocr"] = True
            return trace, None
        except Exception as e:
            trace["final_result"] = self._error_result(e, resume_id, cv_path)
            return trace, None

    def finish_with_trace(self,
                          trace: Dict[str, Any],
                          processed_data: List[Dict[str, Any]],
                          structure_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pipeline stage 3: post-process the LLM output into trace["final_result"].
        Stage 2 is llm_client.extract_info / extract_info_batch.
        """
        try:
            trace["structure_output"] = structure_output
            trace["final_result"] = self.data_processor.post_process(
                trace["text_lines"], structure_output, processed_data)
            self.data_processor.merge_pdf_links(trace["final_result"], trace["cv_path"])
        except Exception as e:
            trace["final_result"] = self._error_result(e, trace["resume_id"], trace["cv_path"])
        return trace

    def pipeline_with_trace(self,
                            cv_path: str,
                            resume_id: str,
                            extract_types: List[str] = None) -> Dict[str, Any]:
        """
        Complete resume analysis pipeline with intermediate trace data.

        Returns:
            Dict containing intermediate artifacts and final_result.
        """
        trace, processed_data = self.prepare_with_trace(cv_path, resume_id, extract_types)
        if processed_data is None:
            return trace

        try:
            structure_output = self.llm_client.extract_info(
                trace["indexed_text_content"], trace["extract_types"], resume_id)
        except Exception as e:
            trace["final_result"] = self._error_result(e, resume_id, cv_path)
            return trace

        return self.finish_with_trace(trace, processed_data, structure_output)

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single file.

        Args:
            file_path: File path

        Returns:
            Result of processing
        """
        if not self.file_processor:
            raise ValueError("File processor is not initialized")

        return self.file_processor.process_file(file_path)

    def extract_info_only(self, text_content: str, extract_types: List[str], resume_id: str) -> Dict[str, Any]:
        """
        Extract information only (no file processing).

        Args:
            text_content: Text content
            extract_types: Extraction types
            resume_id: Resume ID

        Returns:
            Extraction result
        """
        if not self.llm_client:
            raise ValueError("LLM client is not initialized")

        try:
            result = self.llm_client.extract_info(text_content, extract_types, resume_id)
            return result
        except Exception:
            raise

    def update_config(self, **kwargs) -> None:
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(config.processing, key):
                setattr(config.processing, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)


# Create default analyzer factory
def create_analyzer(init_ocr: bool = True, init_llm: bool = True) -> ResumeAnalyzer:
    """Create analyzer instance"""
    return ResumeAnalyzer(init_ocr, init_llm)
