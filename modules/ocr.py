import logging
import os
from pathlib import Path

import easyocr


class OCRVisor:
    """
    A class to perform OCR detection on image files using EasyOCR.

    Attributes:
    -----------
    languages : list
        A list of language codes for OCR detection.
    input_dir : str
        The path to the image file or directory to be processed.
    output_dir : str
        The directory where OCR results will be saved.

    Methods:
    --------
    __init__(languages: list, input_dir: str, output_dir: str = 'output')
        Initializes OCRDetection with languages, input file or directory, and output directory.

    _load_reader(use_gpu: bool = True) -> None
        Loads the EasyOCR Reader with specified languages. Attempts GPU usage, falls back to CPU on error.

    _setup_directories(input_dir: str, output_dir: str) -> None
        Sets up the input and output directories for OCR processing.

    _convert_to_text(data: list) -> str
        Converts the OCR result list into a single string of detected text.

    _read_file(file_path: str) -> str
        Reads and processes a single image file for OCR detection.

    _save_output(data: str, output_filepath: Path) -> None
        Saves OCR-detected text to a file in the output directory.

    _process_single_file(file_path: str) -> None
        Processes a single image file and saves the OCR result.

    data_pipeline() -> None
        Processes all image files in the input directory for OCR detection and saves the results.
    """
    def __init__(self, languages: list, input_dir: str, output_dir: Path = 'output'):
        """
        Initializes OCRDetection with languages, input file or directory, and output directory.

        :param languages: A list of languages to be used by EasyOCR.
        :param input_dir: Path to the image file or directory for OCR processing.
        :param output_dir: Directory where the OCR results will be saved (default is 'output').
        :raises ValueError: If no languages are specified.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        if not languages:
            raise ValueError("At least one language must be specified.")
        self.languages = languages
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._load_reader()

    def _load_reader(self, use_gpu: bool = True) -> None:
        """
        Loads the EasyOCR Reader with specified languages. Attempts GPU usage, falls back to CPU on error.

        :param use_gpu: If True, the reader will use GPU (default is True).
        """
        try:
            self.reader = easyocr.Reader(self.languages, gpu=use_gpu)
        except Exception as e:
            self.logger.error(f"Error loading EasyOCR Reader with GPU: {e}. Falling back to CPU.", exc_info=True)
            self.reader = easyocr.Reader(self.languages, gpu=False)

    @staticmethod
    def _convert_to_text(data: list) -> str:
        """
        Converts the OCR result list into a single string of detected text.

        :param data: A list of OCR-detected text data.
        :type data: list
        :return: The extracted text, or a message indicating no text was detected.
        :rtype: str
        """
        if not data:
            return "No text detected."
        return '\n'.join([item[1] for item in data])

    def _read_file(self, file_path: str) -> str:
        """
        Reads and processes a single image file for OCR detection.

        :param file_path: The path to the image file to be processed.
        :return: The OCR-detected text from the image.
        :rtype: str
        :raises FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        try:
            result = self.reader.readtext(file_path)
            return self._convert_to_text(result)
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return ''

    def _save_output(self, data: str, output_filepath: Path) -> None:
        """
        Saves OCR-detected text to a file in the output directory.

        :param data: The text data to be saved.
        :param output_filepath: The full file path where the output should be saved.
        :type output_filepath: Path
        :raises IOError: If there is an error writing to the file.
        """
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(data)
            self.logger.info(f"Saved output to: {output_filepath}")
        except IOError as e:
            self.logger.error(f"Error writing to file {output_filepath}: {e}", exc_info=True)

    def _process_single_file(self, file_path: str) -> None:
        """
        Processes a single image file and saves the OCR result.

        :param file_path: The path to the image file to be processed.
        """
        text = self._read_file(file_path)
        file_name = Path(file_path).stem
        output_filepath = self.output_dir / f'ocr_{file_name}.txt'
        self._save_output(text, output_filepath)

    def data_pipeline(self) -> None:
        """
        Processes all image files in the input directory for OCR detection and saves the results.

        :raises ValueError: If the input directory is neither a valid file nor a directory.
        """
        path = Path(self.input_dir)

        if path.is_file():  # If it's a single file, process it
            self.logger.info(f"Processing file: {self.input_dir}")
            self._process_single_file(self.input_dir)
            self.logger.info("Finished processing 1 file.")
        elif path.is_dir():  # If it's a directory, process all files in the directory
            self.logger.info(f"Processing files in directory: {self.input_dir}")

            # Count the total number of valid files
            valid_files = [file for file in path.glob('*') if file.is_file() and
                           file.suffix in ['.bmp', '.jpeg', '.jpg', '.png', '.tiff']]
            total_files = len(valid_files)

            if total_files == 0:
                self.logger.info("No valid image files found in directory.")
                return

            counter = 0
            for file in valid_files:
                counter += 1
                self.logger.info(f"Processing file #{counter}: {file}")
                self._process_single_file(str(file))
                self.logger.info(f"Completed processing file #{counter}: {file}")

            self.logger.info(f"Finished processing {counter} of {total_files} files in directory.")
        else:
            raise ValueError(f"{self.input_dir} is neither a valid file nor a directory.")
