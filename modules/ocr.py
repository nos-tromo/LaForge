import logging
import os
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import torch


class OCRVisor:
    """
    A class to perform OCR on images and PDF files, and save the extracted text to a specified output directory.

    Attributes:
        languages (list): A list of language codes for OCR processing.
        input_dir (str): Path to the input directory or image/PDF file.
        output_dir (Path): Path to the output directory where results will be saved.
        file_name (str): Name of the input file or directory.
        file_extensions (tuple): A tuple of valid image file extensions.

    Methods:
        __init__(languages, input_dir, output_dir): Initializes the OCRVisor object.
        _load_model(): Loads the OCR detection and recognition models.
        _convert_to_text(data): Converts a list of extracted text lines to a string.
        _model_inference(image): Runs OCR on a given image and returns text lines.
        _process_file(file_path): Processes a single image or PDF file and returns the extracted text.
        _save_output(output_filepath, data): Saves the extracted text to a specified file.
        data_pipeline(): Processes all valid files (images or PDFs) in the input directory or a single file.
    """

    def __init__(self, languages: list, input_dir: str, output_dir: Path = "output"):
        """
        Initializes the OCRVisor instance.

        :param languages: A list of language codes for OCR processing.
        :param input_dir: Path to the input directory or image/PDF file.
        :param output_dir: Path to the output directory where results will be saved.
        :raises ValueError: If no languages are provided.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        if not languages:
            raise ValueError("At least one language must be specified.")
        self.languages = languages
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_name = Path(self.input_dir).stem
        self._load_model()

        # Load valid image extensions from a file
        img_extensions_path = "config/img_extensions.txt"
        try:
            self.file_extensions = tuple(
                map(
                    lambda ext: f".{ext.strip()}",
                    open(
                        img_extensions_path,
                        encoding="utf-8"
                    ).readlines()
                )
            )
            self.logger.info("Image extensions loaded.")
        except FileNotFoundError as e:
            self.logger.error(f"No image extensions file found: {e}", exc_info=True)

    def _load_model(self) -> None:
        """
        Loads the OCR detection and recognition models along with their processors.
        This method must be called during initialization to load the required models.
        """
        try:
            self.det_processor, self.det_model = load_det_processor(), load_det_model()
            self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
        except Exception as e:
            self.logger.error(f"Error loading OCR model: {e}", exc_info=True)

    def _model_inference(self, image: Image) -> list:
        """
        Runs OCR on a given image and returns a list of extracted text lines.

        :param image: An image object to process.
        :return: A list of text lines extracted from the image.
        """
        batch_size = 10 if torch.cuda.is_available() else 5 if torch.backends.mps.is_available() else 1

        predictions = run_ocr(
            [image],
            [self.languages],
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor,
            batch_size=batch_size
        )
        ocr_result = predictions[0]
        return ocr_result.text_lines

    @staticmethod
    def _convert_to_text(data: list) -> str:
        """
        Converts a list of text lines into a single string.

        :param data: A list of extracted text lines.
        :return: Concatenated text from all lines, or a message if no text is detected.
        """
        if not data:
            return "No text detected."
        return "\n".join([item for item in data])

    def _process_file(self, file_path: str) -> str:
        """
        Processes a single image or PDF file and returns the extracted text.

        :param file_path: Path to the image or PDF file.
        :return: Extracted text from the image or PDF.
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file format is not supported.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        extracted_texts = []

        # Process image file types
        if file_path.lower().endswith(self.file_extensions):
            try:
                image = Image.open(file_path)
                text_lines = self._model_inference(image)
                extracted_texts = [line.text for line in text_lines]
            except Exception as e:
                self.logger.error(f"Error processing image file {file_path}: {e}", exc_info=True)
                return ''

        # Process PDF files
        elif file_path.lower().endswith(".pdf"):
            try:
                pdf_images = convert_from_path(file_path)
                for page_num, image in enumerate(pdf_images):
                    text_lines = self._model_inference(image)
                    extracted_texts.extend([line.text for line in text_lines])
            except Exception as e:
                self.logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True)
                return ''

        else:
            self.logger.error(f"Unsupported file format: {file_path}")
            raise ValueError(f"Unsupported file format: {file_path}")

        return self._convert_to_text(extracted_texts)

    def _save_output(self, output_filepath: Path, data: str) -> None:
        """
        Saves the extracted text data to a file.

        :param output_filepath: The file path where the data will be saved.
        :param data: The text data to save.
        :raises IOError: If an error occurs while writing to the file.
        """
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(data)
            self.logger.info(f"Saved output to: {output_filepath}")
        except IOError as e:
            self.logger.error(f"Error writing to file {output_filepath}: {e}", exc_info=True)

    def data_pipeline(self) -> None:
        """
        Executes the data processing pipeline on the input directory or file.

        Processes all valid image files in the input directory or a single image/PDF file,
        performing OCR and saving the results to the output directory.

        :raises ValueError: If the input path is neither a valid file nor a directory.
        """
        path = Path(self.input_dir)

        # Process a single file
        if path.is_file():
            self.logger.info(f"Processing file: {self.input_dir}")
            text = self._process_file(self.input_dir)
            output_filepath = self.output_dir / f"ocr_{self.file_name}.txt"
            self._save_output(output_filepath, text)
            self.logger.info(f"Finished processing file '{self.file_name}'.")

        # Process all files in a directory
        elif path.is_dir():
            self.logger.info(f"Processing files in directory: {self.input_dir}")
            valid_files = [file for file in path.glob("*") if file.is_file() and file.suffix.lower() in self.file_extensions]
            total_files = len(valid_files)

            if total_files == 0:
                self.logger.info("No valid image files found in directory.")
                return

            counter = 0
            for file in valid_files:
                counter += 1
                self.logger.info(f"Processing file #{counter}: {file}")
                text = self._process_file(file)
                output_filepath = self.output_dir / f"ocr_{file.stem}.txt"
                self._save_output(output_filepath, text)
                self.logger.info(f"Finished processing file #{counter}: {file}")

            self.logger.info(f"Finished processing {counter} of {total_files} files in directory.")

        else:
            raise ValueError(f"{self.input_dir} is neither a valid file nor a directory.")
