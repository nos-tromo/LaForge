import logging
import os
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor


class OCRVisor:
    """
    A class to perform OCR on images and save the extracted text.

    :param languages: A list of language codes for OCR processing.
    :type languages: list
    :param input_dir: Path to the input directory or image file.
    :type input_dir: str
    :param output_dir: Path to the output directory where results will be saved.
    :type output_dir: pathlib.Path, optional
    """

    def __init__(self, languages: list, input_dir: str, output_dir: Path = 'output'):
        """
        Initialize the OCRVisor instance.

        :param languages: A list of language codes for OCR processing.
        :type languages: list
        :param input_dir: Path to the input directory or image file.
        :type input_dir: str
        :param output_dir: Path to the output directory where results will be saved.
        :type output_dir: pathlib.Path, optional
        :raises ValueError: If no languages are specified.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        if not languages:
            raise ValueError("At least one language must be specified.")
        self.languages = languages
        self.input_dir = input_dir
        self.output_dir = output_dir
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the OCR detection and recognition models along with their processors.

        :raises Exception: If an error occurs while loading the models.
        """
        try:
            self.det_processor, self.det_model = load_det_processor(), load_det_model()
            self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
        except Exception as e:
            self.logger.error(f"Error loading OCR model: {e}", exc_info=True)

    @staticmethod
    def _convert_to_text(data: list) -> str:
        """
        Convert a list of text lines into a single string.

        :param data: A list of extracted text lines.
        :type data: list
        :return: Concatenated text from all lines or a message if no text is detected.
        :rtype: str
        """
        if not data:
            return "No text detected."
        return '\n'.join([item for item in data])

    def _model_inference(self, file_path: str) -> str:
        """
        Perform OCR on a single image or PDF file.

        :param file_path: Path to the image or PDF file.
        :type file_path: str
        :return: Extracted text from the image or PDF.
        :rtype: str
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file format is not supported.
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        extracted_texts = []

        # Check if the file is an image or a PDF
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Process as an image
            try:
                image = Image.open(file_path)
                predictions = run_ocr(
                    [image],
                    [self.languages],
                    self.det_model,
                    self.det_processor,
                    self.rec_model,
                    self.rec_processor
                )
                ocr_result = predictions[0]
                text_lines = ocr_result.text_lines
                extracted_texts = [line.text for line in text_lines]
            except Exception as e:
                self.logger.error(f"Error reading image file {file_path}: {e}", exc_info=True)
                return ''

        elif file_path.lower().endswith('.pdf'):
            # Process as a PDF
            try:
                # Convert PDF pages to images
                pdf_images = convert_from_path(file_path)
                for page_num, image in enumerate(pdf_images):
                    # Run OCR on each page image
                    predictions = run_ocr(
                        [image],
                        [self.languages],
                        self.det_model,
                        self.det_processor,
                        self.rec_model,
                        self.rec_processor
                    )
                    ocr_result = predictions[0]
                    text_lines = ocr_result.text_lines
                    extracted_texts.extend([line.text for line in text_lines])
            except Exception as e:
                self.logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True)
                return ''

        else:
            self.logger.error(f"Unsupported file format: {file_path}")
            raise ValueError(f"Unsupported file format: {file_path}")

        return self._convert_to_text(extracted_texts)

    def _save_output(self, data: str, output_filepath: Path) -> None:
        """
        Save the extracted text data to a file.

        :param data: The text data to save.
        :type data: str
        :param output_filepath: The file path where the data will be saved.
        :type output_filepath: pathlib.Path
        :raises IOError: If an error occurs while writing to the file.
        """
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(data)
            self.logger.info(f"Saved output to: {output_filepath}")
        except IOError as e:
            self.logger.error(f"Error writing to file {output_filepath}: {e}", exc_info=True)

    def _process_single_file(self, file_path: str) -> None:
        """
        Process a single image file for OCR and save the result.

        :param file_path: Path to the image file.
        :type file_path: str
        """
        text = self._model_inference(file_path)
        file_name = Path(file_path).stem
        output_filepath = self.output_dir / f'ocr_{file_name}.txt'
        self._save_output(text, output_filepath)

    def data_pipeline(self) -> None:
        """
        Execute the data processing pipeline on the input directory or file.

        Processes all valid image files in the input directory or a single image file,
        performing OCR and saving the results to the output directory.

        :raises ValueError: If the input path is neither a valid file nor a directory.
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
                           file.suffix.lower() in ['.bmp', '.jpeg', '.jpg', '.png', '.tiff']]
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
