import logging
from datetime import datetime
from pathlib import Path
logfile_name = f"laforge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logs_dir = Path(".logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logfile_path = logs_dir / logfile_name
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logfile_path),
        logging.StreamHandler()
    ]
)

import sys

from modules.ocr import OCRVisor
from modules.translation import LLMAnalysis


def setup_directories(output_dir: str = "output") -> Path:
    """
    Sets up the input and output directories for OCR processing.

    :param output_dir: The path to the output directory for saving OCR results.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")
    return output_dir


def ocr_pipeline(
        languages: list,
        data: str,
        output_dir: Path
) -> None:
    ocr = OCRVisor(
        languages=languages,
        input_dir=data,
        output_dir=output_dir
    )
    ocr.data_pipeline()


def translation_pipeline(
        output_dir: Path,
        target_language: str,
        task: str
):
    llm_translator = LLMAnalysis(
        language=target_language
    )
    counter = 0
    for file in output_dir.iterdir():
        if file.is_file() and file.suffix.lower() == ".txt":
            counter += 1
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
            result = llm_translator.data_pipeline(
                task=task,
                data=text
            )
            with open(file, "a", encoding="utf-8") as f:
                f.write("\n\n\nTRANSLATION:\n\n")
                f.write(result.upper())
            logging.info(f"Translated file #{counter}: {file}")


def main() -> None:
    try:
        data = sys.argv[1] if len(sys.argv) > 2 else input("Enter the file path: ")
        languages = sys.argv[2].split(",") if len(sys.argv) > 1 else ["en"]  # set default language to english
        translate = sys.argv[3].lower() == "translate" if len(sys.argv) > 3 else False

        output_dir = setup_directories()

        ocr_pipeline(
            languages=languages,
            data=data,
            output_dir=output_dir
        )

        if translate:
            translation_pipeline(
                output_dir=output_dir,
                target_language="de",
                task="translate"
            )
        else:
            logging.info("Skipping translation.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        raise
    finally:
        logging.info("The end of our elaborate plans, the end of everything that stands.")


if __name__ == "__main__":
    main()
