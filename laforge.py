# configure root logger
import logging
from datetime import datetime
from pathlib import Path
logfile_name = f'laforge_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logs_dir = Path('logs')
logs_dir.mkdir(parents=True, exist_ok=True)
logfile_path = logs_dir / logfile_name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logfile_path),  # logging to a file
        logging.StreamHandler()  # logging to the console (stdout)
    ]
)

import sys

from modules.ocr import OCRVisor


def main() -> None:
    languages = sys.argv[1].split(",") if len(sys.argv) > 1 else ['en']  # Default to English
    data = sys.argv[2] if len(sys.argv) > 2 else input("Enter the file path: ")
    ocr = OCRVisor(languages, data)
    ocr.data_pipeline()


if __name__ == '__main__':
    main()
