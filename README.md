# LaForge

> This is a personal project that is under heavy development. It could, and likely does, contain bugs, incomplete code,
> or other unintended issues. As such, the software is provided as-is, without warranty of any kind.

This project utilizes [EasyOCR](https://github.com/JaidedAI/EasyOCR) to facilitate batch processing of image data.

![Geordi](static/geordi.gif)

## Setup
```bash
git clone https://github.com/nos-tromo/LaForge.git
cd LaForge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Basic usage:
```bash
python laforge.py [lang] [path/to/file]
```
To detect multiple languages:
```bash
python laforge.py [lang_1,lang_2...lang_n] [path/to/data]
```
Batch processing:
```bash
python laforge.py [lang] [directory]
```
File output is stored under `output`.

## Examples
```bash
# single processing
python laforge.py en data/image.png
python laforge.py es,fr data/menu.jpg 
# batch processing
python laforge.py de fotos
python laforge.py ch_sim docs 
```