# Dataset Formats

## Default Data Directories
Data is stored in the following locations (defined in `constants.py`):
- ChartQA: `CHARTQA_DIR` - Chart question-answering dataset
- VQA-v2: `VQA_V2_DIR` - Visual question-answering dataset
- MMMU: `MMMU_DIR` - Multimodal university benchmark

## ChartQA Dataset
- Source: https://github.com/vis-nlp/ChartQA
- Structure:
  ```json
  {
    "imgname": "00006834003066.png",  // Image filename in png directory
    "query": "What is the value?",     // Question text
    "label": "42"                      // Answer text
  }
  ```
- Directory structure:
  ```
  CHARTQA_DIR/
  ├── train/
  │   ├── png/
  │   └── train_augmented.json
  ├── val/
  │   ├── png/
  │   └── val_augmented.json
  └── test/
      ├── png/
      └── test_augmented.json
  ```

## VQA-v2 Dataset (via ParlAI)
- Source: https://visualqa.org/ (accessed through ParlAI)
- ParlAI format:
  ```python
  {
    'text': "What color is the cat?",  # Question
    'labels': ["black", "dark", "black and white"],  # Multiple ground truth answers
    'image': "path/to/image.jpg",  # Path to image file
    'episode_done': True  # ParlAI episode marker
  }
  ```
- Splits:
  - train -> 'train'
  - validation -> 'valid'
  - test -> 'test'
- Note: ParlAI handles data downloading and preprocessing automatically

## MMMU Dataset
- Source: https://huggingface.co/datasets/MMMU/MMMU
- Structure:
  ```python
  {
    'question': "Describe the image...",
    'answer': "The image shows...",
    'image_url': "https://...",  # Direct URL to image
    'subject': "Computer_Science"
  }
  ```
- Splits:
  - train -> 'dev'
  - validation -> 'validation'
  - test -> 'test'
- Long-form answers: Responses with >15 words or containing newlines 