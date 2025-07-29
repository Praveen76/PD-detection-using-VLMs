# Parkinson's Disease Detection using Vision-Language Models (VLMs)

This project explores the use of pretrained Vision-Language Models (VLMs) for **zero-shot Parkinson's Disease (PD) detection** from clinical video data. By leveraging multimodal prompts and structured visual features, the system evaluates motor symptoms across 325+ patient videos—without requiring any model fine-tuning.

## 📌 Highlights

- **Zero-Shot Evaluation** using 12 carefully designed diagnostic prompts.
- **325+ Clinical Videos** covering various PD-related motor patterns.
- **Modular Pipeline** for video selection, embedding extraction, surrogate feature computation, and evaluation.
- Built with **PyTorch**, **OpenCLIP**, and **DeepSpeed** for scalable inference.

---

## 📁 Project Structure

```

├── Annotations/              # Label files and structured QA data
├── Dataset/                  # Raw and processed video files
├── Download\_Voxceleb/        # Utility to fetch VoxCeleb data (if needed)
├── EmbeddingEvaluation/      # Evaluation scripts for single/multi-view embeddings
│   └── SingleViewEmbedding/
├── Experiments/              # All experiments using different VLMs
├── Metadata/                 # Patient meta-info and clinical context
├── SmileFeatures/            # Smile-related feature extraction
│   └── Evaluation/
├── SurrogateFeatures/        # Temporal and spatial surrogate metrics (e.g., velocity, angle)
├── Utils/                    # Common utilities
├── VideoEmbeddings/          # Precomputed CLIP/VL embeddings
├── VideoSelection/           # Filtering and selection logic
├── videoSamples/ViViT\_v02/   # ViViT-style samples for benchmarking
├── practice/                 # Temporary experiments / notebooks
├── remaining\_annotations.txt # Remaining manual label entries
├── LICENSE
├── README.md
├── pyrightconfig.json
└── requirements.txt          # Environment dependencies

````

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/Praveen76/PD-detection-using-VLMs.git
cd PD-detection-using-VLMs
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Data & Embeddings

Place video data under `Dataset/`, annotations under `Annotations/`, and ensure embeddings are generated or loaded into `VideoEmbeddings/`.

---

## 📊 Evaluation

Run the evaluation using the zero-shot prompts defined in the pipeline:

```bash
python EmbeddingEvaluation/SingleViewEmbedding/eval_pipeline.py
```

Prompts are encoded using CLIP, and scores are aggregated over frames and prompts.

---

## 🧠 Model & Approach

* Uses **pretrained CLIP-based VLMs** for visual-textual embedding.
* Applies **12 structured prompts** targeting PD symptoms like tremor, rigidity, and facial expression.
* Embedding similarity used as proxy for class confidence.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.
now if you'd like to include figures, paper links, or citation blocks. I can adapt it to academic, demo, or production tone as needed.

