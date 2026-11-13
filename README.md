# Plant Disease Classification Using Deep Learning

A deep learning project that classifies plant leaf diseases using a custom Convolutional Neural Network (CNN) and ResNet18 transfer learning. The best-performing model is integrated into a FastAPI application and packaged using Docker for local deployment.

## Project Overview

Plant diseases can reduce crop quality and agricultural productivity. Identifying diseases early can help farmers and agricultural professionals take timely action.

This project uses computer vision and deep learning to classify plant leaf images into 15 disease and healthy-leaf categories.

**Key highlights**
- Image classification across 15 plant disease categories.
- Custom CNN model built using PyTorch.
- ResNet18 transfer learning using pretrained weights.
- Model evaluation using accuracy, precision, recall and F1-score.
- FastAPI prediction endpoint.
- Docker-based application deployment.
- CPU-based model inference.

## Model Performance

Both models were evaluated on the same independent test set of 2,064 images.

| Evaluation Metric | Custom CNN | ResNet18 |
|---|---:|---:|
| Test Accuracy | 91.62% | **93.02%** |
| Macro F1-score | 0.90 | **0.92** |
| Weighted F1-score | 0.92 | **0.93** |

ResNet18 achieved a 1.40 percentage-point improvement in test accuracy over the custom CNN.

**Selected model: ResNet18**

## Dataset

- Dataset: PlantVillage
- Source: [Kaggle – PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Total images: 20,638
- Number of classes: 15
- Image format: JPG, JPEG and PNG
- Input image size for model: 224 × 224 pixels
- Color channels: RGB

### Dataset Split

| Dataset | Images |
|---|---:|
| Training | 16,510 |
| Validation | 2,064 |
| Testing | 2,064 |
| **Total** | **20,638** |

The dataset was split into training, validation and test sets. The dataset itself is not included in this GitHub repository.

### Supported Classes

1. Pepper bell bacterial spot
2. Pepper bell healthy
3. Potato early blight
4. Potato late blight
5. Potato healthy
6. Tomato bacterial spot
7. Tomato early blight
8. Tomato late blight
9. Tomato leaf mold
10. Tomato Septoria leaf spot
11. Tomato spider mites
12. Tomato target spot
13. Tomato yellow leaf curl virus
14. Tomato mosaic virus
15. Tomato healthy

## Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch, Torchvision |
| Computer Vision | Pillow |
| Data Processing | NumPy, Pandas |
| Machine Learning Utilities | Scikit-learn |
| API Development | FastAPI |
| API Server | Uvicorn |
| Containerization | Docker |
| Version Control | Git, GitHub |

## Project Architecture

```text
Plant-Disease-Classification/
│
├── api/
│   └── main.py
│
├── models/
│   ├── best_cnn_model.pth
│   └── best_resnet18_model.pth
│
├── src/
│   ├── cnn_model.py
│   ├── resnet_model.py
│   ├── train_cnn.py
│   ├── train_resnet.py
│   ├── predict.py
│   ├── data_pipeline.py
│   ├── create_split.py
│   ├── validate_split.py
│   ├── evaluate_cnn.py
│   ├── evaluate_resnet.py
│   ├── error_analysis.py
│   └── dataset_summary.py
│
├── notebooks/
├── reports/
├── Dockerfile
├── .dockerignore
├── .gitignore
└── requirements.txt
```

## Model Development Workflow

1. Dataset collection and image inspection.
2. Data validation and preprocessing.
3. Training, validation and test dataset splitting.
4. Custom CNN model development.
5. ResNet18 transfer learning.
6. Model evaluation and comparison.
7. Error analysis and confusion matrix generation.
8. Saving the best-performing model.
9. FastAPI integration.
10. Docker containerization and local deployment.

## API Documentation

The trained ResNet18 model is integrated into a FastAPI application.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | API health check |
| POST | `/predict` | Predict plant disease from an uploaded image |

### Prediction Request

Upload a JPG or PNG image using the `/predict` endpoint.

Example response:

```json
{
  "status": "success",
  "filename": "plant_leaf.jpg",
  "predicted_disease": "Potato___healthy",
  "confidence": 0.9735
}
```

The confidence value is the model's softmax score. It should not be interpreted as a guarantee that the prediction is correct.

### Run API Locally

Clone the repository:

```bash
git clone https://github.com/mrpraveen731-cmd/Plant-Disease-Classification.git
cd Plant-Disease-Classification
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Open the Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Docker Deployment

The application is containerized using Docker.

### Build Docker Image

```bash
docker build -t plant-disease-api:v1 .
```

### Run Docker Container

```bash
docker run -d --name plant-disease-container -p 8000:8000 plant-disease-api:v1
```

### Access API

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

### Stop Container

```bash
docker stop plant-disease-container
```

## Limitations

- The model is trained using the PlantVillage dataset, which contains relatively controlled leaf images.
- Real-world images may have different lighting, backgrounds, camera quality and disease symptoms.
- Some classes have fewer test images than others. For example, Potato healthy has only 15 test samples.
- High model confidence does not guarantee a correct prediction.
- The current application performs image classification, not object detection or disease severity estimation.
- The API is deployed locally using Docker; public cloud deployment is not included.

## Future Improvements

- Evaluate the model using real-world field images.
- Improve generalization through additional datasets and augmentation.
- Explore model optimization and inference performance.
- Build a user-friendly frontend.
- Deploy the application to a cloud platform.
- Add monitoring, logging and prediction history.

## Author

**Praveen S.**

B.Sc. Statistics | Aspiring AI/ML Engineer

- GitHub: https://github.com/mrpraveen731-cmd
- Project Repository: https://github.com/mrpraveen731-cmd/Plant-Disease-Classification

---

*This project was developed as an academic capstone project to demonstrate practical skills in deep learning, computer vision, model evaluation, API development and Docker-based deployment.*