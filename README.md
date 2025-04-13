# 🩺 Pneumonia Detection from CXR using Transfer Learning Models
![Project Status](https://img.shields.io/badge/status-Completed-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%20%2B%20Transfer%20Learning-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

## 📌 Project Overview
This project focuses on developing a deep learning-based system to automatically detect Pneumonia from chest X-ray images. Recognizing the critical importance of early diagnosis in respiratory diseases, especially pneumonia, this project compares the performance of five state-of-the-art convolutional neural network (CNN) architectures to identify the most effective model for accurate detection.

The following models were tested and evaluated:

`DenseNet121` `ConvNeXtBase` `ResNet50V2` `ResNet101V2` `VGG16`

Each model was trained and validated on a labeled dataset of chest X-ray images, designed to distinguish between Normal and Pneumonia-infected cases. By experimenting with multiple architectures, the goal is to identify the most accurate and reliable model for real-world deployment in clinical decision support systems.

The project emphasizes:

- Comparative analysis of model performance

- Evaluation based on accuracy, precision, recall, and F1-score

- Clean codebase with individual notebooks for each model under the src/ directory

The outcome provides valuable insights into the effectiveness of different CNN architectures for medical imaging tasks, specifically pneumonia detection.

## 📂 Dataset

The dataset sourced from Kaggle. To access it click [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?raw=true)

**Note:** Due to the large size of the dataset, it is **not included in this repository**. Please download the dataset manually from Kaggle and place it in the following structure:

- Classes: `Normal`  `Pneumonia`

- Data Split: `Train` `Test`

<img src="https://github.com/ShaikhBorhanUddin/Pneumonia-Detection-Project/blob/main/data/Dataset_Visualization.png?raw=true" alt="Dashboard" width="1010"/>

## 📁 Project Structure

```bash
📂 Pneumonia-Detection-Project/
├── 📁 src/
│   ├── 📓 ConvNeXtBase_Pneumonia.ipynb
│   ├── 📓 DenseNet121_Pneumonia.ipynb
│   ├── 📓 ResNet50V2_Pneumonia.ipynb
│   ├── 📓 ResNet101V2_Pneumonia.ipynb
│   └── 📓 VGG16_Pneumonia.ipynb
│
├── 📁 data/                     # Dataset not included due to large size
├── 📁 outputs/                  # Results and Visualizations
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 LICENSE
```
## 🧾 Requirements

`Python 3.x` `TensorFlow` `Keras` `Matplotlib` `Numpy` `Scikit-learn`
## 📊 Model Performance Comparison

| Model          | Accuracy | F1 Score | Loss   | Precision | Recall  |
|----------------|----------|----------|--------|-----------|---------|
| ConvNeXtBase   | 0.9705   | 0.9544   | 0.0747 | 0.9705    | 0.9705  |
| DenseNet121    | 0.9086   | 0.9285   | 0.3432 | 0.9086    | 0.9086  |
| ResNet50V2     | 0.9537   | 0.9459   | 0.1723 | 0.9537    | 0.9537  |
| ResNet101V2    | 0.9595   | 0.9356   | 0.1784 | 0.9595    | 0.9595  |
| VGG16          | 0.9595   | 0.9192   | 0.1030 | 0.9595    | 0.9595  |


## 🤝 Contributing
Contributions are welcome!
Feel free to fork the project and submit a pull request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙌 Acknowledgements
- ```Paul Mooney``` for the dataset
- TensorFlow / Keras community
- Medical professionals contributing to open datasets

## 🌟 Let's Connect!
If you like this project, please give it a ⭐!
Feel free to connect with me on LinkedIn or check out more of my work on GitHub.
