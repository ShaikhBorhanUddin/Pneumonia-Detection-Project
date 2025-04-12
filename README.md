# 🩺 Pneumonia Detection from Chest X-Rays using DenseNet121
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
- Source:

The dataset sourced from Kaggle. To access it click [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?raw=true)

- Classes:

`Normal`  `Pneumonia`

- Data Split:

`Train` `Test`

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
