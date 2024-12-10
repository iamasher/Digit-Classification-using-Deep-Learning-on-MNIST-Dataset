
# **Digit Classification using Deep Learning on MNIST Dataset**

This repository contains our **Final Year Minor Project**: **Digit Classification using Deep Learning on the MNIST Dataset**. In this project, we use **Convolutional Neural Networks (CNNs)** to accurately classify handwritten digits from the MNIST dataset and extend the functionality to predict digits from custom input images.  

This work represents the culmination of our learning and collaborative efforts as students of **Loknayak Jai Prakash Institute of Technology, Chapra**, under the guidance of **Professor Sudhir Pandey**.

---

## **Project Team**
We are a group of final-year Computer Science Engineering students collaborating on this project. Our names, roles, and responsibilities are listed below in order of registration number. 

| Name                | Registration Number      | Roles & Responsibilities |
|---------------------|--------------------------|--------------------------|
| Md Abdullah         | 21105117011              |                          |
| Rupesh Kumar        | 21105117023              |                          |
| Md Asher            | 21105117055              |                       |
| Fariya Rafat        | 21105117057              |                   |

### Project Guide:
Professor Sudhir Pandey, Loknayak Jai Prakash Institute of Technology, Chapra.


---

## **Project Overview**
This project implements a **deep learning model** using **Convolutional Neural Networks (CNNs)** to recognize handwritten digits. The primary dataset used is the **MNIST dataset**, which contains 60,000 training and 10,000 testing images of digits (0-9). Additionally, the model can predict digits from custom images, providing practical real-world applications.

### **Key Features**
- High accuracy on the MNIST dataset test set.
- Supports prediction on custom input images.
- Modular codebase for easy modifications and experimentation.
- Includes data preprocessing, model training, evaluation, and testing workflows.

---

## **Tech Stack**
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Libraries Used**:  
  - NumPy (for numerical computations)  
  - Matplotlib (for visualization)  
  - PIL (for image processing)  
- **Dataset**: MNIST Handwritten Digits Dataset

---

## **Requirements**
To run this project, you need to install the following dependencies:

- **TensorFlow** (for building and training the CNN model)
- **Keras** (high-level neural networks API)
- **NumPy** (for handling numerical data)
- **Matplotlib** (for data visualization and plotting)
- **Pillow (PIL)** (for image processing)

You can install all required dependencies by running:

```bash
pip install -r requirements.txt
```

The **`requirements.txt`** file includes the following libraries:
```
tensorflow==2.x.x
numpy==1.x.x
matplotlib==3.x.x
Pillow==8.x.x
```

---

## **Setup and Usage**
### **Steps to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/iamasher/Digit-Classification-using-Deep-Learning-on-MNIST-Dataset.git
   cd Digit-Classification-using-Deep-Learning-on-MNIST-Dataset
   ```
2. Train the model:
   ```bash
   python train.py
   ```
3. Use the trained model to predict digits from custom images:
   ```bash
   python predict.py --image-path path/to/your/image.jpg
   ```

---

## **Contributing**
This repository is collaboratively managed by the project team. Contributions are limited to team members. If you have suggestions or feedback, feel free to reach out.

---

## **Acknowledgments**
We extend our heartfelt gratitude to **Professor Sudhir Pandey** for his valuable guidance and mentorship throughout this project. His insights and support have been instrumental in shaping our work.

We also thank **Loknayak Jai Prakash Institute of Technology, Chapra**, for providing the platform and resources to execute this project.

---

## **License**
This repository is private and intended for academic purposes only.  

