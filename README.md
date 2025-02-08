# frosthack


#Plastic Detection in River using Machine Learning

This GitHub repository contains a project that focuses on detecting plastic in rivers using machine learning techniques. The project utilizes the YOLOv8 model for transfer learning to try to achieve accurate and efficient plastic detection.


#**Features**

Utilizes YOLOv8 model for plastic detection
Training code for fine-tuning the YOLOv8 model with custom plastic dataset
Pre-trained weights for YOLOv8m model with plastic detection capability
Evaluation code to measure the performance of the model on validation datset
Inference code for detecting plastic in river images or videos( webcam or from local video)
Dataset for training and testing the mode


#**Requirements**


To run the model:

-Create a virtual environment using: python -m venv venv

-Activate the virtual environment: ./venv/scripts/activate

-Install the required packages from requirements.txt: pip install -r requirements.txt

-To run inference, use the predict.py file

#**Sample**


Input Image:

![unpredicted image](https://github.com/user-attachments/assets/d7354723-bc40-4afa-9efe-39edfaee56a9)

Predicted Image:

![predicted image](https://github.com/user-attachments/assets/b4f46b09-7f81-4fb1-8194-ac175d0ce5bc)


#**Training Curves and details**:-

Training was done in two seperate instances, each of 300 epochs. The details of the second instance is shown below:

![Screenshot 2025-02-08 060458](https://github.com/user-attachments/assets/a8fb568b-5c35-4271-8ea0-bfda09a21c72)


#**Confusion Matrix**

'0' stands for the detected plastic

![Screenshot 2025-02-08 060757](https://github.com/user-attachments/assets/9c6f4828-9ef2-4c8e-b696-683b06f0c585)

#**Precision-Recall Curve**

'0' refers to the class 'Plastic'

![Screenshot 2025-02-08 061158](https://github.com/user-attachments/assets/e903e627-48b2-44cb-b317-64d9ea1cdc79)

**Acknowledgements**:

The project uses the YOLOv8m model uses the following resources:

-YOLO

-Initial Dataset provided by REVA University

-Annotaions done using Make Sense

-Model prepared using Google Colab










