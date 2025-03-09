# Home Energy Prediction

This is a project I worked on to learn about predicting home energy usage using time-series models such as LSTMs. The goal was to build a predictive model from scratch, focusing on understanding how to preprocess data, create a custom AI model, and train it effectively. I did not use pre-trained models, which did affect my accuracy, but the goal of this project was to design the architecture myself to gain deeper insights into how these systems work.

The data for this project was collected directly from my own house over the span of approximately one year, and thought me how to work with unprocessed data. Due to the limited amount of data, the model's accuracy is not perfect and it struggles to capture long-term trends. However, the process of making this gave me valuable experience in training machine learning models on unstructured data.

The purpose of this project was educational, so the accuracy is limited.

![energy_usage](https://github.com/user-attachments/assets/94a1ce64-fddf-4476-b8d3-ab91fafe552e)

## How to Use
   ```
   pip install -r requirements.txt
   python src/main.py --predict
   ```

This is to do model inference to visualize a plot that predicts the energy usage on a pre-defined time span.
