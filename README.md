# House Price Prediction

A Machine Learning web app that predicts house prices based on features like bedrooms, bathrooms, area, city, and location.

Built using Python, Scikit-learn, and Flask.

## Features
- Predicts house price in real-time
- Takes 15 inputs like bedrooms, bathrooms, sqft_living, floors, city, statezip, country
- Clean web interface using Flask + HTML
- Trained on King County Housing Dataset

## Tech Stack
- Python
- Pandas, NumPy, Scikit-learn
- Flask
- HTML / CSS

## Project Structure
House_Price_prediction/
- app.py - Flask web app
- main.py - Model training code
- data.csv - Dataset (King County Houses)
- requirements.txt - Dependencies
- templates/index.html - Frontend form
- Procfile - For deployment

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train the model (this will create HPP_Model.pkl):
python main.py

3. Run the web app:
python app.py

4. Open in browser:
http://127.0.0.1:5000

## Example Input
- Bedrooms: 3
- Bathrooms: 2
- Sqft Living: 1570
- City: Federal Way
- Statezip: WA 98003
- Country: USA

Output: Predicted Price

## Author
Venneladhibba
