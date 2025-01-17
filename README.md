# Stock Price Prediction and News Sentiment Analysis using LSTM and Random Forest 

<p> 
  This project aimed to build a web-based stock price prediction system using LSTM for
forecasting stock prices and Random Forest for sentiment analysis of trading news.<br>
The goal was to help traders make better decisions by providing stock predictions based on historical
data and real-time sentiment from news headlines.<br>
The project successfully achieved its objectives by creating a functional prediction model and integrating it with a sentiment analysis
model.<br>
The system offers a user-friendly interface where users can view predictions and
analyze stock trends based on both price history and news sentiment.
</p>

<h3>
  Tools Used:
</h3>
<hr>
<ul>
  <li>
    Programming languages: Python (machine learning models), JavaScript (frontend interactivity)
  </li>
  <li>
    Frameworks: Django (backend), Streamlit (interactive dashboard)
  </li>
  <li>
    Libraries:TensorFlow/PyTorchc(LSTM implementation), Scikit-learn (Random Forest sentiment analysis), Plotly/Matplotlib (data visualization), Pandas and NumPy (data manipulation)
  </li>
  <li>
    Database: Excel for processing data and Kaggle for storing user data, stock data, and analysis results
  </li>
  <li>
    Other Tools: Selenium/BeautifulSoup (web scraping)
  </li>
</ul>

<h3>
  Installation:
</h3>
  <hr>
<ol>
  <li>
    <strong>Clone the repository:</strong>
    <pre><code>git clone git@github.com:flexie0o0/Stock-Price-Prediction-and-News-Sentiment-Analysis-using-LSTM-and-Random-Forest.git </code></pre>
  </li>
  <li>
    <strong> Install virtual environment: </strong>
    <pre> pip install virtualenv </pre>
  </li>
    <li>
    <strong> Create a virtual environment: </strong>
    <pre> virtualenv venv </pre>
  </li>
    <li>
    <strong> Activate the virtual environment: </strong>
    <pre> venv\Scripts\activate </pre>
  </li>
    <li>
    <strong> Install django: </strong>
    <pre> pip install django </pre>
  </li>
    <li>
    <strong> Create a django project: </strong>
    <pre> django-admin startproject project_name </pre>
  </li>
    <li>
    <strong> Create a django app: </strong>
    <pre> python manage.py startapp app_name </pre>
  </li>
      <li>
    <strong> Create a superuser: </strong>
    <pre> python manage.py createsuperuser </pre>
  </li>
      <li>
    <strong> Create a django app: </strong>
    <pre> python manage.py startapp app_name </pre>
  </li>
</ol>

<h3>
  Running the app
</h3>
<hr>
<ol>
  <li>
    <strong> Running the django app: </strong>
    <pre> python manage.py runserver </pre>
  </li>
  <li>
    <strong> Running the streamlit app: </strong>
    <pre> streamlit run prediction/1_NASDAQ_Stock_Prediction.py </pre>
  </li>
</ol>

<h4>Note: </h4>
<p> Replace your gmail account and password in their respective fields in stockmarket->info.py </p>
