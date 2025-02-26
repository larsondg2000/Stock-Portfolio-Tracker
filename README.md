# Portfolio Tracker Web Application

A Flask-based web application for tracking stock portfolios, dividends, and portfolio analysis.

![Portfolio Tracker](static/images/wallstreet.jpg)

## Features

- **User Authentication**: Create an account and securely login to manage your portfolio
- **Portfolio Management**: Add, edit, and remove stocks from your portfolio
- **Real-time Data**: Fetches current stock prices and information using Yahoo Finance API
- **Dividend Tracking**: Monitor dividend income and analyze dividend performance
- **Risk Analysis**: Advanced portfolio risk metrics 
- **Historical Returns**: Individual stock and portfolio returns
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
portfolio-tracker/
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── portfolio.db            # SQLite database
├── static/                 # Static files
│   ├── css/
│   │   └── style.css       # Custom CSS
│   └── images/
│       ├── risk_formulas.png
│       ├── sharp.png
│       └── wallstreet.jpeg
└── templates/              # HTML templates
    ├── base.html           # Base template with common elements
    ├── index.html          # Landing page
    ├── login.html          # Login page
    ├── signup.html         # Signup page
    ├── portfolio.html      # Portfolio management page
    ├── dividends.html      # Dividend tracking page
    └── analysis.html       # Portfolio analysis page
```

## Installation and Setup

1. Clone the repository or download the source code

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create the database directory:
   ```bash
   mkdir -p instance
   ```

5. Create the images directory in static folder and add required images:
   ```bash
   mkdir -p static/images
   ```
   
   Copy the following images to the `static/images` folder:
   - risk_formulas.png
   - sharp.png
   - wallstreet.jpg 

6. Run the application:
   ```bash
   python app.py
   ```

7. Open your web browser and navigate to `http://127.0.0.1:5000`

## Deployment to Web Hosting

### Option 1: Deploy to PythonAnywhere

1. Create an account on [PythonAnywhere](https://www.pythonanywhere.com/)

2. Upload your project files to PythonAnywhere using their file uploader or by cloning from a Git repository

3. Create a virtual environment and install dependencies:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.9 myenv
   pip install -r requirements.txt
   ```

4. Configure a new web app:
   - Go to the Web tab
   - Create a new web app
   - Select "Flask" as the framework
   - Set the Python version to match your local development environment
   - Set the path to your Flask application (app.py)
   - Set the working directory to your project directory

5. Configure WSGI file to point to your app:
   ```python
   import sys
   path = '/home/yourusername/portfolio-tracker'
   if path not in sys.path:
       sys.path.insert(0, path)
   from app import app as application
   ```

6. Restart your web app

### Option 2: Deploy to Heroku

1. Create a Procfile in your project root:
   ```
   web: gunicorn app:app
   ```

2. Create a runtime.txt file to specify Python version:
   ```
   python-3.9.16
   ```

3. Ensure your app uses environment variables for sensitive data:
   ```python
   import os
   app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-default-key')
   app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///portfolio.db')
   ```

4. Install the Heroku CLI and login:
   ```bash
   heroku login
   ```

5. Create a new Heroku app:
   ```bash
   heroku create portfolio-tracker
   ```

6. Set environment variables:
   ```bash
   heroku config:set SECRET_KEY=your-secret-key
   ```

7. Push your code to Heroku:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

8. Set up a PostgreSQL database on Heroku:
   ```bash
   heroku addons:create heroku-postgresql:hobby-dev
   ```

9. Open your deployed application:
   ```bash
   heroku open
   ```

## Customization

- **Account Types**: Modify the account options in the app.py file to match your personal accounts
- **UI Theme**: Customize the colors and styles in the static/css/style.css file
- **Add Features**: Extend the application with additional features like tax optimization or asset allocation

## License

This project is licensed under the MIT License - see the LICENSE file for details.