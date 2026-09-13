# from flask import Flask
# from flask_cors import CORS

# from config import Config
# from app.extensions import db


# def create_app():
#     app = Flask(__name__)

#     app.config.from_object(Config)

#     CORS(app)

#     db.init_app(app)

#     return app

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)