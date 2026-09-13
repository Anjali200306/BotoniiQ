from app import create_app
from app.extensions import db

# Import models so SQLAlchemy knows about them
from app.models.test_user import TestUser

app = create_app()

with app.app_context():
    db.create_all()
    print("Database tables created successfully!")