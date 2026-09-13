from flask import Blueprint


scan_bp = Blueprint(
    "scan",
    __name__,
    url_prefix="/api/scan"
)


# Import routes after creating the blueprint
from app.blueprints.scan import routes