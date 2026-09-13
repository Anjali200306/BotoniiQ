from flask import Flask

from config import Config
from app.extensions import (
    db,
    migrate,
    cors,
    bcrypt,
    jwt
)


def create_app():

    app = Flask(__name__)

    app.config.from_object(
        Config
    )

    # ========================================================
    # EXTENSIONS
    # ========================================================

    db.init_app(app)

    migrate.init_app(
        app,
        db
    )

    cors.init_app(
        app,
        resources={
            r"/api/*": {
                "origins": "http://localhost:3000"
            }
        }
    )

    bcrypt.init_app(app)

    jwt.init_app(app)

    # ========================================================
    # BLUEPRINTS
    # ========================================================

    from app.blueprints.auth import auth_bp

    app.register_blueprint(
        auth_bp
    )

    from app.blueprints.scan import scan_bp

    app.register_blueprint(
        scan_bp
    )

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    @app.route("/")
    def health_check():

        return {
            "status":
                "BotaniQ backend is running"
        }

    return app