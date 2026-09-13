# import os
# from dotenv import load_dotenv

# load_dotenv()


# class Config:
#     SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

#     SQLALCHEMY_DATABASE_URI = (
#         f"postgresql+psycopg://"
#         f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
#         f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}"
#         f"/{os.getenv('DB_NAME')}"
#     )

#     SQLALCHEMY_TRACK_MODIFICATIONS = False


import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ========================================================
    # SECURITY
    # ========================================================

    SECRET_KEY = os.environ.get(
        "SECRET_KEY",
        "dev-fallback-key"
    )

    JWT_SECRET_KEY = os.environ.get(
        "SECRET_KEY",
        "dev-fallback-key"
    )

    # ========================================================
    # DATABASE
    # ========================================================

    DB_USER = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST")
    DB_PORT = os.environ.get("DB_PORT")
    DB_NAME = os.environ.get("DB_NAME")

    SQLALCHEMY_DATABASE_URI = (
        f"postgresql://"
        f"{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ========================================================
    # BOTONIIQ ML
    # ========================================================

    BOTONIIQ_ML_DIR = os.environ.get(
        "BOTONIIQ_ML_DIR"
    )

    if not BOTONIIQ_ML_DIR:
        raise RuntimeError(
            "BOTONIIQ_ML_DIR is not configured in .env"
        )

    SPECIES_MODEL_PATH = os.path.join(
        BOTONIIQ_ML_DIR,
        "outputs",
        "models",
        "species_v4_best.pth"
    )

    SPECIES_MAPPING_PATH = os.path.join(
        BOTONIIQ_ML_DIR,
        "outputs",
        "models",
        "species_v4_class_names.json"
    )

    DISEASE_MODEL_PATH = os.path.join(
        BOTONIIQ_ML_DIR,
        "outputs",
        "models",
        "disease_efficientnet_b0_best.pth"
    )

    DISEASE_MAPPING_PATH = os.path.join(
        BOTONIIQ_ML_DIR,
        "outputs",
        "models",
        "disease_class_names.json"
    )

    # ========================================================
    # UPLOADS
    # ========================================================

    UPLOAD_FOLDER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "uploads",
        "scans"
    )

    MAX_CONTENT_LENGTH = 10 * 1024 * 1024

    ALLOWED_IMAGE_EXTENSIONS = {
        "jpg",
        "jpeg",
        "png",
        "webp"
    }