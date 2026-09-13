import os
import uuid
from pathlib import Path

from flask import current_app, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from PIL import Image
from werkzeug.utils import secure_filename

from app.blueprints.scan import scan_bp
from app.extensions import db
from app.ml.inference import PlantInference
from app.models.scan import ScanRecord


# ============================================================
# ML ENGINE
# ============================================================

_inference_engine = None


def get_inference_engine():
    global _inference_engine

    if _inference_engine is None:
        _inference_engine = PlantInference(
            species_model_path=current_app.config[
                "SPECIES_MODEL_PATH"
            ],
            species_mapping_path=current_app.config[
                "SPECIES_MAPPING_PATH"
            ],
            disease_model_path=current_app.config[
                "DISEASE_MODEL_PATH"
            ],
            disease_mapping_path=current_app.config[
                "DISEASE_MAPPING_PATH"
            ],
        )

    return _inference_engine


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in current_app.config[
        "ALLOWED_IMAGE_EXTENSIONS"
    ]


# ============================================================
# POST /api/scan
# ============================================================

@scan_bp.route("", methods=["POST"])
@jwt_required()
def create_scan():

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    user_id = int(
        get_jwt_identity()
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if "image" not in request.files:
        return jsonify({
            "error": "Image file is required"
        }), 400

    image_file = request.files["image"]

    if not image_file.filename:
        return jsonify({
            "error": "No image selected"
        }), 400

    if not allowed_file(image_file.filename):
        return jsonify({
            "error": (
                "Unsupported image format. "
                "Allowed: jpg, jpeg, png, webp"
            )
        }), 400

    # --------------------------------------------------------
    # UPLOAD DIRECTORY
    # --------------------------------------------------------

    upload_folder = Path(
        current_app.config["UPLOAD_FOLDER"]
    )

    upload_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # UNIQUE FILENAME
    # --------------------------------------------------------

    original_name = secure_filename(
        image_file.filename
    )

    extension = original_name.rsplit(
        ".",
        1
    )[1].lower()

    unique_filename = (
        f"{uuid.uuid4().hex}.{extension}"
    )

    image_path = upload_folder / unique_filename

    # --------------------------------------------------------
    # SAVE TEMP IMAGE
    # --------------------------------------------------------

    image_file.save(
        str(image_path)
    )

    try:

        # ----------------------------------------------------
        # VERIFY IMAGE
        # ----------------------------------------------------

        with Image.open(image_path) as image:

            image.verify()

        # ----------------------------------------------------
        # ML INFERENCE
        # ----------------------------------------------------

        engine = get_inference_engine()

        result = engine.predict(
            image_path
        )

        # ----------------------------------------------------
        # SPECIES
        # ----------------------------------------------------

        species_final = result[
            "species"
        ]["final"]

        species_top_3 = result[
            "species"
        ]["top_3"]

        species_name = species_final[
            "name"
        ]

        species_confidence = species_final[
            "confidence"
        ]

        # ----------------------------------------------------
        # DISEASE
        # ----------------------------------------------------

        disease_result = result[
            "disease"
        ]

        disease_name = None
        disease_confidence = None
        is_healthy = None

        if disease_result["available"]:

            disease_final = disease_result[
                "final"
            ]

            disease_name = disease_final[
                "name"
            ]

            disease_confidence = disease_final[
                "confidence"
            ]

            is_healthy = disease_result[
                "is_healthy"
            ]

        # ----------------------------------------------------
        # DATABASE RECORD
        # ----------------------------------------------------

        scan = ScanRecord(
            user_id=user_id,

            image_path=str(
                image_path
            ),

            species_prediction=species_name,

            species_confidence=species_confidence,

            top_3_fallback=species_top_3,

            is_healthy=is_healthy,

            disease_name=disease_name,

            disease_confidence=disease_confidence,

            severity=None,

            gradcam_image_path=None,

            growth_stage=None,

            recommendations=None
        )

        db.session.add(scan)

        db.session.commit()

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({
            "message": "Plant scan completed",

            "scan": scan.to_dict(),

            "ml": {
                "device": result["device"]
            }
        }), 201

    except Exception as error:

        db.session.rollback()

        # Remove failed upload
        if image_path.exists():
            try:
                image_path.unlink()
            except OSError:
                pass

        current_app.logger.exception(
            "Plant scan failed"
        )

        return jsonify({
            "error": "Plant scan failed",
            "details": str(error)
        }), 500

# ============================================================
# GET /api/scan/history
# ============================================================

@scan_bp.route("/history", methods=["GET"])
@jwt_required()
def get_scan_history():

    user_id = int(get_jwt_identity())

    scans = (
        ScanRecord.query
        .filter_by(user_id=user_id)
        .order_by(ScanRecord.created_at.desc())
        .all()
    )

    return jsonify({
        "count": len(scans),
        "scans": [scan.to_dict() for scan in scans]
    }), 200


# ============================================================
# GET /api/scan/<scan_id>
# ============================================================

@scan_bp.route("/<int:scan_id>", methods=["GET"])
@jwt_required()
def get_scan_details(scan_id):

    user_id = int(get_jwt_identity())

    scan = (
        ScanRecord.query
        .filter_by(
            id=scan_id,
            user_id=user_id
        )
        .first()
    )

    if not scan:
        return jsonify({
            "error": "Scan not found"
        }), 404

    return jsonify({
        "scan": scan.to_dict()
    }), 200