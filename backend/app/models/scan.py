from datetime import datetime
from zoneinfo import ZoneInfo

from app.extensions import db


class ScanRecord(db.Model):
    __tablename__ = "scan_records"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    image_path = db.Column(
        db.String(500),
        nullable=False
    )

    # ========================================================
    # SPECIES
    # ========================================================

    species_prediction = db.Column(
        db.String(200)
    )

    species_confidence = db.Column(
        db.Float
    )

    top_3_fallback = db.Column(
        db.JSON
    )

    # ========================================================
    # DISEASE
    # ========================================================

    is_healthy = db.Column(
        db.Boolean
    )

    disease_name = db.Column(
        db.String(200)
    )

    disease_confidence = db.Column(
        db.Float
    )

    severity = db.Column(
        db.String(50)
    )

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    gradcam_image_path = db.Column(
        db.String(500)
    )

    # ========================================================
    # ADDITIONAL INFORMATION
    # ========================================================

    growth_stage = db.Column(
        db.String(100)
    )

    recommendations = db.Column(
        db.JSON
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(
            ZoneInfo("Asia/Kolkata")
        )
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_path": self.image_path,

            "species": {
                "name": self.species_prediction,
                "confidence": self.species_confidence,
                "top_3": self.top_3_fallback
            },

            "disease": {
                "available": self.disease_name is not None,
                "name": self.disease_name,
                "confidence": self.disease_confidence,
                "is_healthy": self.is_healthy,
                "severity": self.severity
            },

            "gradcam_image_path": self.gradcam_image_path,
            "growth_stage": self.growth_stage,
            "recommendations": self.recommendations,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            )
        }