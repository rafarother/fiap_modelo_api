import os
import logging
import datetime
import jwt
from functools import wraps

from flask import Flask, request, jsonify
import joblib
import numpy as np

from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError


# ======================
# CONFIGURAÇÕES
# ======================

JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

DB_URI = os.environ.get("DB_URI")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo")


# ======================
# DATABASE (LAZY INIT)
# ======================

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def get_db_session():
    """
    Cria engine e sessão SOB DEMANDA.
    Evita crash em cold start no Vercel.
    """
    engine = create_engine(
        DB_URI,
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0
    )
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# ======================
# MODELO ML
# ======================

ml_model = joblib.load(
    os.path.join(os.path.dirname(__file__), "modelo_iris.pkl")
)
logger.info("Modelo carregado com sucesso.")


# ======================
# APP
# ======================

app = Flask(__name__)

predictions_cache = {}

TEST_USER = "admin"
TEST_PASSWORD = "secret"


# ======================
# AUTH
# ======================

def create_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow()
        + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization")

        if not auth or not auth.startswith("Bearer "):
            return jsonify({"error": "Token missing"}), 401

        token = auth.split(" ")[1]

        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorated


# ======================
# ROTAS
# ======================

@app.route("/")
def home():
    return "API Flask funcionando no Vercel!"


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)

    if (
        data.get("username") == TEST_USER
        and data.get("password") == TEST_PASSWORD
    ):
        return jsonify({"token": create_token(TEST_USER)})

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/predict", methods=["POST"])
@token_required
def predict():
    data = request.get_json(force=True)

    try:
        features = (
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"]),
        )
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    if features in predictions_cache:
        predicted_class = predictions_cache[features]
        logger.info("Cache hit %s", features)
    else:
        input_data = np.array([features])
        predicted_class = int(ml_model.predict(input_data)[0])
        predictions_cache[features] = predicted_class

    db = None
    try:
        db = get_db_session()
        db.add(
            Prediction(
                sepal_length=features[0],
                sepal_width=features[1],
                petal_length=features[2],
                petal_width=features[3],
                predicted_class=predicted_class,
            )
        )
        db.commit()
    except SQLAlchemyError as e:
        logger.error("Erro ao salvar no banco: %s", e)
    finally:
        if db:
            db.close()

    return jsonify({"predicted_class": predicted_class})


@app.route("/predictions", methods=["GET"])
@token_required
def list_predictions():
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))

    db = None
    try:
        db = get_db_session()
        preds = (
            db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return jsonify(
            [
                {
                    "id": p.id,
                    "sepal_length": p.sepal_length,
                    "sepal_width": p.sepal_width,
                    "petal_length": p.petal_length,
                    "petal_width": p.petal_width,
                    "predicted_class": p.predicted_class,
                    "created_at": p.created_at.isoformat(),
                }
                for p in preds
            ]
        )
    finally:
        if db:
            db.close()
