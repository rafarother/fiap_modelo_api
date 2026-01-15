import os
import logging
import datetime
import jwt
from functools import wraps

from flask import Flask, request, jsonify
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker



JWT_SECRET = 'MEUSEGREDOAQUI'
JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_SECONDS = 3600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo")

DB_URI = os.environ.get("DB_URI")
engine = create_engine(DB_URI, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)

ml_model = joblib.load(os.path.join(os.path.dirname(__file__), 'modelo_iris.pkl'))
logger.info("Modelo carregado com sucesso.")


app = Flask(__name__)
predictions_cache = {}

TEST_USER = "admin"
TEST_PASSWORD = "secret"

def create_token(username):
    payload = {
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        #pegar token do jeader Authorization: Bearer <token>
        # decodificar e checar expiracao
        return f(*args, **kwargs)
    return decorated


@app.route("/")
def home():
    return "API Flask funcionando no Vercel!"

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    username = data.get('username')
    password = data.get('password')
    if username == TEST_USER and password == TEST_PASSWORD:
        token = create_token(username)
        return jsonify({'token': token})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401
    
@app.route('/predict', methods=['POST'])
@token_required
def predict():
    data = request.get_json(force=True)

    sepal_length = data.get('sepal_length')
    sepal_width = data.get('sepal_width')
    petal_length = data.get('petal_length')
    petal_width = data.get('petal_width')

    features = (sepal_length, sepal_width, petal_length, petal_width)

    if features in predictions_cache:
        predicted_class = predictions_cache[features]
        logger.info("Cache hit para %s", features)
    else:
        input_data = np.array([features])
        predicted_class = int(ml_model.predict(input_data)[0])
        predictions_cache[features] = predicted_class
        logger.info("Cache update para %s", features)

    db = SessionLocal()
    new_prediction = Prediction(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predicted_class=predicted_class
    )
    db.add(new_prediction)
    db.commit()
    db.close()

    return jsonify({"predicted_class": predicted_class})

@app.route('/predictions', methods=['GET'])
@token_required
def list_predictions():
    """
    Lista as predições armazenadas no banco de dados.
    Parâmetros de consulta opcionais (via query string):
     - limit (int): quantos registros retornar (default 10)
     - offset (int): quantos registros pular (default 0)
    Exemplo: /predictions?limit=5&offset=10
    """
    limit = int(request.args.get('limit', 10))
    offset = int(request.args.get('offset', 0))
    db = SessionLocal()
    preds = db.query(Prediction).order_by(Prediction.created_at.desc()).offset(offset).limit(limit).all()
    db.close()
    results = []
    for p in preds:
        results.append({
            'id': p.id,
            'sepal_length': p.sepal_length,
            'sepal_width': p.sepal_width,
            'petal_length': p.petal_length,
            'petal_width': p.petal_width,
            'predicted_class': p.predicted_class,
            'created_at': p.created_at.isoformat()
        })
    return jsonify(results)

##if __name__ == '__main__':
  ##  app.run(debug=True)
  ## remover por causa do vercel