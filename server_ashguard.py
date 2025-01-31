#!/usr/bin/env python3
import os
import numpy as np
from datetime import datetime

from flask import Flask, request, jsonify
import dash
from dash import dcc, html
import plotly.graph_objs as go

# --- Intégration de la base de données avec SQLAlchemy ---
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Récupération de l'URL de connexion à la base DB_ASHGUARD depuis les variables d'environnement
db_url = os.environ.get("DB_ASHGUARD")
if not db_url:
    raise Exception("La variable d'environnement DB_ASHGUARD n'est pas définie.")

engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SensorData(Base):
    __tablename__ = "sensor_data"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float)
    voc = Column(Float)
    co2 = Column(Float)
    precision = Column(Integer)
    humidity = Column(Float)
    iaq = Column(Float)
    heading = Column(Float)
    pitch = Column(Float)
    roll = Column(Float)

# Pour un déploiement rapide, on peut créer automatiquement la table.
# En production, il est recommandé d'utiliser un outil de migration (ex. Alembic).
Base.metadata.create_all(bind=engine)

# --- Variable globale pour le dashboard ---
latest_data = {
    "temperature": 25.0,
    "voc": 0.0,
    "co2": 400.0,
    "precision": 0,
    "humidity": 50.0,
    "iaq": 50.0,
    "heading": 0,
    "pitch": 0,
    "roll": 0
}

# --- Partie Flask : API REST pour recevoir les données ---
server = Flask(__name__)

@server.route('/api/receiver', methods=['POST'])
def receive_data():
    """
    Reçoit les données JSON postées depuis l'appareil.
    Payload attendu (exemple) :
    {
        "temperature": 23.5,
        "voc": 120,
        "co2": 800,
        "precision": 2,
        "humidity": 45,
        "iaq": 60,
        "heading": 180,
        "pitch": 10,
        "roll": -5
    }
    """
    global latest_data
    try:
        data = request.get_json()
        for key in ["temperature", "voc", "co2", "precision", "humidity", "iaq", "heading", "pitch", "roll"]:
            if key in data:
                latest_data[key] = data[key]

        print("Données reçues :", data)

        # Sauvegarde des données dans la base DB_ASHGUARD
        db = SessionLocal()
        sensor_record = SensorData(
            temperature=data.get("temperature"),
            voc=data.get("voc"),
            co2=data.get("co2"),
            precision=data.get("precision"),
            humidity=data.get("humidity"),
            iaq=data.get("iaq"),
            heading=data.get("heading"),
            pitch=data.get("pitch"),
            roll=data.get("roll")
        )
        db.add(sensor_record)
        db.commit()
        db.close()

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print("Erreur lors de la réception :", e)
        return jsonify({"status": "error", "message": str(e)}), 400

# --- Partie Dash : Dashboard interactif ---
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app.title = "Dashboard des Capteurs"

app.layout = html.Div([
    html.H1("Dashboard des Capteurs"),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    html.Div([
        dcc.Graph(id='temperature-gauge', style={'display': 'inline-block', 'width': '16%'}),
        dcc.Graph(id='voc-gauge', style={'display': 'inline-block', 'width': '16%'}),
        dcc.Graph(id='co2-gauge', style={'display': 'inline-block', 'width': '16%'}),
        dcc.Graph(id='precision-gauge', style={'display': 'inline-block', 'width': '16%'}),
        dcc.Graph(id='humidity-gauge', style={'display': 'inline-block', 'width': '16%'}),
        dcc.Graph(id='iaq-gauge', style={'display': 'inline-block', 'width': '16%'})
    ]),
    html.Div([ dcc.Graph(id='compass-gauge') ]),
    html.Div([ dcc.Graph(id='cube-3d') ])
])

@app.callback(
    [dash.dependencies.Output('temperature-gauge', 'figure'),
     dash.dependencies.Output('voc-gauge', 'figure'),
     dash.dependencies.Output('co2-gauge', 'figure'),
     dash.dependencies.Output('precision-gauge', 'figure'),
     dash.dependencies.Output('humidity-gauge', 'figure'),
     dash.dependencies.Output('iaq-gauge', 'figure'),
     dash.dependencies.Output('compass-gauge', 'figure'),
     dash.dependencies.Output('cube-3d', 'figure')],
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Récupération des dernières données
    temp = latest_data.get("temperature", 25.0)
    voc = latest_data.get("voc", 0.0)
    co2 = latest_data.get("co2", 400.0)
    precision = latest_data.get("precision", 0)
    humidity = latest_data.get("humidity", 50.0)
    iaq = latest_data.get("iaq", 50.0)
    heading = latest_data.get("heading", 0)
    pitch = latest_data.get("pitch", 0)
    roll = latest_data.get("roll", 0)
    
    # Gauge pour la température
    temp_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=temp,
        title={'text': "Température (°C)"},
        gauge={'axis': {'range': [0, 50]}}
    ))
    
    # Gauge pour les COV (VOC)
    voc_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=voc,
        title={'text': "COV (VOC)"},
        gauge={'axis': {'range': [0, 500]}}
    ))
    
    # Gauge pour le CO₂
    co2_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=co2,
        title={'text': "CO₂ (ppb)"},
        gauge={'axis': {'range': [0, 2000]}}
    ))
    
    # Gauge pour la précision
    precision_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=precision,
        title={'text': "Précision"},
        gauge={'axis': {'range': [0, 3], 'dtick': 1}}
    ))
    
    # Gauge pour l'humidité
    humidity_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=humidity,
        title={'text': "Humidité (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    
    # Gauge pour l'IAQ
    iaq_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=iaq,
        title={'text': "IAQ"},
        gauge={'axis': {'range': [0, 500]}}
    ))
    
    # Boussole pour le heading
    compass = go.Figure(go.Indicator(
        mode="gauge+number",
        value=heading,
    
        number={'suffix': "°"},
        title={'text': "Heading"},
        gauge={
            'axis': {'range': [0, 360]},
            'bar': {'color': "darkblue"},
            'steps': [{'range': [0, 360], 'color': "lightgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': heading
            }
        }
    ))
    
    # Cube 3D dynamique (rotation selon pitch et roll)
    cube_vertices = np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5]
    ])
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    R_pitch = np.array([
        [ np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad),  np.cos(roll_rad)]
    ])
    R = R_pitch @ R_roll
    rotated_vertices = cube_vertices.dot(R.T)
    faces = [
        [0, 1, 3], [0, 3, 2],
        [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4],
        [1, 5, 7], [1, 7, 3]
    ]
    x = rotated_vertices[:, 0]
    y = rotated_vertices[:, 1]
    z = rotated_vertices[:, 2]
    i_idx, j_idx, k_idx = [], [], []
    for face in faces:
        i_idx.append(face[0])
        j_idx.append(face[1])
        k_idx.append(face[2])
    cube_mesh = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i_idx, j=j_idx, k=k_idx,
            opacity=0.5,
            color='orange'
        )
    ])
    cube_mesh.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-1, 1]),
            yaxis=dict(nticks=4, range=[-1, 1]),
            zaxis=dict(nticks=4, range=[-1, 1]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    return temp_gauge, voc_gauge, co2_gauge, precision_gauge, humidity_gauge, iaq_gauge, compass, cube_mesh

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run_server(host='0.0.0.0', port=port)
