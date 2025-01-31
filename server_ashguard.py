#!/usr/bin/env python3
"""
Application de réception et affichage des données capteurs

- L’API Flask écoute sur l’endpoint `/api/receiver` et reçoit des données JSON.
- Les données attendues (exemple) :
  {
      "temperature": 23.5,
      "pressure": 1013.25,
      "iaq": 50,
      "heading": 270,
      "pitch": 10,
      "roll": -5
  }
- Le dashboard Dash affiche :
    • Des indicateurs (gauges) pour la température, la pression et l’IAQ.
    • Une boussole (gauge) pour le heading.
    • Un cube 3D qui se fait tourner en fonction de pitch et roll.
    
Pour déployer sur Render en HTTPS :
1. Créez un nouveau service Web sur Render et indiquez ce fichier comme point d'entrée.
   Render fournira automatiquement un certificat SSL (HTTPS).
2. Pour la base de données, rendez-vous sur le dashboard Render,
   cliquez sur "New" puis "Database" et choisissez PostgreSQL.
   Configurez ensuite vos variables d’environnement de connexion dans votre service.
"""

import os, math, json
import numpy as np
from flask import Flask, request, jsonify
import dash
from dash import dcc, html
import plotly.graph_objs as go

# Variable globale pour stocker les dernières données reçues
latest_data = {
    "temperature": 25.0,  # en °C
    "pressure": 1013.25,  # en hPa
    "iaq": 50,           # indice de qualité d'air (exemple)
    "heading": 0,        # en degrés
    "pitch": 0,          # en degrés
    "roll": 0            # en degrés
}

# --- Partie Flask : API REST ---

server = Flask(__name__)

@server.route('/api/receiver', methods=['POST'])
def receive_data():
    """
    Reçoit les données JSON postées depuis l’Arduino.
    Met à jour la variable globale latest_data.
    """
    global latest_data
    try:
        data = request.get_json()
        # Mise à jour des valeurs reçues (si présentes)
        if "temperature" in data:
            latest_data["temperature"] = data["temperature"]
        if "pressure" in data:
            latest_data["pressure"] = data["pressure"]
        if "iaq" in data:
            latest_data["iaq"] = data["iaq"]
        if "heading" in data:
            latest_data["heading"] = data["heading"]
        if "pitch" in data:
            latest_data["pitch"] = data["pitch"]
        if "roll" in data:
            latest_data["roll"] = data["roll"]
        print("Données reçues :", data)
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
    # Composant pour mettre à jour les figures toutes les secondes
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),

    # Rangée de gauges pour Température, Pression et IAQ
    html.Div([
        dcc.Graph(id='temperature-gauge', style={'display': 'inline-block', 'width': '33%'}),
        dcc.Graph(id='pressure-gauge', style={'display': 'inline-block', 'width': '33%'}),
        dcc.Graph(id='iaq-gauge', style={'display': 'inline-block', 'width': '33%'}),
    ]),
    
    # Boussole pour le heading
    html.Div([
        dcc.Graph(id='compass-gauge')
    ]),
    
    # Plot 3D dynamique d'un cube (piloté par pitch et roll)
    html.Div([
        dcc.Graph(id='cube-3d')
    ])
])

@app.callback(
    [dash.dependencies.Output('temperature-gauge', 'figure'),
     dash.dependencies.Output('pressure-gauge', 'figure'),
     dash.dependencies.Output('iaq-gauge', 'figure'),
     dash.dependencies.Output('compass-gauge', 'figure'),
     dash.dependencies.Output('cube-3d', 'figure')],
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Récupération des dernières données
    temp    = latest_data.get("temperature", 25.0)
    pres    = latest_data.get("pressure", 1013.25)
    iaq_val = latest_data.get("iaq", 50)
    heading = latest_data.get("heading", 0)
    pitch   = latest_data.get("pitch", 0)
    roll    = latest_data.get("roll", 0)
    
    # --- Gauge Température ---
    temp_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=temp,
        title={'text': "Température (°C)"},
        gauge={'axis': {'range': [0, 50]}}
    ))
    
    # --- Gauge Pression ---
    pres_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pres,
        title={'text': "Pression (hPa)"},
        gauge={'axis': {'range': [900, 1100]}}
    ))
    
    # --- Gauge IAQ ---
    iaq_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=iaq_val,
        title={'text': "IAQ"},
        gauge={'axis': {'range': [0, 500]}}
    ))
    
    # --- Boussole pour le Heading ---
    # On utilise un indicateur gauge dont la plage va de 0 à 360 degrés.
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
    
    # --- Cube 3D dynamique (rotation selon pitch et roll) ---
    # Définition d'un cube centré à (0,0,0) de taille 1.
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
    
    # Matrices de rotation :
    # - Rotation de pitch autour de l'axe Y
    # - Rotation de roll autour de l'axe X
    pitch_rad = np.radians(pitch)
    roll_rad  = np.radians(roll)
    
    R_pitch = np.array([
        [ np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [ 0,                 1, 0                ],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    R_roll = np.array([
        [1, 0,                 0                ],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad),  np.cos(roll_rad)]
    ])
    
    # Rotation combinée (d'abord roll, puis pitch)
    R = R_pitch @ R_roll
    rotated_vertices = cube_vertices.dot(R.T)
    
    # Définition des faces du cube (chaque face représentée par deux triangles)
    faces = [
        [0, 1, 3], [0, 3, 2],  # face gauche
        [4, 6, 7], [4, 7, 5],  # face droite
        [0, 4, 5], [0, 5, 1],  # face inférieure
        [2, 3, 7], [2, 7, 6],  # face supérieure
        [0, 2, 6], [0, 6, 4],  # face arrière
        [1, 5, 7], [1, 7, 3]   # face avant
    ]
    
    # Extraction des coordonnées x, y, z
    x = rotated_vertices[:, 0]
    y = rotated_vertices[:, 1]
    z = rotated_vertices[:, 2]
    
    # Construction des listes d’indices pour Mesh3d
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
    
    return temp_gauge, pres_gauge, iaq_gauge, compass, cube_mesh

# --- Lancement de l'application ---
if __name__ == '__main__':
    # Pour Render, assurez-vous que votre service web écoute sur le port défini dans la variable d'environnement PORT.
    port = int(os.environ.get("PORT", 5000))
    # Lancement de l'application Dash (accessible via HTTPS sur Render)
    app.run_server(host='0.0.0.0', port=port)
