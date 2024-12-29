import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import requests


# Charte graphique graphique (crimi)
charte_graphique = {
    "Vols sans violence": "Magenta",
    "Escroquerie": "Green",
    "Coups et blessures volontaires": "Cyan",
    "Stupéfiants": "Pink",
    "Vols avec violence": "Red",
    "Violences sexuelles" : "Orange",
    "Tentatives d'homicides" : "Blue",
    "Homicides" : "Purple"
}

# Charte graphique cartographie (crimi)
charte_graphique2 = {
    "Vols sans violence": "PuRd",
    "Escroquerie": "BuGn",
    "Coups et blessures volontaires": "Greens",
    "Stupéfiants": "RdPu",
    "Vols avec violence": "Reds",
    "Violences sexuelles": "Oranges",
    "Tentatives d'homicides": "Blues",
    "Homicides": "Purples"
}

# Charte graphique (légi)
charte_graphique3 = {
    "Texte": "Magenta",
    "Arrete": "Green",
    "Loi": "Cyan",
    "Decret": "Pink",
    "Ordonnance": "Red",
}

all = ["Arrete", "Loi", "Decret", "Ordonnance"]

# Dictionnaire géométrie (carto)
url = "https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb"
contours_dpt = gpd.read_file(url)
dictionnaire_geo = {row['code']: row['geometry'] for _, row in contours_dpt.iterrows()}

def create_custom_greys_cmap():
    """
    Crée une échelle de gris personalisée pour les cartes de densité.

    """
    colors = [
        (0.95, 0.95, 0.95), 
        (0.8, 0.8, 0.8),    
        (0.6, 0.6, 0.6),    
        (0.4, 0.4, 0.4),   
        (0.2, 0.2, 0.2)     
    ]
    return LinearSegmentedColormap.from_list("CustomGreys", colors)

# Echelle de gris
custom_greys_cmap = create_custom_greys_cmap()