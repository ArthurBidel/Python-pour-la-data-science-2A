import requests
import pandas as pd
import geopandas as gpd
import os
import zipfile
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
import s3fs

from scipy.signal import savgol_filter

from pathlib import Path


import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from IPython.display import Image, display
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.animation as animation
from matplotlib import colors
from IPython.display import Image, display
import os
from shapely import wkt 


# Définir la charte graphique pour qu'elle soit uniforme entre les différentes stats descriptives (on veut que nos graphiques matchent avec nos cartes)
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

# Dictionnaire associant chaque indicateur à une couleur spécifique
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

#URL de téléchargement des contours départementaux
url = "https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb"
contours_dpt = gpd.read_file(url)

# Créer le dictionnaire des géométries via une compréhension
dictionnaire_geo = {row['code']: row['geometry'] for _, row in contours_dpt.iterrows()}

#Version de la fonction de tracage où on peut appliquer une échelle logarithmique ou lisser si on le veut (argument à appeler quand on appliquer la fonction)

def tracer_evolution_taux(
    df, charte_graphique, taux="Taux (/10 000)", title="Évolution des taux d'infractions", 
    xlabel="Date", ylabel="Nombre d'occurence pour 10 000 habitants", use_log_scale=False, 
    smooth=False, window_size=100, time_period=[]
):
    """
    Trace l'évolution des taux pour une liste d'indicateurs, avec option de lissage.

    Args:
        df (pd.DataFrame): Le dataframe contenant les données.
        charte_graphique (dict): Dictionnaire {nom_indicateur: couleur}.
        taux (str): Colonne du dataframe contenant les valeurs à tracer.
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe des x.
        ylabel (str): Label de l'axe des y.
        use_log_scale (bool): Utiliser une échelle logarithmique sur l'axe des y.
        smooth (bool): Lisser les courbes avec une moyenne mobile.
        window_size (int): Taille de la fenêtre pour la moyenne mobile.
        time_period (list): Liste avec [date_debut, date_fin] pour filtrer les données temporellement.
    """
    plt.figure(figsize=(14, 7))

    # Vérifier et convertir la colonne 'Date' en datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # 'coerce' transforme les erreurs en NaT

    for indicateur, couleur in charte_graphique.items():
        # Filtrer les données pour l'indicateur
        filtre = df['Indicateur'] == indicateur
        
        # Appliquer borne temporelle si demandé :
        if time_period != []:
            # Appliquer d'abord le filtre sur l'indicateur, puis appliquer la borne temporelle
            dates = df.loc[
                (df['Date'] >= pd.to_datetime(time_period[0])) & 
                (df['Date'] <= pd.to_datetime(time_period[1])) & 
                filtre, 'Date'
            ]
            valeurs = df.loc[
                (df['Date'] >= pd.to_datetime(time_period[0])) & 
                (df['Date'] <= pd.to_datetime(time_period[1])) & 
                filtre, taux
            ]
        else:
            # Si pas de borne temporelle, appliquer seulement le filtre sur l'indicateur
            dates = df.loc[filtre, 'Date']
            valeurs = df.loc[filtre, taux]

        # Appliquer un lissage si demandé
        if smooth:
            valeurs = valeurs.rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Tracer la courbe
        plt.plot(dates, valeurs, color=couleur, linewidth=0.8, label=indicateur)
    
    # Ajouter des titres, légendes, et la grille
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Appliquer une échelle logarithmique si demandé
    if use_log_scale:
        plt.yscale('log')

    # Ajustement et affichage
    plt.tight_layout()
    plt.show()

def tracer_evolution_taux_relatif_lisse(
    df, charte_graphique, 
    title="Évolution des taux d'infractions (Indice 1 en 1996)", 
    xlabel="Date", ylabel="Taux normalisé"
    ):
    """
    Trace l'évolution des taux normalisés pour une liste d'indicateurs avec lissage.

    Args:
        df (pd.DataFrame): Le dataframe contenant les données.
        charte_graphique (dict): Dictionnaire {nom_indicateur: couleur}.
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe des x.
        ylabel (str): Label de l'axe des y.
    """
    window_length = 200
    polyorder = 5

    plt.figure(figsize=(14, 7))

    # Vérifier et convertir la colonne 'Date' en datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convertir, remplacer erreurs par NaT

    for indicateur, couleur in charte_graphique.items():
        # Extraire les données pour l'indicateur
        filtre = df['Indicateur'] == indicateur
        data_indicateur = df.loc[filtre].copy()

        # Trier les données par date pour garantir le bon affichage
        data_indicateur = data_indicateur.sort_values(by='Date').reset_index(drop=True)

        # Normaliser les valeurs en divisant par la valeur initiale (première valeur)
        valeur_initiale = data_indicateur.iloc[0]['Taux (/10 000)']
        data_indicateur['Taux normalisé'] = data_indicateur['Taux (/10 000)'] / valeur_initiale

        # Appliquer un lissage aux taux normalisés avec Savitzky-Golay
        if len(data_indicateur) >= window_length:
            taux_lisse = savgol_filter(data_indicateur['Taux normalisé'], window_length=window_length, polyorder=polyorder)
        else:
            taux_lisse = data_indicateur['Taux normalisé']  # Pas de lissage si pas assez de points

        # Tracer la courbe
        plt.plot(
            data_indicateur['Date'], 
            taux_lisse, 
            color=couleur, 
            linewidth=0.8, 
            label=indicateur
        )

    # Ajouter des titres, légendes et la grille
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustement et affichage
    plt.tight_layout()
    plt.show()

# Fonction pour tracer le boxplot avec couleurs personnalisées pour chaque saison
def boxplot_indicateur_par_saison(df, indicateur):
    # Ajouter la colonne 'Saison' au dataframe
    
    # Filtrer les données pour l'indicateur spécifié
    df_indicateur_filtre = df[df['Indicateur'] == indicateur]
    
    # Définir les couleurs pour chaque saison
    saison_colors = {
        'Hiver': 'cornflowerblue',  # Bleu clair pour l'hiver
        'Printemps': 'limegreen',  # Vert clair pour le printemps
        'Été': 'gold',  # Jaune orangé pour l'été
        'Automne': 'brown'  # Marron-rouge pour l'automne
    }
    
    # Créer le boxplot avec la palette de couleurs définie
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Saison', y='Taux (/10 000)', data=df_indicateur_filtre, hue='Saison',
                palette=saison_colors)
    
    # Ajouter un titre et des labels
    plt.title(f'Boxplot du taux de {indicateur} selon les saisons')
    plt.xlabel('Saison')
    plt.ylabel('Taux (/10 000)')
    
    # Afficher le plot
    plt.show()

def évolution_indicateur(df, indicateur):
    # Définir le nombre de cartes par ligne
    cartes_par_ligne = 7
    
    # Récupérer toutes les années uniques comme chaînes de caractères dans les données
    annees = df['Année'].unique()
    annees = sorted(annees)
    
    # Calculer le nombre de lignes nécessaires
    total_annees = len(annees)
    lignes = (total_annees // cartes_par_ligne) + (1 if total_annees % cartes_par_ligne != 0 else 0)
    
    # Créer une figure avec une taille ajustée pour les petites cartes
    fig, axes = plt.subplots(nrows=lignes, ncols=cartes_par_ligne, figsize=(12, lignes * 2))
    
    # Aplatir la liste des axes pour y accéder plus facilement
    axes = axes.flatten()

    # Créer une liste pour stocker les valeurs des taux pour l'échelle partagée
    taux_values = []

    # Parcours des années dans les données (qui sont des chaînes de caractères)
    for idx, annee in enumerate(annees):
        # Filtrer les données pour l'indicateur et l'année en cours
        df_filtre = df[(df['Année'] == annee) & (df['Indicateur'] == indicateur)]




        # Ajouter une colonne 'geometry' en mappant le dictionnaire de géométrie sur la colonne 'Département'
        df_filtre["geometry"] = df_filtre["Département"].map(dictionnaire_geo)

        # Créer un GeoDataFrame avec la nouvelle colonne 'geometry'
        gdf = gpd.GeoDataFrame(df_filtre, geometry="geometry")
        
        # Ajouter les valeurs des taux dans la liste pour l'échelle partagée
        taux_values.extend(gdf['Taux (/10 000)'].dropna().tolist())
        
        # Vérifier si le GeoDataFrame n'est pas vide
        if not gdf.empty:
            # Afficher la carte avec le taux (%) sur l'axe correspondant
            gdf.plot(column='Taux (/10 000)', cmap=charte_graphique2.get(f'{indicateur}'), ax=axes[idx], legend=False)
            
            # Titre de la carte
            axes[idx].set_title(f"{annee}", fontsize=6)
            axes[idx].axis("off")  # Enlever les axes
            axes[idx].set_aspect(1.4)
        else:
            axes[idx].axis("off")  # Si aucune donnée, ne pas afficher d'axes

    # Créer un ScalarMappable pour la légende partagée
    sm = plt.cm.ScalarMappable(cmap=charte_graphique2.get(f'{indicateur}'), norm=mpl.colors.Normalize(vmin=min(taux_values), vmax=max(taux_values)))
    sm.set_array([])  # Nécessaire pour la légende
    
    # Ajouter la légende commune à la dernière case vide (celle qui reste vide)
    cbar = fig.colorbar(sm, 
                        ax=axes[-1], 
                        orientation='horizontal', 
                        fraction=0.045, 
                        pad=0.06, 
                        label="Occurences pour \n 10 000 habitants")

    # Supprimer le contour de la dernière case où la légende est placée
    axes[-1].axis("off")

    fig.suptitle(f"{indicateur}", fontsize=13, fontweight='bold', y=0.68) # titre commun

    # Ajuster l'espacement pour les sous-graphes et la légende
    plt.subplots_adjust(hspace=0.1, bottom=0, top=0.6)  # Ajuster l'espacement vertical et l'espacement en bas
    
    # Afficher la figure
    plt.show()


def evolution_indicateur_animation(df, indicateur, dictionnaire_geometrie):
    import os
    from IPython.display import display, Image
    from matplotlib import animation, colors

    plt.ioff()
    
    # Préparer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Créer une liste des années de 1996 à 2022
    annees = list(range(1996, 2023))
    
    # Fonction d'initialisation pour l'animation
    def init():
        ax.clear()
        ax.set_title(f"{indicateur} - Initialisation")
        ax.axis("off")
        return []
    
    # Fonction de mise à jour pour chaque frame de l'animation
    def update(frame):
        ax.clear()
        annee = annees[frame]
        
        # Filtrer les données pour l'indicateur et l'année en cours
        df_filtre = df[(df['Année'] == annee) & (df['Indicateur'] == indicateur)].copy()
        
        # Ajouter les géométries depuis le dictionnaire
        df_filtre['geometry'] = df_filtre['Département'].map(dictionnaire_geometrie)
        

        
        # Créer un GeoDataFrame à partir du DataFrame filtré
        gdf = gpd.GeoDataFrame(df_filtre, geometry='geometry')
        
        # Vérifier si le GeoDataFrame n'est pas vide
        if not gdf.empty:
            # Calculer les limites de couleur
            vmin = gdf['Taux (/10 000)'].min()
            vmax = gdf['Taux (/10 000)'].max()
            
            # Tracer la carte
            gdf.plot(column='Taux (/10 000)', 
                     cmap=charte_graphique2.get(f'{indicateur}'), 
                     ax=ax, 
                     legend=False,
                     vmin=vmin,
                     vmax=vmax,
                     edgecolor='0.8',
                     linewidth=0.7)
            
            # Titre de la carte
            ax.set_title(f"{indicateur} - {annee}")
        else:
            print(f"Année {annee} : Aucune donnée disponible pour {indicateur}")
            print(df_filtre)
        
        ax.axis("off")
        ax.set_aspect('equal') 
        ax.set_aspect(1.4)  # Étirement vertical de la carte 
        
        return []
    
    # Créer l'animation
    anim = animation.FuncAnimation(fig, 
                                   update, 
                                   init_func=init,
                                   frames=len(annees), 
                                   interval=500,  # 500 ms entre chaque frame
                                   blit=False)
    
    # Ajouter une barre de couleur
    vmin = df[df['Indicateur'] == indicateur]['Taux (/10 000)'].min()
    vmax = df[df['Indicateur'] == indicateur]['Taux (/10 000)'].max()
    sm = plt.cm.ScalarMappable(cmap=charte_graphique2.get(f'{indicateur}'), norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1, label="Occurences pour 10 000 habitants")
    
    # Sauvegarder l'animation au format GIF
    os.makedirs('animations', exist_ok=True)  # Crée un dossier 'animations' s'il n'existe pas
    
    # Chemin de sauvegarde
    save_path = f'animations/evolution_{indicateur.replace(' ', '_')}.gif'
    
    # Sauvegarde en GIF
    anim.save(save_path, writer='pillow', fps=2)

    # Afficher l'animation
    display(Image(filename=save_path))
    
    print(f"Animation sauvegardée dans {save_path}")
    
    return anim
