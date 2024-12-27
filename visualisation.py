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


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from IPython.display import Image, display
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

# Dictionnaire associant chaque indicateur (législatif) à une couleur spécifique
charte_graphique3 = {
    "Texte": "Magenta",
    "Arrete": "Green",
    "Loi": "Cyan",
    "Decret": "Pink",
    "Ordonnance": "Red",
}

all = ["Arrete", "Loi", "Decret", "Ordonnance"]

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
def boxplot_indicateur_par_saison(df, indicateur, title="Boxplot"):
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
    plt.title(title, fontsize=14)
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
        df_filtre.loc[:, "geometry"] = df_filtre["Département"].map(dictionnaire_geo)

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


from matplotlib.colors import LinearSegmentedColormap

# Définir une colormap personnalisée
def create_custom_greys_cmap():
    # Points de contrôle pour la colormap (0: blanc, 1: noir, ajustés pour nuances intermédiaires)
    colors = [
        (0.95, 0.95, 0.95),  # Très clair
        (0.8, 0.8, 0.8),    # Clair
        (0.6, 0.6, 0.6),    # Intermédiaire
        (0.4, 0.4, 0.4),    # Assez foncé
        (0.2, 0.2, 0.2)     # Foncé mais pas noir
    ]
    return LinearSegmentedColormap.from_list("CustomGreys", colors)

# Créer la colormap
custom_greys_cmap = create_custom_greys_cmap()


def animer_evolution_densite(df, colonne_densite, dictionnaire_geometrie):
    import os
    from IPython.display import display, Image
    from matplotlib import animation, colors
    import geopandas as gpd

    plt.ioff()
    
    # Préparer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Créer une liste des années disponibles dans le DataFrame
    annees = sorted(df['Année'].unique())
    
    # Fonction d'initialisation pour l'animation
    def init():
        ax.clear()
        ax.set_title("Initialisation")
        ax.axis("off")
        return []
    
    # Fonction de mise à jour pour chaque frame de l'animation
    def update(frame):
        ax.clear()
        annee = annees[frame]
        
        # Filtrer les données pour l'année en cours
        df_filtre = df[df['Année'] == annee].copy()
        
        # Ajouter les géométries depuis le dictionnaire
        df_filtre['geometry'] = df_filtre['Département'].map(dictionnaire_geometrie)
        
        # Créer un GeoDataFrame à partir du DataFrame filtré
        gdf = gpd.GeoDataFrame(df_filtre, geometry='geometry')
        
        # Vérifier si le GeoDataFrame n'est pas vide
        if not gdf.empty:
            # Calculer les limites de couleur
            vmin = df[colonne_densite].min()
            vmax = df[colonne_densite].max()
            
            # Tracer la carte
            gdf.plot(column=colonne_densite, 
                     cmap=custom_greys_cmap,  # Colormap générique
                     ax=ax, 
                     legend=False,
                     vmin=vmin,
                     vmax=vmax,
                     edgecolor='0.8',
                     linewidth=0.5)
            
            # Titre de la carte
            ax.set_title(f"Densité de population - {annee}")
        else:
            print(f"Année {annee} : Aucune donnée disponible")
        
        ax.axis("off")
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
    vmin = df[colonne_densite].min()
    vmax = df[colonne_densite].max()
    sm = plt.cm.ScalarMappable(cmap=custom_greys_cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1, label="Densité (hab/km²)")
    
    # Sauvegarder l'animation au format GIF
    os.makedirs('animations', exist_ok=True)  # Crée un dossier 'animations' s'il n'existe pas
    
    # Chemin de sauvegarde
    save_path = f'animations/evolution_densite.gif'
    
    # Sauvegarde en GIF
    anim.save(save_path, writer='pillow', fps=2)

    # Afficher l'animation
    display(Image(filename=save_path))
    
    print(f"Animation sauvegardée dans {save_path}")
    
    return anim



def evolution_idf_animation(df, indicateur):
    import os
    from IPython.display import display, Image
    from matplotlib import animation, colors
    import geopandas as gpd
    import matplotlib.pyplot as plt
    
    plt.ioff()
    
    # Filtrer pour l'Île-de-France (codes 75, 77, 78, 91, 92, 93, 94, 95)
    idf_codes = ['75', '77', '78', '91', '92', '93', '94', '95']
    df_idf = df[df['Département'].isin(idf_codes)].copy()
    
    # Ajouter la géométrie à partir du dictionnaire
    df_idf['geometry'] = df_idf['Département'].map(dictionnaire_geo)

    nom_departement = {
        '75': 'Paris', '77': 'Seine-et-Marne', '78': 'Yvelines', '91': 'Essonne', 
        '92': 'Hauts-de-Seine', '93': 'Seine-Saint-Denis', '94': 'Val-de-Marne', '95': 'Val-d\'Oise'
    }
    df_idf['Nom_Departement'] = df_idf['Département'].map(nom_departement)

    gdf_idf = gpd.GeoDataFrame(df_idf, geometry='geometry')
    
    # Créer une liste des années
    annees = sorted(df_idf['Année'].unique())

    # Définir les limites des valeurs de densité pour la colorbar
    vmin = df_idf[indicateur].min()
    vmax = df_idf[indicateur].max()
    
    # Préparer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # Configurer la colorbar une seule fois
    sm = plt.cm.ScalarMappable(cmap=custom_greys_cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # Nécessaire pour éviter les erreurs avec colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', label=f"{indicateur}")
    
    # Fonction d'initialisation pour l'animation
    def init():
        ax.clear()
        ax.set_title("Initialisation")
        ax.axis("off")
        return []
    
    # Fonction de mise à jour pour chaque frame de l'animation
    def update(frame):
        ax.clear()
        annee = annees[frame]
        
        # Filtrer les données pour l'année en cours
        gdf_frame = gdf_idf[gdf_idf['Année'] == annee]
        
        # Définir les limites des axes pour centrer sur l'Île-de-France
        bounds = gdf_frame.total_bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        
        # Tracer la carte
        gdf_frame.plot(
            column=indicateur,
            cmap=custom_greys_cmap,  # Utilisez une cmap adaptée
            ax=ax,
            legend=False,
            edgecolor='0.8',
            linewidth=0.7
        )
        
        # Ajouter le titre de l'année
        ax.set_title(f"{indicateur} - {annee}", fontsize=14)
        ax.axis("off")
        ax.set_aspect(1.4)

        # Ajouter les noms des départements
        for _, row in gdf_frame.iterrows():
            # Récupérer le centroïde du département
            centroid = row['geometry'].centroid
            # Ajouter le texte sans boîte blanche, en gris foncé et avec une police plus fine
            ax.text(
                centroid.x, centroid.y, 
                row['Nom_Departement'], 
                ha='center', va='center', 
                fontsize=8, color='#666666', 
                fontweight='light',  
            )
        
        return []
    
    # Créer l'animation
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(annees),
        interval=500,
        blit=False
    )
    
    # Sauvegarder l'animation au format GIF
    os.makedirs('animations', exist_ok=True)
    save_path = 'animations/evolution_idf_densite.gif'
    anim.save(save_path, writer='pillow', fps=2)
    
    # Afficher l'animation
    display(Image(filename=save_path))
    print(f"Animation sauvegardée dans {save_path}")
    return anim



def get_increase(df, indicateur, date1, date2): 
    """
    Calcule l'évolution en pourcentage du taux de délinquance pour un indicateur spécifique entre 1996 et 2022.

    Args : 
        df : le dataframe, la fonction suppose qu'il a la structure de df_indicateurs_nat ou df_indicateurs_dep. 
        En particulier elle suppose l'existence d'une colonne 'Date', 'Indicateur' et 'Taux (/10 000)'.
        indicateur : un des huits indicateurs de délinquance
        date1, date2 : les bornes temporelles souhaitées au format '1996-01-01'

    Returns : L'évolution en pourcentage de l'indicateur
    """
    try:
        # Avoir les extremum
        nombre_1996 = df.loc[
                        (df['Date'] == date1) & (df['Indicateur'] == indicateur), 'Taux (/10 000)'
                        ].iloc[0]

        nombre_2022 = df.loc[
                        (df['Date'] == date2) & (df['Indicateur'] == indicateur), 'Taux (/10 000)'
                        ].iloc[0]
        
        # Calculer l'évolution en pourcentage
        evolution = ((nombre_2022 - nombre_1996) / nombre_1996) * 100

        print(f" {indicateur} entre {date1} et {date2}: {evolution} %")
    
    except KeyError:
        raise KeyError(f"L'indicateur '{indicateur}' ou la colonne 'Année' est introuvable dans le dataframe.")

    except IndexError:
        raise IndexError(f"Les données pour 1996 ou 2022 sont manquantes dans le dataframe.")



def get_mean(df, indicateur, date_comp):
    """
    Calcule la moyenne du taux d'infraction pour l'indicateur et permet de la comparer avec le taux actuel.

    Parameters:
        df (pd.DataFrame): Le dataframe contenant les colonnes 'Date', 'Indicateur', et 'Taux (/10 000)'.
        indicateur (str): Le libellé de l'indicateur à analyser.
        date_comp (str): La date choisie pour calculer l'écart (au format 'YYYY-MM-DD').

    Returns:
        tuple: Moyenne du taux (float), taux à la date voulue (float), écart à la moyenne à la date choisie (float).
    """
    try:
        # Filtrer les données pour l'indicateur donné
        indicateur_data = df[df['Indicateur'] == indicateur]
        
        # Calculer la moyenne du taux
        mean_taux = indicateur_data['Taux (/10 000)'].mean()
        
        # Récupérer le taux à la date choisie
        taux_date = indicateur_data.loc[
            indicateur_data['Date'] == date_comp, 'Taux (/10 000)'
        ].iloc[0]
        
        # Calculer l'écart à la moyenne
        ecart = taux_date - mean_taux
        
        print(f"Nombre d'infractions pour 10 000 habitants moyen : {mean_taux}")
        print(f"Nombre au {date_comp} : {taux_date}")
        print(f"Différence : {ecart}")
    
    except KeyError:
        raise KeyError("Assurez-vous que les colonnes 'Date', 'Indicateur', et 'Taux (/10 000)' sont présentes dans le dataframe.")
    except IndexError:
        raise IndexError(f"Aucune donnée trouvée pour la date {date_comp} et l'indicateur '{indicateur}'.")

def camembert(df):
    """
    Crée un diagramme camembert montrant la répartition des types de textes de loi.
    
    Parameters:
    df (DataFrame): Le DataFrame contenant les données avec une colonne 'Nature'
    """
    # Définir les couleurs pour chaque type de texte
    color_mapping = {
        "ARRETE": "Green",
        "LOI": "Cyan",
        "DECRET": "Pink",
        "ORDONNANCE": "Red"
    }
    
    # Calculer les pourcentages pour chaque type de texte
    type_counts = df['Nature'].value_counts()
    percentages = (type_counts / len(df) * 100).round(1)
    
    # Créer la figure
    plt.figure(figsize=(8, 6))
    
    # Attribuer les couleurs (gris par défaut si type non spécifié)
    colors = [color_mapping.get(str(type_text).upper(), 'gray') for type_text in type_counts.index]
    
    # Créer le camembert
    wedges, texts, autotexts = plt.pie(percentages, 
                                      labels=type_counts.index,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85,
                                      explode=[0.05] * len(type_counts))
    
    # Personnaliser l'apparence
    plt.title("Figure 6 - Répartition globale des types de textes de loi (toutes années confondues)",fontsize=14, pad=20, x=0.5)
    
    # Personnaliser les textes
    plt.setp(autotexts, size=10, weight='bold')
    plt.setp(texts, size=10)
    
    # Ajouter une légende avec les counts absolus
    legend_labels = [f"{index} ({count:,} textes)" for index, count in type_counts.items()]
    plt.legend(wedges, legend_labels,
              title="Types de textes",
              loc="lower right",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()

    plt.show()
    
    return plt.gcf(), plt.gca()

def tri_occurrence(df):
    '''
    Fonction récapitulative de tout le réagencement (que tu avais fait AnhLinh)
    '''
    df_résultat = df.groupby(["Année", "Mois"]).size().reset_index(name="Texte")

    types = df['Nature'].unique()

    for type in types : 
        type_lower = type.capitalize()
        grouped_type = (df[df["Nature"] == type]
            .groupby(["Année", "Mois"])
            .size()
            .reset_index(name=f"{type_lower}")
        )
        df_résultat = pd.merge(df_résultat,
                                grouped_type,
                                how="left",
                                on=["Année", "Mois"]
                            )

    # Remplacer les NaN par 0 pour les colonnes ajoutées
    df_résultat.fillna(0, inplace=True)

    df_résultat['day']=1
    df_résultat['Mois'] = df_résultat['Mois'].astype(int)
    df_résultat.rename(columns={'Mois': 'month'}, inplace=True) # Bizarre car le reste est en fr?
    df_résultat['Année'] = df_résultat['Année'].astype(int)
    df_résultat.rename(columns={'Année': 'year'}, inplace=True) # Bizarre car le reste est en fr?
    df_résultat.head()


    df_résultat['Date'] = pd.to_datetime(df_résultat[['year', 'month', 'day']])
    df_res = df_résultat.melt(id_vars=['year', "month", 'day', 'Date'],
                      var_name='Indicateur',  
                      value_name='Nombre')  

    df_sorted = df_res.sort_values(by='Date').reset_index(drop=True)
    df_sorted['Cumulatif'] = df_sorted.groupby('Indicateur')['Nombre'].cumsum()

    return(df_sorted)

def plot_histogram(df, types, charte_graphique=charte_graphique3, numero_figure ='Figure -'):
    """
    Trace un histogramme empilé du nombre de textes par mois pour différents types de textes.
    
    Parameters:
    df (DataFrame): Le DataFrame contenant les données.
    types (list): Liste des types de textes à analyser (ex: ['LOI', 'DECRET']).
    charte_graphique3 (dict): Dictionnaire des couleurs par type de texte.
    """
    if df.empty:
        print("Il n'y a pas de donnée existante")
        return
        
    # Filtrer les données par les types de texte
    df_filtered = df[df['Indicateur'].isin(types)].copy()
    
    # S'assurer que la colonne 'Date' est de type datetime
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
    
    # Créer un pivot table pour avoir les données dans le format requis
    df_pivot = df_filtered.pivot_table(
        index='Date',
        columns='Indicateur',
        values='Nombre',
        aggfunc='sum',
        fill_value=0
    )
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Tracer l'histogramme empilé
    bottom = np.zeros(len(df_pivot))
    bars = []
    
    for type_texte in df_pivot.columns:
        bars.append(
            ax.bar(df_pivot.index, df_pivot[type_texte], 
                  bottom=bottom, 
                  width=20, 
                  label=type_texte,
                  color=charte_graphique[type_texte],
                  alpha=0.7)
        )
        bottom += df_pivot[type_texte]
    
    # Ajouter des titres et des labels
    ax.set_title(f"{numero_figure}Répartition des textes par type : {', '.join(types)}", 
                fontsize=14, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Nombre de textes', fontsize=12)
    
    # Ajouter une légende
    ax.legend(title="Types de textes", 
             bbox_to_anchor=(1.05, 1), 
             loc='upper left')
    
    # Améliorer l'aspect du graphique
    plt.xticks(rotation=45)
    
    # Ajuster automatiquement les marges pour éviter que la légende soit coupée
    plt.tight_layout()
    
    plt.show()

    return fig, ax

def nb_lignes_traitant(df, keyword, column='Titre'):
    """
    Compte le nombre de lignes dans un DataFrame où un mot clé apparaît dans une colonne donnée.
    """
    mask = df[column].str.contains(keyword, case=False, na=False)
    return mask.sum()

def filter_rows_with_keyword(df, keyword):
    """
    Filtre les lignes d'un DataFrame où un mot clé apparaît dans une colonne donnée.
    """
    filtered_df = df[df['Titre'].str.contains(keyword, case=False, na=False)]
    return filtered_df

keywords_laws = [
    "vol", "infraction", "fraude", "victime", "menace", "crime", "trafic", "viol", 
    "exécution", "violence", "dégradation", "corruption", "tir", "contravention", 
    "peine", "contrefaçon", "délit", "procès-verbal", "escroquerie", "fraude fiscale", 
    "garde à vue", "responsabilité pénale", "génocide", "harcèlement", "outrage", 
    "attentat", "condamnation", "maltraitance", "dommages", "accident", "agression", 
    "meurtre", "emprisonnement", "mutilation", "exploitation", "inculpation", 
    "répression", "pillage", "racket", "intimidation", "usurpation", "abandon", 
    "violence conjugale", "attentat terroriste", "légitime défense", "banditisme", 
    "incendie criminel", "extorsion", "abus de pouvoir", "tentative de meurtre", 
    "violence policière", "assassinat", "évasion", "assistance aux criminels", 
    "falsification", "blanchiment", "punition", "récidive", "détournement", 
    "menace terroriste", "armes", "violences urbaines", "enlèvement", 
    "otage", "trahison"
]


def count_crime_keywords(df, column='Titre'):
    if df.empty:
        print("Le DataFrame est vide.")
        return
    
    all_text = ' '.join(df[column].dropna()).lower()
    word_counts = {word: all_text.count(word) for word in keywords_laws}
    
    return pd.DataFrame.from_dict(word_counts, orient='index', columns=['Fréquence']).sort_values(by='Fréquence', ascending=False)
