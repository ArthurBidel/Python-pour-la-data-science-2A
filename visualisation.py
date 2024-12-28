# Gestion des fichiers et des chemins
import os
from pathlib import Path

# Traitement des données
import numpy as np
import pandas as pd
import geopandas as gpd

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors, dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from IPython.display import Image, display

# Outils supplémentaires
import requests
from scipy.signal import savgol_filter
import s3fs


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


def tracer_evolution_taux(
    df, charte_graphique, taux="Taux (/10 000)", title="Évolution des taux d'infractions", 
    xlabel="Date", ylabel="Nombre d'occurence pour 10 000 habitants", use_log_scale=False, 
    smooth=False, window_size=100, time_period=[]
):
    """
    Trace l'évolution d'une valeur à choisir, permet de choisir une échelle logarithmique et/ou un lissage.

    Args:
        df (pd.DataFrame): Le dataframe contenant les données.
        charte_graphique (dict): Dictionnaire {Indicateur : Couleur}.
        taux (str): Colonne du dataframe contenant les valeurs à tracer.
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe des ordonnées.
        ylabel (str): Label de l'axe des abscisses.
        use_log_scale (bool): Option échelle logarithmique sur l'axe des ordonnées.
        smooth (bool): Option de lissage.
        window_size (int): Taille de la fenêtre pour la moyenne mobile du lissage.
        time_period (list): Liste avec [date_debut, date_fin] pour filtrer les données temporellement.
    """
    plt.figure(figsize=(14, 7))
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  

    for indicateur, couleur in charte_graphique.items():
        # Filtre l'indicateur choisi
        filtre = df['Indicateur'] == indicateur
        # Appliquer borne temporelle si demandé :
        if time_period != []:
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
            # Si pas de borne temporelle
            dates = df.loc[filtre, 'Date']
            valeurs = df.loc[filtre, taux]
        # Appliquer un lissage si demandé
        if smooth:
            valeurs = valeurs.rolling(window=window_size, center=True, min_periods=1).mean()

        plt.plot(dates, valeurs, color=couleur, linewidth=0.8, label=indicateur)
    # Appliquer une échelle logarithmique si demandé
    if use_log_scale:
        plt.yscale('log')

    # Affichage
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def tracer_evolution_taux_relatif_lisse(
    df, charte_graphique, 
    title="Évolution des taux d'infractions (Indice 1 en 1996)", 
    xlabel="Date", ylabel="Taux normalisé"
    ):
    """
    Trace l'évolution des taux de délinquance normalisés avec un lissage de Savitzky-Golay.

    Args:
        df (pd.DataFrame): Uniquement utilisé pour df_indicateurs_nat.
        charte_graphique (dict): Dictionnaire {Indicateur : Couleur}.
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe des abscisses.
        ylabel (str): Label de l'axe des ordonnées.
    """
    window_length = 200
    polyorder = 5
    plt.figure(figsize=(14, 7))

    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convertir, remplacer erreurs par NaT

    for indicateur, couleur in charte_graphique.items():
        # Préparation et tri dans le df
        filtre = df['Indicateur'] == indicateur
        data_indicateur = df.loc[filtre].copy()
        data_indicateur = data_indicateur.sort_values(by='Date').reset_index(drop=True)
        valeur_initiale = data_indicateur.iloc[0]['Taux (/10 000)']
        data_indicateur['Taux normalisé'] = data_indicateur['Taux (/10 000)'] / valeur_initiale
        # Appliquer un lissage aux taux normalisés avec Savitzky-Golay
        if len(data_indicateur) >= window_length:
            taux_lisse = savgol_filter(data_indicateur['Taux normalisé'], window_length=window_length, polyorder=polyorder)
        else:
            taux_lisse = data_indicateur['Taux normalisé']  # Pas de lissage si pas assez de points
        plt.plot(
            data_indicateur['Date'], 
            taux_lisse, 
            color=couleur, 
            linewidth=0.8, 
            label=indicateur
        )
    # Affichage
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def boxplot_indicateur_par_saison(df, indicateur, title="Boxplot"):
    """
    Trace le boxplot pour la répartitionn annuelle (selon les saisons) d'un indicateur de criminalité à choisir.

    Args:
        df (pd.DataFrame): Doit contenir une colonne 'Indicateur'.
        indicateur : Une des valeurs de la colonne en question.
        title (str): Titre du graphique.
    """
    df_indicateur_filtre = df[df['Indicateur'] == indicateur]
    # Définition des couleurs pour chaque saison
    saison_colors = {
        'Hiver': 'cornflowerblue', 
        'Printemps': 'limegreen', 
        'Été': 'gold',
        'Automne': 'brown' 
    }
    # Tracer la figure
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Saison', y='Taux (/10 000)', data=df_indicateur_filtre, hue='Saison',
                palette=saison_colors)
    # Affichage
    plt.title(title, fontsize=14)
    plt.xlabel('Saison')
    plt.ylabel('Taux (/10 000)')
    plt.show()

def evolution_indicateur_animation(df, indicateur):
    """
    Renvoit une carte animée de la France métropolitaine avec la répartition d'un certain indicateur de criminalité pour chaque année.
    Sauvegarde l'animation au format gif dans le dossier 'animations'.

    Args:
        df (pd.DataFrame): Doit contenir une colonne 'Indicateur' et une colonne 'Taux(/10 000)'.
        indicateur : Une des valeurs de la colonne en question.
    """
    # Préparation
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 8))
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
        # Filtrage pour l'indicateur souhaité
        df_filtre = df[(df['Année'] == annee) & (df['Indicateur'] == indicateur)].copy()
        # Ajout des géométries depuis le dictionnaire
        df_filtre['geometry'] = df_filtre['Département'].map(dictionnaire_geo)
        gdf = gpd.GeoDataFrame(df_filtre, geometry='geometry')
        if not gdf.empty:
            vmin = gdf['Taux (/10 000)'].min()
            vmax = gdf['Taux (/10 000)'].max()
            gdf.plot(column='Taux (/10 000)', 
                     cmap=charte_graphique2.get(f'{indicateur}'), 
                     ax=ax, 
                     legend=False,
                     vmin=vmin,
                     vmax=vmax,
                     edgecolor='0.8',
                     linewidth=0.7)
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
    
    # Affichage et sauvegarde
    vmin = df[df['Indicateur'] == indicateur]['Taux (/10 000)'].min()
    vmax = df[df['Indicateur'] == indicateur]['Taux (/10 000)'].max()
    sm = plt.cm.ScalarMappable(cmap=charte_graphique2.get(f'{indicateur}'), norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1, label="Occurences pour 10 000 habitants")

    os.makedirs('animations', exist_ok=True)  # Crée un dossier 'animations' s'il n'existe pas
    save_path = f'animations/evolution_{indicateur.replace(' ', '_')}.gif'
    anim.save(save_path, writer='pillow', fps=2)

    display(Image(filename=save_path))
    print(f"Animation sauvegardée dans {save_path}")
    return anim

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

def animer_evolution_densite(df, colonne_densite):
    """
    Renvoit une carte animée de la France métropolitaine avec la densité pour chaque année.
    Sauvegarde l'animation au format gif dans le dossier 'animations'.

    Args:
        df (pd.DataFrame): Le DateFrame contenant une colonne 'Densité'.
        indicateur : Une des valeurs de la colonne en question.
    """
    # Préparation
    custom_greys_cmap = create_custom_greys_cmap()
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 8))
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
        df_filtre = df[df['Année'] == annee].copy()
        df_filtre['geometry'] = df_filtre['Département'].map(dictionnaire_geo)
        gdf = gpd.GeoDataFrame(df_filtre, geometry='geometry')
        if not gdf.empty:
            vmin = df[colonne_densite].min()
            vmax = df[colonne_densite].max()
            gdf.plot(column=colonne_densite, 
                     cmap=custom_greys_cmap, 
                     ax=ax, 
                     legend=False,
                     vmin=vmin,
                     vmax=vmax,
                     edgecolor='0.8',
                     linewidth=0.5)
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
                                   interval=500, 
                                   blit=False)
    
    # Affichage, sauvegarde et présentation
    vmin = df[colonne_densite].min()
    vmax = df[colonne_densite].max()
    sm = plt.cm.ScalarMappable(cmap=custom_greys_cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.036, pad=0.1, label="Densité (hab/km²)")
    
    os.makedirs('animations', exist_ok=True)
    save_path = f'animations/evolution_densite.gif'
    anim.save(save_path, writer='pillow', fps=2)

    display(Image(filename=save_path))
    print(f"Animation sauvegardée dans {save_path}")
    return anim

def evolution_idf_animation(df, densite):
    """
    Renvoit une carte animée de l'Ile-de-France avec la densité pour chaque année.
    Sauvegarde l'animation au format gif dans le dossier 'animations'.

    Args:
        df (pd.DataFrame): Le DataFrame contenant la colonne de 'Densité'.
        indicateur : Une des valeurs de la colonne en question.
    """
    custom_greys_cmap = create_custom_greys_cmap()
    plt.ioff()
    idf_codes = ['75', '77', '78', '91', '92', '93', '94', '95']
    df_idf = df[df['Département'].isin(idf_codes)].copy()
    df_idf['geometry'] = df_idf['Département'].map(dictionnaire_geo)
    nom_departement = {
        '75': 'Paris', '77': 'Seine-et-Marne', '78': 'Yvelines', '91': 'Essonne', 
        '92': 'Hauts-de-Seine', '93': 'Seine-Saint-Denis', '94': 'Val-de-Marne', '95': 'Val-d\'Oise'
    }
    df_idf['Nom_Departement'] = df_idf['Département'].map(nom_departement)
    gdf_idf = gpd.GeoDataFrame(df_idf, geometry='geometry')
    annees = sorted(df_idf['Année'].unique())
    vmin = df_idf[densite].min()
    vmax = df_idf[densite].max()
    # Préparer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    # Configurer la colorbar une seule fois
    sm = plt.cm.ScalarMappable(cmap=custom_greys_cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # Nécessaire pour éviter les erreurs avec colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', label=f"{densite}")
    
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
        # Filtre pour l'année
        gdf_frame = gdf_idf[gdf_idf['Année'] == annee]
        # Définir les limites des axes pour centrer sur l'Île-de-France
        bounds = gdf_frame.total_bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        
        gdf_frame.plot(
            column=densite,
            cmap=custom_greys_cmap, 
            ax=ax,
            legend=False,
            edgecolor='0.8',
            linewidth=0.7
        )
        ax.set_title(f"{densite} - {annee}", fontsize=14)
        ax.axis("off")
        ax.set_aspect(1.4)

        # Ajouter les noms des départements
        for _, row in gdf_frame.iterrows():
            # Récupérer le centroïde du département
            centroid = row['geometry'].centroid
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
    
    # Sauvegarde et affichage
    os.makedirs('animations', exist_ok=True)
    save_path = 'animations/evolution_idf_densite.gif'
    anim.save(save_path, writer='pillow', fps=2)
    
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
        # Extremum xtremum
        nombre_1996 = df.loc[
                        (df['Date'] == date1) & (df['Indicateur'] == indicateur), 'Taux (/10 000)'
                        ].iloc[0]
        nombre_2022 = df.loc[
                        (df['Date'] == date2) & (df['Indicateur'] == indicateur), 'Taux (/10 000)'
                        ].iloc[0]
        # Calcul de l'évolution en %
        evolution = ((nombre_2022 - nombre_1996) / nombre_1996) * 100
        # Affichage
        print(f" {indicateur} entre {date1} et {date2}: {evolution} %")
    
    except KeyError:
        raise KeyError(f"L'indicateur '{indicateur}' ou la colonne 'Année' est introuvable dans le dataframe.")

    except IndexError:
        raise IndexError(f"Les données pour 1996 ou 2022 sont manquantes dans le dataframe.")

def camembert(df):
    """
    Crée un diagramme camembert montrant la répartition des types de textes de loi.
    
    Args:
    df (DataFrame): Le DataFrame contenant les données avec une colonne 'Nature'.
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

    plt.figure(figsize=(8, 6))
    
    # Attribuer les couleurs (gris par défaut si type non spécifié)
    colors = [color_mapping.get(str(type_text).upper(), 'gray') for type_text in type_counts.index]
    
    wedges, texts, autotexts = plt.pie(percentages, 
                                      labels=type_counts.index,
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      pctdistance=0.85,
                                      explode=[0.05] * len(type_counts))
    
    # Affichage et apparence 
    plt.title("Figure 6 - Répartition globale des types de textes de loi (toutes années confondues)",fontsize=14, pad=20, x=0.5)
    plt.setp(autotexts, size=10, weight='bold')
    plt.setp(texts, size=10)
    legend_labels = [f"{index} ({count:,} textes)" for index, count in type_counts.items()]
    plt.legend(wedges, legend_labels,
              title="Types de textes",
              loc="lower right",
              bbox_to_anchor=(1, 0, 0.5, 1))
    plt.axis('equal')  
    plt.gca().set_axis_off() 
    plt.tight_layout()
    plt.show()

def tri_occurrence(df):
    '''
    Calcule le nombre de publication par mois pour chaque type de textes de loi.

    Args: 
    df: Le DateFrame contenant chacun des textes de lois et leur nature (df_loda ici)
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
    df_résultat.fillna(0, inplace=True) # Remplacer les NaN par 0 pour les colonnes ajoutées
    df_résultat['day']=1
    df_résultat['Mois'] = df_résultat['Mois'].astype(int)
    df_résultat.rename(columns={'Mois': 'month'}, inplace=True) # On renomme de cette manière pour pouvoir utiliser datetime
    df_résultat['Année'] = df_résultat['Année'].astype(int)
    df_résultat.rename(columns={'Année': 'year'}, inplace=True) 
    df_résultat.head()
    df_résultat['Date'] = pd.to_datetime(df_résultat[['year', 'month', 'day']])
    df_res = df_résultat.melt(id_vars=['year', "month", 'day', 'Date'],
                      var_name='Indicateur',  
                      value_name='Nombre')  
    df_sorted = df_res.sort_values(by='Date').reset_index(drop=True)
    df_sorted['Cumulatif'] = df_sorted.groupby('Indicateur')['Nombre'].cumsum()
    return(df_sorted)

def plot_histogram(df, types, numero_figure ='Figure -'):
    """
    Trace un histogramme empilé du nombre de textes par mois pour différents types de textes.
    
    Parameters:
    df (DataFrame): Le DataFrame contenant les données.
    types (list): Liste des types de textes à analyser (ex: ['LOI', 'DECRET']).
    charte_graphique3 (dict): Dictionnaire des couleurs par type de texte.
    numero_figure = Pour ajuster le titre
    """
    # Préparation
    if df.empty:
        print("Il n'y a pas de donnée existante")
        return
    df_filtered = df[df['Indicateur'].isin(types)].copy()
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
    
    # Créer un pivot table pour avoir les données dans le format requis
    df_pivot = df_filtered.pivot_table(
        index='Date',
        columns='Indicateur',
        values='Nombre',
        aggfunc='sum',
        fill_value=0
    )
    
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
                  color=charte_graphique3[type_texte],
                  alpha=0.7)
        )
        bottom += df_pivot[type_texte]
    
    # Apparence et affichage
    ax.set_title(f"{numero_figure}Répartition des textes par type : {', '.join(types)}", 
                fontsize=14, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Nombre de textes', fontsize=12)
    ax.legend(title="Types de textes", 
             bbox_to_anchor=(1.05, 1), 
             loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def nb_lignes_traitant(df, keyword, column='Titre'):
    """
    Compte le nombre de lignes dans un DataFrame où un mot clé apparaît dans une colonne donnée.

    Args: 
    df (DataFrame)
    keyword: Mot clé qu'on recherche dans le titre
    colum='Titre': Pour parcourir chacun des textes
    """
    mask = df[column].str.contains(keyword, case=False, na=False)
    return mask.sum()





# FONCTION EN ATTENTE DE SAVOIR SI UTILISEE

def filter_rows_with_keyword(df, keyword): # UTILISE ?
    """
    Filtre les lignes d'un DataFrame où un mot clé apparaît dans une colonne donnée.

    Args:
    df(DataFrame)
    keyword : mot clé d'intérêt
    """
    filtered_df = df[df['Titre'].str.contains(keyword, case=False, na=False)]
    return filtered_df

keywords_laws = ["délinquance", "crime", "délit", "Homicides", "Vols", "Stupéfiants", "Escroquerie",
                "Contrefaçon", "Sequestrations", "Recels", "Proxénétisme", "Menaces", "Cambriolages",
                "infraction", "Attentats", "dégradations", "Outrages"]

def count_crime_keywords(df, column='Titre'): # UTILISE ?
    """
    Idk what it doas help Wiwi
    """
    if df.empty:
        print("Le DataFrame est vide.")
        return
    all_text = ' '.join(df[column].dropna()).lower()
    word_counts = {word: all_text.count(word) for word in keywords_laws}
    return pd.DataFrame.from_dict(word_counts, orient='index', columns=['Fréquence']).sort_values(by='Fréquence', ascending=False)




# FONCTION NON UTILISEE ANYMORE

def get_mean(df, indicateur, date_comp): # PAS UTILISE
    """
    Calcule la moyenne du taux d'infraction pour l'indicateur et permet de la comparer avec le taux actuel.

    Args :
        df (pd.DataFrame): Le dataframe contenant les colonnes 'Date', 'Indicateur', et 'Taux (/10 000)'.
        indicateur (str): Le libellé de l'indicateur à analyser.
        date_comp (str): La date choisie pour calculer l'écart (au format 'YYYY-MM-DD').

    Returns:
        tuple: Moyenne du taux (float), taux à la date voulue (float), écart à la moyenne à la date choisie (float).
    """
    try:
        indicateur_data = df[df['Indicateur'] == indicateur]
        # Calcul de la moyenne
        mean_taux = indicateur_data['Taux (/10 000)'].mean()
        # Récupérer le taux à la date choisie
        taux_date = indicateur_data.loc[
            indicateur_data['Date'] == date_comp, 'Taux (/10 000)'
        ].iloc[0]
        # Calculer l'écart à la moyenne
        ecart = taux_date - mean_taux
        # Affichage
        print(f"Nombre d'infractions pour 10 000 habitants moyen : {mean_taux}")
        print(f"Nombre au {date_comp} : {taux_date}")
        print(f"Différence : {ecart}")
    
    except KeyError:
        raise KeyError("Assurez-vous que les colonnes 'Date', 'Indicateur', et 'Taux (/10 000)' sont présentes dans le dataframe.")

    except IndexError:
        raise IndexError(f"Aucune donnée trouvée pour la date {date_comp} et l'indicateur '{indicateur}'.")

def évolution_indicateur(df, indicateur): # PAS UTILISE 
    """
    Renvoit plusieurs carte de la France métropolitaine avec la répartition d'un certain indicateur pour chaque année.

    Args:
        df (pd.DataFrame): Doit contenir une colonne 'Indicateur'.
        indicateur : Une des valeurs de la colonne en question.
    """
    cartes_par_ligne = 7
    annees = df['Année'].unique()
    annees = sorted(annees)
    total_annees = len(annees)
    lignes = (total_annees // cartes_par_ligne) + (1 if total_annees % cartes_par_ligne != 0 else 0)
    fig, axes = plt.subplots(nrows=lignes, ncols=cartes_par_ligne, figsize=(12, lignes * 2))
    axes = axes.flatten()
    # Echelle partagée
    taux_values = []

    for idx, annee in enumerate(annees):
        df_filtre = df[(df['Année'] == annee) & (df['Indicateur'] == indicateur)]
        # Ajouter une colonne de géométrie avec le dictionnaire défini plus haut
        df_filtre.loc[:, "geometry"] = df_filtre["Département"].map(dictionnaire_geo)
        gdf = gpd.GeoDataFrame(df_filtre, geometry="geometry")
        # Ajouter les valeurs des taux dans la liste pour l'échelle partagée
        taux_values.extend(gdf['Taux (/10 000)'].dropna().tolist())

        if not gdf.empty:
            gdf.plot(column='Taux (/10 000)', cmap=charte_graphique2.get(f'{indicateur}'), ax=axes[idx], legend=False)
            axes[idx].set_title(f"{annee}", fontsize=6)
            axes[idx].axis("off")  
            axes[idx].set_aspect(1.4)
        else:
            axes[idx].axis("off")

    # Légende partagée
    sm = plt.cm.ScalarMappable(cmap=charte_graphique2.get(f'{indicateur}'), norm=mpl.colors.Normalize(vmin=min(taux_values), vmax=max(taux_values)))
    sm.set_array([])  
    cbar = fig.colorbar(sm, 
                        ax=axes[-1], 
                        orientation='horizontal', 
                        fraction=0.045, 
                        pad=0.06, 
                        label="Occurences pour \n 10 000 habitants")
    axes[-1].axis("off") # Supprime le contour de la dernière case où la légende est placée

    # Affichage et finalisation
    fig.suptitle(f"{indicateur}", fontsize=13, fontweight='bold', y=0.68) # Titre commun
    plt.subplots_adjust(hspace=0.1, bottom=0, top=0.6)  # Ajuste espacements
    plt.show()
