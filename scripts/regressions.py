from IPython.display import clear_output
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.api import add_constant

def mise_en_forme_reg(keyword, indicateur):
    '''
    Prépare un DataFrame structuré pour effectuer une analyse de régression.

    Args:
    keyword (str): Mot-clé utilisé pour filtrer les articles dans le DataFrame `df_loda`.
    indicateur (str): Nom de l'indicateur de criminalité choisi.
    
    Returns: Un DataFrame contenant les données fusionnées et prêtes pour une analyse de régression.
    '''
    # Préparation de LODA
    df_loda_reg = df_loda.drop([ "Unnamed: 0", "ID", "Date", "Nature", "Etat", "Origine", "Date Publication", "Mois"], axis = 1)
    df_loda_filtre = filter_keyword(df_loda_reg,fr"\b{keyword}s?\b")
    df_loda_reg_filtre = df_loda_filtre.groupby("Année").size().reset_index(name="Nombre d'articles")
    
    # Préparation du taux de pauvreté 
    df_indicateurs_nat.head()
    df_indicateurs_nat_reg = df_indicateurs_nat.loc[: ,["Année", "Taux de pauvreté (%)"]]
    df_pauvrete_percent = df_indicateurs_nat_reg.drop_duplicates()

    # Préparation des autres régresseurs et filtrage sur l'indicateur de criminalité choisi
    df_indicateurs_reg = df_indicateurs_dep.drop([ "Unnamed: 0", "Superficie (km2)", "Nombre" , "Département"], axis = 1)
    df_indicateurs_reg = df_indicateurs_reg[df_indicateurs_reg["Indicateur"] == indicateur]

    df_pauvrete_loda_nbr= pd.merge(df_pauvrete_percent, df_loda_reg_filtre, on="Année", how="outer")
    df_reg =pd.merge(df_pauvrete_loda_nbr, df_indicateurs_reg, on = "Année", how = "outer")

    df_reg["Nombre d'articles"] = df_reg["Nombre d'articles"].fillna(0) # On remplace les valeurs manquante par 0
    df_reg = df_reg.drop(["Indicateur"], axis = 1) # On se débarasse de la colonne 'Indicateur' sur laquelle on ne régresse pas
    df_reg = df_reg.set_index(['Nom Département', 'Année']) # On met en index les colonnes qui indices nos variables

    return(df_reg)

def regression(df, title, entity_effects=False, lag = False):

    """
    Effectue une régression sur données de panel, affiche le résumé et trace le graphique.
    Si entity_effects=True, calcule également les effets fixes par département.

    Args:
        df (DataFrame): Le DataFrame contenant les données.
        title (str): Ajuster le titre du graphique.
        entity_effects (bool): Inclure ou non les effets fixes des entités (par défaut: False).

    Returns:
        dict: Résultats de la régression, prédictions, résidus et, si applicable, effets fixes
    """

    if lag :
        # Création des variables décalées
        df["Articles_lag"+str(lag)] = df.groupby('Nom Département')["Nombre d'articles"].shift(lag)
        df["Taux_lag"+str(lag)] = df.groupby('Nom Département')['Taux (/10 000)'].shift(lag)

        # Supprimer les lignes avec des NaN créées par le décalage
        df["Articles_lag"+str(lag)] = df["Articles_lag"+str(lag)].fillna(0)
        df["Taux_lag"+str(lag)] = df["Taux_lag"+str(lag)].fillna(0)

        # Régression
        y_lag = df['Taux (/10 000)']
        X_lag = df[["Taux de pauvreté (%)", "Articles_lag"+str(lag), "Population", "Densité"]]
        X_lag = add_constant(X_lag)

        # Modèle de régression
        model = PanelOLS(y_lag, X_lag, entity_effects=entity_effects)



    else :
        # Définir les variables dépendante et explicatives
        y = df['Taux (/10 000)']
        X = df[["Taux de pauvreté (%)","Nombre d'articles", "Population", "Densité"]]
        X = add_constant(X)
        # Modèle de régression
        model = PanelOLS(y, X, entity_effects=entity_effects)

    results = model.fit()
    # Ajouter les prédictions et les résidus dans le DataFrame
    df['Predicted'] = results.predict().fitted_values
    df['Residuals'] = df['Taux (/10 000)'] - df['Predicted']

    # Afficher le résumé
    print(results.summary)

    # Calcul des effets fixes si demandé
    if entity_effects:
        try:
            effects_fixed = results.estimated_effects
            # Extraire l'effet fixe correspondant au Département 
            departments = effects_fixed.index.get_level_values(0)  
            effects_fixed.index = departments  
            print("\nEffets fixes par département:\n", effects_fixed)
        except KeyError as e:
            print(f"Erreur lors de l'accès aux effets fixes: {e}")
        except AttributeError:
            print("Les effets fixes ne sont pas disponibles dans cette configuration.")
    else :
        effects_fixed = 'Ici, les effets fixes ne sont pas calculer'

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    if lag : 
        sns.scatterplot(        
        x=df["Articles_lag"+str(lag)], 
        y=df['Taux (/10 000)'], 
        color='blue', 
        alpha=0.6, 
        label='Observations'
        )
        sns.lineplot(
            x=df["Articles_lag"+str(lag)], 
            y=df['Predicted'], 
            color='red', 
            label='Prédictions'
        )

    else: 
        sns.scatterplot(        
        x=df["Nombre d'articles"], 
        y=df['Taux (/10 000)'], 
        color='blue', 
        alpha=0.6, 
        label='Observations'
        )
        sns.lineplot(
        x=df["Nombre d'articles"], 
        y=df['Predicted'], 
        color='red', 
        label='Prédictions'
        )


    plt.title(title)
    plt.xlabel("Nombre d'articles")
    plt.ylabel("Nombre d'occurences pour 10 000 habitants")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "results": results,
        "df_enriched": df,
        "effects_fixed": effects_fixed,
    }

def Wald_test(FEM_results, Pooled_results):
    # Statistique du Wald test : différence entre les log-vraisemblances
    wald_stat = 2 * (FEM_results["results"].loglik - Pooled_results["results"].loglik)
    n_entities = FEM_results["df_enriched"].index.get_level_values(0).nunique() 
    degrees_of_freedom = n_entities- 1 
    p_value = chi2.sf(wald_stat, degrees_of_freedom)

    # 4. Afficher les résultats
    print("Wald Test Statistic:", wald_stat)
    print("Degrees of Freedom:", degrees_of_freedom)
    print("P-value:", p_value)

    # Interprétation
    if p_value < 0.05:
        return("Le modèle FEM est statistiquement significatif. Utilisez FEM.")
    else:
        return("Pas d'évidence pour préférer FEM. Utilisez le modèle pooled OLS.")
        
def regression_lags(df, title, entity_effects=False):
    """
    Effectue des régressions avec des lags de 1 à 5 sur les articles,
    et retourne les p-values des coefficients de 'Articles_lag x' et le R² de chaque régression.

    Args:
        df (DataFrame): Le DataFrame contenant les données.
        title (str): Ajuster le titre du graphique.
        entity_effects (bool): Inclure ou non les effets fixes des entités.

    Returns:
        list: Liste de tuples (p-value de 'Articles_lag x', R²) pour x allant de 1 à 5.
    """
    results_list = []

    for lag in range(1, 6):
        # Appel de la fonction existante pour chaque lag
        res = regression(df.copy(), title + f" (Lag {lag})", entity_effects, lag=lag)
        clear_output(wait=True)

        r_squared = float(res['results'].rsquared)  # Conversion en float
        
        # Extraction des coefficients sous forme de DataFrame
        summary_df = res['results'].params.to_frame(name='coef')
        summary_df['pvalue'] = res['results'].pvalues
        summary_df['stderr'] = res['results'].std_errors
        
        # Filtrer pour obtenir 'Articles_lag x'
        article_lag_row = summary_df.filter(like=f"Articles_lag{lag}", axis=0)
        
        if not article_lag_row.empty:
            p_value = float(article_lag_row['pvalue'].values[0])  # Conversion directe
            results_list.append((p_value, r_squared))

    return results_list
