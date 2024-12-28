# Python-pour-la-data-science-2A

*Projet de William Olivier, Anh Linh Piketty et Arthur Bidel effectué dans le cadre du cours "Python pour la data science" au premier semestre de l'année 2024/2025*

### Contexte et objectif

Ce projet s'inscrit dans le débat récurrent autour de l'insécurité et de l'efficacité de la législation en matière de délinquance en France. Nous cherchons à explorer dans quelle mesure l'évolution de la production législative influe sur les taux de criminalité et de délits. L'objectif est double : dresser un aperçu descriptif des tendances de ces phénomènes sur les 25 dernières années, et vérifier si un lien empirique existe entre législation sécuritaire et baisse de la délinquance.

### Table des matières

* [Récupération des données de Légifrance via une API](#section1)
    * [Installation et importation des modules](#section11)
    * [Requêtes sur l'API](#section12)
    * [Travail sur les fichiers extraits](#section13)
* [Nettoyage des données de Légifrance](#section2)
* [Sauvegarde des tableaux de données finalisées](#section3)

### Définitions

### Récupération des données

Notre travail s'appuie sur les sources suivantes :

Wikipédia – Informations sur la superficie des départements.
Légifrance – Données sur l'activité législative.
Ministère de l'Intérieur – Statistiques relatives à la délinquance et contours géographiques des départements.
INSEE – Données démographiques et indicateurs liés aux taux de pauvreté.

Les données provenant de Légifrance ont été extraites à l'aide de l'API Piste, mise à disposition par les services publics.
<span style="color:red;">**+ ajouter des infos liées aux scraping de wikipedia ?**</span>
### Présentation du dépôt

Notre production s'articule autour de deux déclinaisons du fichier. La première version **[main.ipynb](./main.ipynb)** présente le code uniquement accompagné de commentaires, sans exécution préalable, tandis que la seconde version **[main_executed.ipynb](./main_executed.ipynb)** propose un code déjà exécuté, ce qui permet d'afficher les résultats même en cas de problème d'accès aux bases de données.

La version exécutée fait office de livrable final.

Le dossier **[data/](./data/)** héberge une copie locale d'une partie des données extraites, notamment pour éviter d'avoir à exécuter les cellules en lien avec l'API Légifrance.

Dans le script **[visualisation/](./visualisation/)**, plusieurs fonctions et structures (charte graphique, dictionnaire etc) sont implémentées. Ce choix de ne pas les inclure directement dans les notebooks vise à améliorer la clarté du code.

Enfin, le fichier **[requirements/](./requirements/)** permet à pip d'installer toutes les bibliothèques nécessaires pour préparer l'environnement au projet.