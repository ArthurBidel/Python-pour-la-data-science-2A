# Etude des liens entre production législative et délinquance en France

*Projet de William Olivier, Anh Linh Piketty et Arthur Bidel effectué dans le cadre du cours "Python pour la data science" au premier semestre de l'année 2024/2025*

## Contexte et objectif

Ce projet s'inscrit dans le débat récurrent autour de l'insécurité et de l'efficacité de la législation en matière de délinquance en France. Nous cherchons à explorer dans quelle mesure l'évolution de la production législative influe sur les taux de criminalité et de délits. L'objectif est double : dresser un aperçu descriptif des tendances de ces phénomènes sur les 25 dernières années, et vérifier si un lien empirique existe entre législation sécuritaire et baisse de la délinquance.

## Table des matières

* [Définitions](#section1)
* [Récupération des données](#section2)
* [Présentation du dépôt](#section3)

## Définitions <a class="anchor" id="section1"></a>

La **criminalité** correspond à l'ensemble des actes délictueux ou criminels, incluant les crimes (meurtres, agressions, etc.) et les délits (vols, fraudes, etc.). La criminalité est généralement mesurée par le nombre d'infractions enregistrées sur une période donnée.

La **délinquance** constitue tout comportement illégal ou antisocial, souvent plus large que la criminalité, et inclut des actes qui ne sont pas nécessairement considérés comme des crimes graves, mais qui enfreignent néanmoins la loi. Elle englobe des actes tels que les vols, les agressions, ou des infractions mineures.

La **production législative** désigne l'ensemble des activités liées à la création, à l'adoption, la modification et à la promulgation de lois et de règlements par les instances législatives d'un pays. Dans notre analyse, la production législative réfère aux lois, mais aussi aux autres actes réglementaires qui détaillent et précisent l'application de celles-là comme les ordonnances, les décrets et les arrêtés. Cette activité législative reflète donc l'évolution des priorités sociales, économiques et politiques d'un gouvernement.

## Récupération des données <a class="anchor" id="section2"></a>

Notre travail s'appuie sur les sources suivantes :

* Wikipédia – Informations sur la superficie des départements.
* Légifrance – Données sur l'activité législative.
* Ministère de l'Intérieur – Statistiques relatives à la délinquance et contours géographiques des départements.
* INSEE – Données démographiques et indicateurs liés aux taux de pauvreté.

Les données provenant de Légifrance ont été extraites à l'aide de l'API Piste, mise à disposition par les services publics.
<span style="color:red;">**+ ajouter des infos liées aux scraping de wikipedia ?**</span>

## Présentation du dépôt <a class="anchor" id="section3"></a>

Notre production s'articule autour de deux déclinaisons du fichier. La première version `main.ipynb` présente le code uniquement accompagné de commentaires, sans exécution préalable, tandis que la seconde version `main_executed.ipynb` propose un code déjà exécuté, ce qui permet d'afficher les résultats même en cas de problème d'accès aux bases de données.

La version exécutée fait office de livrable final.

Les deux notebooks `database_délinquance.ipynb` et `database_légifrance.ipynb` présentent, respectivement, la récupération et le nettoyage des données relatives à la délinquance et à la production législative. Ils se concluent tous deux par le téléchargement des bases de données requises pour le bon fonctionnement du fichier principal.

Le dossier 'data' héberge une copie locale d'une partie des données extraites, notamment pour éviter d'avoir à exécuter les cellules en lien avec l'API Légifrance.

Dans le script `visualisation.py`, plusieurs fonctions et structures (charte graphique, dictionnaire etc) sont implémentées. Ce choix de ne pas les inclure directement dans les notebooks vise à améliorer la clarté du code.

Enfin, le fichier `requirements.txt` permet à pip d'installer toutes les bibliothèques nécessaires pour préparer l'environnement au projet.