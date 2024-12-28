# Législation sécuritaire et criminalité en France

*Projet de William Olivier, Anh Linh Piketty et Arthur Bidel effectué dans le cadre du cours "Python pour la data science" au premier semestre de l'année 2024/2025*

## Contexte et objectif

Ce projet s'inscrit dans le débat récurrent autour de l'insécurité et de l'efficacité de la législation en matière de délinquance en France. Nous cherchons à explorer dans quelle mesure l'évolution de la production législative influe sur les taux de criminalité et de délits. L'objectif est double : dresser un aperçu descriptif des tendances de ces phénomènes sur les 25 dernières années, et vérifier si un lien empirique existe entre législation sécuritaire et baisse de la délinquance.

## Sommaire

* [Définitions](#section1)
* [Récupération des données](#section2)
* [Présentation du dépôt](#section3)

## Définitions <a class="anchor" id="section1"></a>

La **criminalité** telle qu'entendu dans ce sujet recouvre une partie des actes délictueux et criminels. On a construit huits indicateurs selon la méthodologie actuelle du SSMSI (le Service Statistique Ministériel de la Sécurité Intérieure). Ils recouvrent essentiellement les violences interpersonnelles et certains types d'atteintes au bien. Par exemple la fraude fiscale ou les actes de hautes trahisons ne sont pas pris en compte comme mesure de "l'insécurité" ou du phénomène délinquant. Selon le type d'infraction les unités de mesures peuvent varier. Pour les violences sexuelles ou les coups et blessures on comptera plutôt le nombre de victimes tandis que pour les infractions liées aux stupéfiants ont compte plutôt le nombre de faits constatés par les force de l'ordre. Plus de détails seront donnés dans le notebook consacré.

La **législation sécuritaire** désigne ici un ensemble de textes de lois filtrés selon des mots-clés correspondant à notre construction des indicateurs de criminalité. Nous avons utilisé l'API de Légifrance pour filtrer sur le contenu des textes de lois et les récupérer dans un DataFrame. Nous avons utilisé le fond LODA qui parcours et renvoit des lois, ordonnances, décrets, décisions et arrêtés. 

## Récupération des données <a class="anchor" id="section2"></a>

Notre travail s'appuie sur les sources suivantes :

* Wikipédia – Informations sur la superficie des départements.
* Légifrance – Données sur l'activité législative.
* Ministère de l'Intérieur – Statistiques relatives à la délinquance et contours géographiques des départements.
* INSEE – Données démographiques et indicateurs liés aux taux de pauvreté.

Les données provenant de Légifrance ont été extraites à l'aide de l'API Piste, mise à disposition par les services publics.
<span style="color:red;">**+ ajouter des infos liées aux scraping de wikipedia ?**</span>

## Présentation du dépôt <a class="anchor" id="section3"></a>

Notre travail s'articule principalement autour de deux déclinaisons du fichier. La première version `main.ipynb` présente le code uniquement accompagné de commentaires, sans exécution préalable, tandis que la seconde version `main_executed.ipynb` propose un code déjà exécuté, ce qui permet d'afficher les résultats même en cas de problème d'accès aux bases de données.

La version exécutée fait office de livrable final.

Les deux notebooks `database_délinquance.ipynb` et `database_légifrance.ipynb` présentent, respectivement, la récupération et le nettoyage des données relatives à la délinquance et à la production législative. Ils se concluent tous deux par l'exportation dans S3 des bases de données requises pour le bon fonctionnement du fichier principal.

Un dernier notebook `annexes.ipynb` contient différents essais plus ou moins infructueux d'utilisation de l'API Légifrance. On a finit par choisir de se restreindre au fond LODA mais on a d'abord tenté de faire des recherches plus élargies ce qui a donné lieux à plusieurs démarches que nous trouvions dommage d'effacer.

Le dossier 'data' héberge une copie locale d'une partie des données extraites, notamment pour pallier à d'éventuel problèmes de récupération des données que nous avions eu pendant la réalisation du projet.

Dans le script `visualisation.py`, plusieurs fonctions et structures (charte graphique, dictionnaire etc) sont implémentées. Ce choix de ne pas les inclure directement dans les notebooks vise à améliorer la clarté du code.

Enfin, le fichier `requirements.txt` permet à pip d'installer toutes les bibliothèques nécessaires pour préparer l'environnement au projet.