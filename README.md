# stage_astro


## Contexte

L'univers est en constante expansion et il est nécessaire, pour le comprendre, de modéliser la toile cosmique ainsi que mesurer l'éloignement des différents corps célestes. Cela est possible en mesurant le *redshift* (décalage vers le rouge) qui quantifie l'éloignement des objets (similaire à l'effet Dopler). Le redshift peut être déduit à partir des magnitudes obersvées dans différentes longueurs d'ondes et de nombreux modèles de Machine Learning ont été appliqués à cette tâche, à partir des magnitudes extraites (D'Isanto et Polsterer 2017) ou bien directement de images (Pasquet et al. 2019). Cependant ces modèles sont très dépendants des données et un énorme biais dans la distribution des redshifts connus vient perturber l'apprentissage. En effet, il est très coûteux d'estimer des redshifts élevés de manière fiable (par spectrométrie) et l'estimation par photométrie (template-fitting et Machine Learning) est une alternative moins fiable mais moins coûteuse. Ce *shift* de distribution qui donne davantage de données labellisées à basses valeurs engendre une dégénérescence des modèle de Machine Learning à mesure que la valeur réponse augmente, l'objectif de ce stage était de proposer une solution au manque de robustesse des modèles classiques grâce à l'apprentissage non-supervisé. Les données non-labellisés n'étant pas biaisé dans leur distribution, il est en théorie possible d'apprendre des représentations robustes et transférables à la tâche down-stream de prédiction du redshift. 


## Les données

Les images utilisées sont issues d'un découpage opéré par les astrophysiciens du Laboratoire d'Astrophysique de Marseille, elles sont de taille 64 pixels par 64 dans 6 longueurs d'ondes observées (U, G, R, I, Z, Y). Elles sont tirées du survey COSMOS qui peut être divisé en COSMOS **DEEP** (D), l'extérieur du champs, et COSMOS **ULTRA-DEEP** (UD), l'intérieur du champ. Les données labellisées sont de l'ordre de 150000 pour COSMOS UD et 20000 pour COSMOS D, parmi les labels certains sont obtenus par template fitting sur du 30 bandes et bien que ce ne soit pas aussi fiable qu'une estimation par spectrométrie ils sont considérés comme de vrais labels. A cela s'ajoute 200000 images non-labellisées de UD et 700000 non-labellisées de D. Une normalisation propre aux images astrophysiques est appliquée avec la formule \(x' = sign(x) * (sqrt(abs(images)+1)-1) \).

## Méthode 

### Choix du modèle 

L'apprentissage non-supervisé, ou plutôt *auto-supervisé* (Self Supervised Learning) peut être décomposé en deux grandes familles que sont les méthodes contrastives dont l'objectif est de maximiser la similarité entre 2 augmentations d'une même image et l'apprentissage de représentation par tâches "pré-textes" (colorier une image, ré-organiser les patch...). La famille choisie ici est celle des modèles contrastifs, parmi eux figurent BYOL (Grill et al. 2020), SimCLR (Chen et al. 2020), BarlowTwins (Zbontar et al. 2021), VICReg (Bardes et al. 2021), DINOv2 (Oquab et al. 2023). Dans un premier temps nous avons opté pour BYOL qui a la particularité de ne pas avoir besoin d'exemples négatifs cependant le modèle s'est confronté au phénomène connu de *collapse* de par cette absence de contrainte sur les paires. Après avoir testé les trois framework de SimCLR, VICReg, et BarlowTwins qui permirent tous d'apprendre des représentation comportant l'information du redshift, c'est SimCLR qui a été choisi pour continuer. Le framework de DINOv2 aurait également pu être utilisé mais il nécessite énormément de ressources de calcul et n'est pas très modulable sur les augmentations de données. En effet, toutes ne s'appliquent pas à l'astrophysique étant donné que l'on doit garder les valeurs de canaux relativement intact pour ne pas modifier l'information du redshift ce qui n'est pas vraiment permis dans DINO avec les crops et masquage de patch, deplus l'utilisation de Dropout n'est pas vraiment adapté à la régression, le problème majeur reste la consommation de ressources.

### Procédure 

La première étape consistait donc à développer les frameworks contrastif et assurer l'apprentissage implicite du redshift, pour ce faire la visualisation des t-SNE s'est révélée très efficace. Donner les features à un classifier (XGBoost, MLP) est également possible, pour "pousser le vice", nous avons confirmé la qualité des features avec un KNN sur les coordonnées de la t-SNE.
Une fois le modèle pré-entrainé, nous avons posé 3 scénarios de finetuning : 5k, 10k et 20k données labellisées provenant de UD. Et 2 cas d'inférence : le reste des données UD, les données D.
L'objectif était de constater l'évolution des métriques astrophysiques (biais, déviance MAD et fraction d'outliers) sur différentes plages de redshift et conclure sur la robsutesse de ce type de modèle.

## Résultats

### Pré entrainement des modèles

Comme énoncé plus haut, l'apprentissage des représentations avec BYOL s'est soldé par un échec à cause du *collapse* des features qui tendaient vers un unique vecteur composé de très grandes valeurs. Cela vient en partie du fait que l'on utilise un réseau convolutif dépourvu de BatchNormalization car elle donne en pratique de moins bons résultats sur la prédiction du redshift (la variance ramenée à 1 est un régulariseur implicite des sorties d'une couche et aurait donc limité ce phénomène). L'avantage de SimCLR sur BYOL est l'utilisation de paires négatives qui forcent le modèle à apprendre des représentations discriminantes vis à vis des autres images du batch. Je suppose que c'est en lien avec cette volonté de "repousser" les vecteurs de paires négatives pour minimiser la loss de contraste mais le modèle tend tout de même à augmenter la dynamique de ses sorties (bien que la cosine similarité s'effectue avec normalisation des projections). L'ajout d'une Batch Normalization en fin d'extracteur résout en partie ce problème, en apparence en tout cas car la dynamique pré-BN continue d'augmenter, une solution alternative à cela a été d'ajouté une régularisation sur l'activité en sortie de la tête de projection et force ainsi le réseau à propager de petites valeurs tout le long (une régularisation L1-L2 a été utilisée).

<table>
  <tr>
    <td style="text-align: center;">
      <h4>SimCLR</h4>
      <img src="./raw_files/img/tsne_simCLR.png" alt="SimCLR" width="300"/>
    </td>
    <td style="text-align: center;">
      <h4>BarlowTwins</h4>
      <img src="./raw_files/img/tsne_barlow.png" alt="BarlowTwins" width="300"/>
    </td>
    <td style="text-align: center;">
      <h4>VICReg</h4>
      <img src="./raw_files/img/tsne_vicreg.png" alt="VICReg" width="300"/>
    </td>
  </tr>
</table>


Ce sont des résultats satisfaisant car on retrouve les hauts redshifts regroupés entre eux, idem pour les bas redshifts. Une solution pour forcer davantage l'apprentissage du redshift est l'utilisation d'une tête de régression branchée sur l'extracteur de caractéristiques, les couleurs sont obtenues comme la différence entre 2 canaux de l'images. Il y a donc 5 couleurs par image et on sait qu'il existe une relation entre la couleur et le redshift comme le montre Masters et al. (2015). 

![SimCLR avec prédiction auxiliaire de la couleur](./raw_files/img/tsne_simCLR_color.png)

La t-SNE comportant une partie aléatoire il n'est pas vraiment possible de tirer une conclusion sur l'utilité de la tête de régression mais certaines autres expériences ont montré un intérêt à l'ajouter, rien en tout cas ne montre d'intérêt à ne pas la mettre.



### Finetuning
A l'issue de ces étapes, on peut considérer l'espace latent appris par Self-Supervised comme de bonne qualité, comme prouvée par KNN sur la projection t-SNE (densité des y prédit en fonction des y réels proche de la droite y=x). Cependant un phénomène de *negative transfer* a été observé dans certaines configurations, en effet les représentations apprises pour la tâche non-supervisée ne sont pas forcément **directement** transférable à la prédiction du reshift. Encore une fois selon mon interprétation, la perte contrastive avec une température proche de 1 rend l'apprentissage plus dur car on obtient des logits entre [-1, 1] et qu'on applique ensuite un softmax pour trouver la paire positive, cela implique une presque impossibilité du modèle à discriminer la paire et donc une perte élevé, plus de mises à jour dans le réseau et donc moins de stabilité. Dans le premier cas la température était de 0.7 avec une loss d'environ 10, dans le cas d'une température de 0.1 comme dans les travaux d'Hayat et al. (2021) la loss s'optimise bien mieux aux environs de 0.6. Mais ce n'est pas tout, les features apprises par SSL ne garantissent aucune cohérence au sein d'une même classe car ce concept n'existe pas lors de l'apprentissage ainsi deux images appartenant à la même classe peuvent être opposées dans l'espace latent du moment qu'ils sont bien discriminés individuellement. C'est un problème que tente de résoudre Zhang et al. (2021) avec leur méthode core-tune qui permet de réorganiser l'espace latent pour prendre en compte les classes et en tirant profit de l'apprentissage non-supervisé c'est à dire en maximisant l'entropie sur l'hypersphère mais en minimisant l'entropie de chaque classe, produisant des classes compactes et discriminantes. Ces travaux visent à résoudre les limites de l'apprentissage Self-Supervised concernant les frontières de décision Sharp ainsi que la robustesse adversariale qui sont précisément les problèmes rencontrés ici, cependant cela a été pensé pour de la classification et s'adapte difficilement à de la régression.

En parallèle de ces constats sur le negative transfer, d'autres tests pour reproduire les résultats d'Hayat et al. ont été menés, résultants en beaucoup de modèles finetunés et d'inférence. Il est possible de les regrouper ainsi :

| **Modèle**         | **Régularisation**          | **Température NTXent** |
|-------------------|-----------------------------|------------------------|
| **CNN basiques**   | Avec activité               | 0.7 (SimCLR original)  |
|                   | Sans activité               | 0.7 (SimCLR original)  |
|                   | Avec activité               | 0.1 (Hayat et al.)     |
|                   | Sans activité               | 0.1 (Hayat et al.)     |
| **ResNet50**       | Sans régularisation         | 0.7 (SimCLR original)  |
|                   | Sans régularisation         | 0.1 (Hayat et al.)     |
| **ViT**            | Avec activité               | 0.7                    |
|                   | Sans activité               | 0.7                    |
|                   | Avec activité               | 0.1                    |
|                   | Sans activité               | 0.1                    |
