# Projet : clustering sous contrainte d'ordre

**Master MLSD, 2026/2027**
**Auteurs :** [Nom A], [Nom B]
**Rendu :** 27/11/2026 (FI) ou 04/12/2026 (FA)

> Les passages `[À COMPLÉTER]` sont à remplacer par vos résultats. Les notes de rédaction en fin de document sont à supprimer avant le rendu.

---

## 1. Introduction

De nombreuses données sont naturellement ordonnées dans le temps ou dans l'espace : signaux audio, mesures de capteurs, séries financières. Dans ces cas, on cherche souvent à les découper en segments homogènes, par exemple pour détecter des ruptures ou attribuer des intervalles de temps à des locuteurs différents. Un algorithme de clustering classique comme le k-means ne tient pas compte de l'ordre des observations : une même classe peut regrouper des points très éloignés dans la séquence, ce qui ne correspond pas à une segmentation.

L'objectif de ce projet est d'implémenter la méthode de programmation dynamique de R. A. Fisher, qui partitionne une séquence en $K$ intervalles consécutifs en minimisant l'inertie intra-classe, et de la comparer à des méthodes de clustering classiques sur deux jeux de données : une séquence simulée bidimensionnelle (`sequencesimu`) et des données réelles d'aiguillage (`aiguillage`, 140 observations dans $\mathbb{R}^{552}$, avec une vraie partition connue).

Le compte-rendu est organisé comme suit. La section 2 présente l'algorithme de Fisher et son implémentation. La section 3 traite les données simulées (choix de $K$ par la méthode du coude, comparaison de partitions). La section 4 traite les données réelles. La section 5 discute les résultats et la section 6 conclut.

---

## 2. Méthode : programmation dynamique de Fisher

### 2.1 Problème

On dispose d'une séquence ordonnée de $n$ observations $(x_1, \dots, x_n)$, avec $x_i \in \mathbb{R}^p$. On cherche à la découper en $K$ segments consécutifs $[1 ; t_1 - 1], [t_1 ; t_2 - 1], \dots, [t_{K-1} ; n]$, en choisissant les $K-1$ instants de rupture $(t_1, \dots, t_{K-1})$. Contrairement à un clustering classique, la contrainte d'ordre impose que chaque classe soit un intervalle de temps.

Le critère à minimiser est l'inertie intra-classe totale :

$$C(t_1, \dots, t_{K-1}) = D(1, t_1 - 1) + \sum_{k=2}^{K-1} D(t_{k-1}, t_k - 1) + D(t_{K-1}, n)$$

où $D(a, b)$ est l'inertie du segment $[a ; b]$, c'est-à-dire la somme des carrés des écarts à la moyenne du segment :

$$D(a, b) = \sum_{i=a}^{b} \sum_{j=1}^{p} \left(x_i^j - \mu_{ab}^j\right)^2, \qquad \mu_{ab}^j = \frac{1}{b - a + 1} \sum_{i=a}^{b} x_i^j .$$

### 2.2 Principe de l'algorithme

Le nombre de partitions possibles en $K$ segments est $\binom{n-1}{K-1}$, ce qui exclut une recherche exhaustive dès que $n$ est grand. L'algorithme de Fisher exploite le principe d'optimalité de Bellman : toute sous-partition d'une partition optimale est elle-même optimale. On définit $M[i, k]$ comme le coût minimal pour découper les $i$ premières observations en $k$ segments. Il vérifie la récurrence :

$$M[i, 1] = D(1, i), \qquad M[i, k] = \min_{t = k, \dots, i} \bigl\{ M[t-1, k-1] + D(t, i) \bigr\} \quad (k \geq 2).$$

Ici $t$ est le début du dernier segment : il coûte $D(t, i)$, et les $t-1$ premières observations sont découpées de façon optimale en $k-1$ segments. La borne $t \geq k$ garantit que chaque segment est non vide. Le minimum global est $C = M[n, K]$. On stocke aussi $A[i, k]$, la valeur de $t$ qui réalise le minimum.

### 2.3 Déroulement

L'algorithme se déroule en quatre étapes :

1. calcul de la matrice triangulaire supérieure $D$ des coûts de tous les segments $[a ; b]$ ;
2. remplissage de $M$ et $A$, pour $k = 1, \dots, K$ ;
3. retour en arrière : $A[n, K]$ donne la dernière rupture $t_{K-1}$, puis $A[t_{K-1} - 1, K-1]$ donne $t_{K-2}$, et ainsi de suite jusqu'à $t_1$ ;
4. construction des labels à partir des points de rupture.

L'ensemble des partitions admissibles étant parcouru implicitement, la solution est un **optimum global** du critère $C$, contrairement au k-means qui converge vers un optimum local.

### 2.4 Complexité

Le remplissage de $M$ et $A$ demande $O(K n^2)$ opérations ($Kn$ cases, chacune nécessitant un minimum sur au plus $n$ candidats). Le calcul de $D$ coûte $O(n^3 p)$ avec notre implémentation directe, car chaque diamètre est recalculé, et pourrait être ramené à $O(n^2 p)$ grâce à des sommes cumulées. La mémoire nécessaire est de $O(n^2)$ pour $D$ et $O(nK)$ pour $M$ et $A$. Pour nos données ($n \leq 160$, $p \leq 552$), le temps de calcul reste de l'ordre de la seconde.

### 2.5 Implémentation et validation

L'algorithme est implémenté en Python (NumPy) dans deux fonctions : `diam(X, a, b)`, qui calcule $D(a, b)$, et `clustfisher(X, K)`, qui renvoie les points de rupture, les labels, le critère $C$ et la matrice $M$. Cette dernière donne, en un seul appel, le critère $C$ pour tout $K \leq K_{\max}$, ce qui permet de tracer la courbe du coude. Les indices suivent la convention du sujet (à partir de 1).

L'implémentation a été validée par :

- le cas $K = 1$, où $C = D(1, n)$ ;
- une comparaison avec une recherche exhaustive sur un petit jeu de données ($n = 10$, $K = 3$) ;
- des données à paliers dont les ruptures sont connues ;
- un calcul de $D$ sur un exemple vérifiable à la main.

### 2.6 Méthodes de comparaison

Nous comparons Fisher à deux méthodes étudiées en cours : **[k-means]** (minimisation de l'inertie intra-classe sans contrainte d'ordre, optimum local, [nombre] initialisations aléatoires) et **[classification ascendante hiérarchique (CAH) avec critère de Ward]**. *[À ADAPTER si vous choisissez un autre algorithme, par exemple un mélange gaussien.]*

Les partitions sont comparées par :

- l'**inertie intra-classe**, qui est le critère minimisé par Fisher (pour comparer les trois méthodes sur la même quantité) ;
- l'**indice de Rand ajusté (ARI)**, qui mesure l'accord entre deux partitions indépendamment de la numérotation des classes ;
- la **position des points de rupture** et, pour les données réelles, la **matrice de confusion** avec la vraie partition.

---

## 3. Application aux données simulées

### 3.1 Données

[À COMPLÉTER : taille de la séquence $n$, dimension $p = 2$, figure de la séquence brute.]

### 3.2 Choix du nombre de segments

Nous calculons le critère $C$ pour $K = 1, \dots, K_{\max}$ ($K_{\max}$ = [valeur]) à partir de la matrice $M$, puis appliquons la méthode du coude.

**[FIGURE 1 : courbe du coude, $C$ en fonction de $K$.]**

[À COMPLÉTER : valeurs de $C$, justification du $K$ retenu (point à partir duquel la décroissance de $C$ devient faible). Valeur retenue : $K$ = [valeur].]

### 3.3 Segmentation par l'algorithme de Fisher

**[FIGURE 2 : séquence colorée par segment, avec les points de rupture en pointillés.]**

[À COMPLÉTER : points de rupture trouvés, effectifs et caractéristiques des segments.]

### 3.4 Comparaison avec k-means et CAH

**[FIGURE 3 : trois partitions côte à côte (Fisher, k-means, CAH).]**

| Méthode | Inertie intra-classe | ARI avec Fisher | Segments contigus ? |
|---|---|---|---|
| Fisher | [ ] | 1 | oui |
| k-means | [ ] | [ ] | [oui / non] |
| CAH (Ward) | [ ] | [ ] | [oui / non] |

[À COMMENTER : les classes des autres méthodes sont-elles des intervalles ? Où se situent les différences (zones de transition, classes mélangées dans le temps) ?]

*Note : sans contrainte d'ordre, k-means et la CAH peuvent obtenir une inertie plus faible que Fisher, car ils ont plus de liberté (la contrainte restreint l'ensemble des partitions admissibles). Une inertie plus basse ne signifie donc pas une meilleure segmentation.*

---

## 4. Application aux données réelles « aiguillage »

### 4.1 Données et protocole

Les données comportent 140 observations dans $\mathbb{R}^{552}$ et une vraie partition en 4 classes (553e colonne). Nous segmentons la séquence en $K = 4$ classes avec Fisher, k-means et la CAH, et comparons chaque partition à la vraie partition.

[À COMPLÉTER : description des données (ce que représente une observation), vraies positions de rupture : [ ].]

### 4.2 Résultats

| Méthode | ARI avec la vraie partition | Erreur de classement | Ruptures trouvées | Temps de calcul |
|---|---|---|---|---|
| Vraie partition | 1 | 0 | [ ] | n.a. |
| Fisher | [ ] | [ ] | [ ] | [ ] s |
| k-means | [ ] | [ ] | [ ] | [ ] s |
| CAH (Ward) | [ ] | [ ] | [ ] | [ ] s |

**[FIGURE 4 : séquence (projection ou première composante) colorée par la vraie partition et par chaque méthode.]**

**[FIGURE 5 ou TABLEAU : matrices de confusion de chaque méthode.]** Attention : les numéros de classes sont arbitraires pour k-means et la CAH, il faut apparier les classes avant de calculer une erreur de classement.

### 4.3 Analyse

[À COMPLÉTER : Fisher retrouve-t-il les vraies ruptures ? À combien d'observations près ? Que font k-means et la CAH (leurs classes sont-elles contiguës) ?]

---

## 5. Discussion

- **Rôle de la contrainte d'ordre.** [À COMPLÉTER : lorsque la vraie structure est composée de segments consécutifs, la contrainte aide, car elle élimine les partitions incohérentes ; dans le cas contraire, elle peut être pénalisante.]
- **Optimalité.** Fisher fournit un optimum global du critère $C$ sous contrainte d'ordre ; k-means ne garantit qu'un optimum local et dépend de l'initialisation.
- **Dimension élevée ($p = 552$).** [À COMPLÉTER : effet sur les distances, sur la qualité de k-means et de la CAH, sur le temps de calcul de $D$.]
- **Choix de $K$.** La méthode du coude est subjective ; [À COMPLÉTER : le coude était-il net sur les données simulées ?]
- **Coût de calcul.** Complexité en $O(Kn^2)$ pour la récurrence, mémoire en $O(n^2)$ : adapté à quelques milliers d'observations, plus coûteux pour de très longues séquences.
- **Limites.** Le critère (inertie intra-classe) suppose des segments homogènes en moyenne ; il détecte mal des changements de variance ou de dynamique avec une moyenne constante. Il n'existe pas de critère automatique de sélection de $K$.

---

## 6. Conclusion

[À COMPLÉTER : rappel de ce qui a été fait (implémentation et validation de l'algorithme de Fisher, deux applications), des principaux résultats chiffrés (ARI, ruptures retrouvées), de l'apport de la contrainte d'ordre, et d'une ou deux perspectives, par exemple des sommes cumulées pour accélérer le calcul de $D$, ou un critère de sélection de $K$ tel que BIC ou un critère pénalisé.]

---

## Annexe : code

Le code Python complet (`fisher.py`, `utils.py`, `autres_algos.py`, `test_fisher.py`, `main.py`) est fourni avec ce rapport. *[Ne pas recopier le code dans le rapport, pour respecter la limite de 10 pages.]*

---

## Notes de rédaction (à supprimer avant le rendu)

**Budget de pages (10 maximum)**

| Section | Pages | Responsable |
|---|---|---|
| 1. Introduction | 0,5 | B |
| 2. Méthode | 2 | A |
| 3. Données simulées | 2,5 | A (rédaction), B (comparaison) |
| 4. Données réelles | 2,5 | B |
| 5. Discussion | 1,5 | A et B |
| 6. Conclusion | 0,5 | A et B |

**Checklist avant rendu**

- [ ] Chaque figure est numérotée, légendée et citée dans le texte.
- [ ] Les chiffres du texte sont identiques à ceux produits par le code.
- [ ] Les deux autres méthodes sont les mêmes pour les deux jeux de données.
- [ ] Le nombre de pages est inférieur ou égal à 10 après export PDF.
- [ ] Tous les `[À COMPLÉTER]` ont été remplacés ou supprimés.
