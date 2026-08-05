# Design : plugin Trino `trino-vector` (v1, KNN exact)

Date : 2026-08-05

## 1. Objectif

Fournir un plugin Trino de fonctions vectorielles permettant la recherche des k plus proches
voisins **exacte** en SQL : métriques de distance, utilitaires de normalisation, et une
agrégation top-k par groupe.

La recherche approximative (ANN) est un objectif ultérieur explicite. Le design de la v1 est
contraint par ce futur : le noyau de calcul et l'énumération des métriques sont isolés pour
être réutilisés tels quels par un index, sans réécriture.

### Non-objectifs de la v1

Sont volontairement exclus, et ne doivent pas être anticipés dans le code :

- index approximatif (HNSW, IVF)
- type vectoriel custom `vector(n)`
- connecteur ou table function (PTF)
- persistance d'index

### Ce que le plugin apporte par rapport à Trino nu

Trino 483 sait déjà faire un top-k exact global : `ORDER BY cosine_similarity(...) LIMIT k`,
avec un TopN distribué efficace. Le plugin ajoute ce qui manque :

- les métriques absentes du moteur : L2, L2 au carré, produit scalaire, L1, distance cosinus
- les utilitaires `l2_norm` et `normalize_vector` (sur vecteurs normalisés, cosinus et produit
  scalaire coïncident, ce qui divise le coût du calcul)
- un top-k **par groupe** à mémoire bornée, qu'aucune construction SQL ne fournit sans trier
  l'intégralité de chaque groupe

## 2. Décisions structurantes

| Décision | Choix | Raison |
| --- | --- | --- |
| Cas d'usage | recherche par similarité | et non classification KNN |
| Surface SQL | fonctions seulement | pas de connecteur : aucune configuration, portée globale |
| Représentation | `array(double)` et `array(real)` natifs | zéro friction avec les données existantes |
| Type custom | aucun | un `vector(n)` ne se justifie qu'avec un index qui stocke |
| Cible | Trino 483, Java 25 | release publiée, donc partageable |
| Nommage | `dev.jaaj.trino.vector` | groupId publiable, pas de split package avec l'amont |

## 3. Build et packaging

Dépôt standalone (le dépôt git s'appelle `trino-knn`, l'artefact `trino-vector` : divergence
assumée).

- parent `io.airlift:airbase:395`, celui de Trino 483
- `groupId` `dev.jaaj.trino`, `artifactId` `trino-vector`
- `<packaging>trino-plugin</packaging>` via `io.trino:trino-maven-plugin:20` avec
  `<extensions>true</extensions>`, qui génère `META-INF/services/io.trino.spi.Plugin` et
  l'arborescence de déploiement
- Java 25, en-têtes de licence Apache, checkstyle et modernizer hérités d'airbase
- Maven Wrapper (`mvnw`) committé dans le dépôt

Dépendances, en `provided` (fournies par le serveur, jamais embarquées) :
`io.trino:trino-spi`, `io.airlift:slice`.
En `compile` : `com.google.guava:guava`.
En `test` : `io.trino:trino-testing`, `io.trino:trino-main`, `io.trino:trino-main` (test-jar),
`org.junit.jupiter:junit-jupiter-api`, `org.assertj:assertj-core`.

Le parent est `airbase` et non `trino-root` : la `dependencyManagement` de Trino n'est donc pas
héritée et **toutes les versions doivent être épinglées explicitement** dans le POM.

## 4. Architecture

```
dev.jaaj.trino.vector
├─ VectorPlugin              implements Plugin, ne remplit que getFunctions()
├─ Metric                    enum L2, L2_SQUARED, COSINE, DOT_PRODUCT, L1
├─ VectorMath                noyau de calcul, package-private, opère sur Block
├─ VectorDistanceFunctions   les @ScalarFunction de distance
├─ VectorFunctions           l2_norm, normalize_vector
├─ VectorErrorCode           ErrorCodeSupplier
└─ agg
   ├─ KnnAggregation         @AggregationFunction("knn_agg")
   ├─ KnnState               @AccumulatorStateMetadata
   ├─ KnnStateFactory
   └─ KnnStateSerializer
```

`VectorMath` est le **seul** endroit du plugin contenant une boucle de distance. Contrainte de
design volontaire : un futur index ANN appellera ce même noyau, et une éventuelle vectorisation
(Vector API) se fera à un seul endroit.

`Metric` est la frontière entre « quelle métrique » et « comment on l'applique ». L'agrégation
ne connaît aucune formule ; elle demande à `Metric` une valeur et un sens de comparaison.

## 5. Noyau de calcul

Les arguments `array(...)` arrivent sous forme de `Block`, pas de `double[]`. Chaque distance se
calcule en **une seule passe** sur les deux blocs, sans allocation ni tableau intermédiaire.

### Règles de bord

| Cas | Comportement |
| --- | --- |
| dimensions différentes | `TrinoException(INVALID_FUNCTION_ARGUMENT)`, message contenant les deux tailles |
| élément `NULL` dans un tableau | `TrinoException(INVALID_FUNCTION_ARGUMENT)` |
| argument tableau `NULL` | résultat `NULL` (comportement standard Trino) |
| deux tableaux vides | `l2_distance`, `l2_squared_distance`, `l1_distance`, `dot_product` valent `0` |
| deux tableaux vides | `cosine_distance` vaut `NULL` |
| norme nulle (`cosine_distance`, `normalize_vector`) | `NULL`, jamais `NaN` |

Un `NaN` renvoyé silencieusement se propagerait dans un `ORDER BY` avec un ordre de tri
arbitraire. `NULL` est le seul résultat qui rend l'indéfini visible.

## 6. Fonctions scalaires

Toutes déclinées en `array(double)` et `array(real)` par surcharge `@ScalarFunction`. Pas de
cast implicite de `real` vers `double` : il doublerait l'empreinte mémoire sur des tables
d'embeddings stockées en float32.

| Signature | Résultat |
| --- | --- |
| `l2_distance(x, y) -> double` | distance euclidienne |
| `l2_squared_distance(x, y) -> double` | son carré, sans `sqrt` |
| `cosine_distance(x, y) -> double` | `1 - cos(x, y)` |
| `dot_product(x, y) -> double` | produit scalaire, valeur brute |
| `l1_distance(x, y) -> double` | distance de Manhattan |
| `l2_norm(x) -> double` | norme euclidienne |
| `normalize_vector(x)` | vecteur de norme 1, du même type que l'entrée |

`normalize_vector` préserve le type d'entrée : `array(double)` en entrée donne `array(double)`,
`array(real)` donne `array(real)`. Promouvoir en `double` doublerait l'empreinte mémoire d'une
colonne float32 sans gain de précision réel.

`cosine_distance` et non `cosine_similarity` : Trino fournit déjà la similarité, et une
*distance* se trie en `ASC` comme toutes les autres métriques, ce qui évite l'erreur classique
du `DESC` oublié.

## 7. Agrégation `knn_agg`

```sql
knn_agg(key, vector, query_vector, k, metric) -> array(row(key, distance))
```

- `key` est générique via `@TypeParameter("K")` : évite un `CAST` à l'appelant
- `k` est un `bigint`, `metric` un `varchar`
- `vector` et `query_vector` sont tous deux `array(double)` **ou** tous deux `array(real)` :
  deux surcharges, comme pour les fonctions scalaires. Mélanger les deux types dans un même
  appel est une erreur de résolution de fonction, gérée par le moteur, pas par le plugin.
- `query_vector`, `k` et `metric` sont lus à la première ligne du groupe et mémorisés dans
  l'état. Trino n'a pas de notion d'argument constant pour une agrégation : c'est la convention
  déjà employée en amont (`LearnLibSvmClassifierAggregation`).

### Métriques acceptées

`'l2'`, `'l2_squared'`, `'cosine'`, `'dot_product'`, `'l1'`. Valeur inconnue :
`TrinoException(INVALID_FUNCTION_ARGUMENT)` listant les valeurs valides.

### Sens de tri

`Metric` porte un drapeau `higherIsCloser`. `dot_product` est une similarité (plus grand = plus
proche) ; les quatre autres sont des distances. La valeur placée dans le résultat est toujours
la **valeur brute** de la métrique, jamais une valeur négée : ce que l'utilisateur lit est ce
que la fonction scalaire homonyme calculerait.

### État

Un tas borné à `k` par groupe. Les clés, de type générique, sont accumulées dans un
`BlockBuilder` avec un tas sur les positions, et le builder est compacté quand il dépasse un
multiple de `k`. C'est le mécanisme employé en amont par `min_by` et `max_by`.

Mémoire en O(k) par groupe, contre O(n) pour un `array_agg` suivi d'un tri. `getEstimatedSize()`
est implémenté : Trino comptabilise alors la mémoire et interrompt proprement la requête au lieu
de tomber en `OutOfMemoryError`.

### Distribution

`@CombineFunction` fusionne deux tas et conserve les `k` meilleurs. Trino exécute toujours une
agrégation en deux passes (partielle par worker, puis finale) : sans `@CombineFunction`
correcte, l'agrégation renvoie des résultats faux dès que les données sont réparties sur
plusieurs splits.

`KnnStateSerializer` sérialise l'état vers `row(bigint, varchar, array(row(K, double)))`,
c'est-à-dire `k`, la métrique, puis le contenu du tas.

`k` et la métrique **doivent** faire partie de l'état sérialisé. L'agrégation finale ne voit
jamais les lignes d'origine : elle ne reçoit que des états partiels désérialisés. Sans ces deux
champs, elle ignorerait à la fois combien de voisins conserver et dans quel sens comparer. Ne
sérialiser que le tas est l'erreur naturelle ici, et elle ne se manifeste qu'en exécution
distribuée, jamais sur un test à une seule partition.

`query_vector` n'est en revanche pas sérialisé : les distances sont déjà calculées au moment de
la fusion, le vecteur requête n'a plus d'utilité.

### Cas de bord

| Cas | Comportement |
| --- | --- |
| `k <= 0` | `TrinoException(INVALID_FUNCTION_ARGUMENT)` |
| `k` > nombre de lignes du groupe | renvoie toutes les lignes du groupe |
| groupe vide | `NULL` |
| `vector` `NULL` sur une ligne | ligne ignorée |
| `key` `NULL` | ligne conservée, clé `NULL` dans le résultat |
| distances égales | ordre entre ex aequo non garanti, documenté comme tel |

Sortie triée du plus proche au plus lointain.

## 8. Tests

Développement en TDD : chaque comportement listé dans ce document a un test écrit avant son
implémentation.

**Niveau 1, unitaire pur sur `VectorMath`.** Valeurs de référence calculées à la main, et
propriétés mathématiques : symétrie, identité des indiscernables, inégalité triangulaire pour L1
et L2. Toutes les règles de bord de la section 5.

**Niveau 2, SQL de bout en bout.** `StandaloneQueryRunner` + `AbstractTestQueryFramework`, sur le
modèle de `TestMLQueries` en amont : un vrai moteur Trino en mémoire, le plugin installé, du SQL
réel sur des `VALUES`. Couvre la résolution des surcharges `double` / `real`, les messages
d'erreur, et tous les cas de bord de la section 7.

**Niveau 3, test différentiel.** Sur données générées, `knn_agg(id, v, q, k, 'l2')` doit
produire exactement le même résultat que `ORDER BY l2_distance(v, q) LIMIT k`. C'est le test qui
attrape les erreurs de tas et de fusion, invisibles sur trois lignes écrites à la main.

Un test force explicitement plusieurs splits afin que le chemin `@CombineFunction` soit
réellement exercé : sur de petits jeux de données, Trino n'agrège que sur un seul worker et ce
chemin n'est jamais emprunté.

## 9. Contrainte d'environnement

Trino 483 impose **Java 25**. La machine de développement ne dispose que des JDK 8, 16 et 17, et
Maven n'est pas installé.

- Maven : résolu par le Maven Wrapper committé dans le dépôt, qui se télécharge seul
- JDK 25 : doit être installé (Temurin 25). Tant qu'il ne l'est pas, le code ne peut être ni
  compilé ni testé, et aucune affirmation de bon fonctionnement ne peut être faite

## 10. Extensions prévues (hors périmètre)

Points d'accroche déjà en place pour la suite :

- `Metric` et `VectorMath` sont réutilisables tels quels par un index ANN
- l'ANN nécessitera un connecteur (les PTF sont déclarées par un connecteur, pas par un plugin
  de fonctions nu) : ce sera un projet distinct, avec son propre cycle spec / plan
- un type `vector(n)` en float32 ne se justifiera qu'à ce moment-là, quand la compacité de
  stockage et la vérification de dimension à l'analyse auront une contrepartie réelle
