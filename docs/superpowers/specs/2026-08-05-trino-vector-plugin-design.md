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

### Ce que Trino 483 fournit déjà

`io.trino.operator.scalar.ArrayVectorFunctions` expose nativement, sur `array(double)`
uniquement :

`euclidean_distance`, `dot_product`, `cosine_similarity`, `cosine_distance`.

Le plugin ne les réimplémente pas. Ce n'est pas seulement une question de doublon :
`GlobalFunctionCatalog.addFunctions` lève une `IllegalArgumentException` quand un plugin
enregistre un couple nom + signature déjà présent. Un plugin qui déclarerait
`dot_product(array(double), array(double))` **empêcherait le serveur de démarrer**.

Une surcharge sur `array(real)` est en revanche une signature distincte et cohabite sans
conflit.

Trino sait par ailleurs déjà faire un top-k exact global : `ORDER BY euclidean_distance(...)
LIMIT k`, avec un TopN distribué efficace.

### Ce que le plugin apporte

- les métriques absentes du moteur : `euclidean_squared_distance`, `manhattan_distance`
- les utilitaires `l2_norm` et `normalize_vector` (sur vecteurs normalisés, cosinus et produit
  scalaire coïncident, ce qui divise le coût du calcul)
- les surcharges `array(real)` de **toutes** les fonctions vectorielles, y compris les quatre
  natives : les tables d'embeddings sont fréquemment stockées en float32, et passer par un
  `CAST` vers `array(double)` double l'empreinte mémoire à chaque lecture
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
En `compile` : `com.google.guava:guava`, `io.trino:trino-array` (pour `ObjectBigArray`, comme
`trino-ml`).
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

Elles sont **alignées sur `ArrayVectorFunctions`**, pas définies indépendamment. Un utilisateur
mélangera nécessairement fonctions natives et fonctions du plugin dans une même requête : deux
sémantiques divergentes seraient un piège silencieux.

| Cas | Comportement | Source |
| --- | --- | --- |
| dimensions différentes | `TrinoException(INVALID_FUNCTION_ARGUMENT)`, « The arguments must have the same length » | natif |
| élément `NULL` dans un tableau | résultat `NULL` | natif (`cosineSimilarity`) |
| argument tableau `NULL` | résultat `NULL` | moteur |
| norme nulle | `TrinoException(INVALID_FUNCTION_ARGUMENT)`, « Vector magnitude cannot be zero » | natif (`cosineSimilarity`) |
| deux tableaux vides | `euclidean_squared_distance` et `manhattan_distance` valent `0` ; `l2_norm` vaut `0` ; `normalize_vector` lève (norme nulle) | cohérence |

Les fonctions susceptibles de renvoyer `NULL` doivent être annotées `@SqlNullable` et retourner
`Double` plutôt que `double`, comme `cosineSimilarity`.

`checkCondition` (`io.trino.util.Failures`) appartient à `trino-main` et **n'est pas dans le
SPI** : le plugin lève directement `new TrinoException(INVALID_FUNCTION_ARGUMENT, message)`.

## 6. Fonctions scalaires

**Nouvelles métriques**, déclinées en `array(double)` et `array(real)` :

| Signature | Résultat |
| --- | --- |
| `euclidean_squared_distance(x, y) -> double` | carré de la distance euclidienne, sans `sqrt` |
| `manhattan_distance(x, y) -> double` | distance L1 |
| `l2_norm(x) -> double` | norme euclidienne |
| `normalize_vector(x)` | vecteur de norme 1, du même type que l'entrée |

**Surcharges `array(real)` des fonctions natives**, qui n'existent qu'en `array(double)` dans
Trino 483 :

`euclidean_distance`, `dot_product`, `cosine_similarity`, `cosine_distance`.

Ces surcharges portent le **même nom** que les natives : c'est légal et souhaitable, la
résolution de surcharge se fait sur la signature. Réutiliser le nom natif garantit qu'une
requête écrite pour `array(double)` fonctionne à l'identique sur une colonne `array(real)`.

Conventions de nommage : le registre est celui de Trino, qui a choisi `euclidean_distance` et
non `l2_distance`. D'où `euclidean_squared_distance` et `manhattan_distance` plutôt que
`l2_squared_distance` et `l1_distance`. `l2_norm` fait exception, c'est le nom mathématique
universellement reconnu.

`normalize_vector` préserve le type d'entrée : `array(double)` donne `array(double)`,
`array(real)` donne `array(real)`. Promouvoir en `double` doublerait l'empreinte mémoire d'une
colonne float32 sans gain de précision réel.

Aucune fonction du plugin ne duplique une signature native. C'est une contrainte de démarrage
du serveur, pas une préférence de style, et un test doit la verrouiller.

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

`'euclidean'`, `'euclidean_squared'`, `'cosine'`, `'dot_product'`, `'manhattan'`. La
comparaison est insensible à la casse. Valeur inconnue :
`TrinoException(INVALID_FUNCTION_ARGUMENT)` listant les valeurs valides.

`Metric` est aussi le point d'implémentation des métriques : chaque constante porte le calcul
correspondant, y compris pour les quatre reprises du moteur. Le plugin ne peut pas appeler
`ArrayVectorFunctions`, qui appartient à `trino-main` et n'est pas dans le SPI.

### Sens de tri

`Metric` porte un drapeau `higherIsCloser`. `dot_product` est une similarité (plus grand = plus
proche) ; les quatre autres sont des distances. La valeur placée dans le résultat est toujours
la **valeur brute** de la métrique, jamais une valeur négée : ce que l'utilisateur lit est ce
que la fonction scalaire homonyme calculerait.

### État

Un tas binaire borné à `k` par groupe, avec le **pire élément à la racine** : ajouter un
candidat quand le tas est plein revient à comparer avec la racine et, le cas échéant, la
remplacer puis redescendre. Coût O(log k) par ligne, mémoire O(k) par groupe, contre O(n) pour
un `array_agg` suivi d'un tri.

Les clés, de type générique, sont conservées comme valeurs natives Java via
`TypeUtils.readNativeValue(Type, Block, int)` et réécrites avec
`TypeUtils.writeNativeValue(Type, BlockBuilder, Object)`. Les deux sont dans le SPI
(`io.trino.spi.type.TypeUtils`).

Ce choix est délibérément le plus simple des trois possibles. `TypedHeap`, employé en amont par
`min_by`, stocke les clés en mémoire plate avec des `MethodHandle` : il appartient à
`trino-main`, n'est pas accessible depuis un plugin, et sa réimplémentation coûterait plusieurs
centaines de lignes. Une variante à `BlockBuilder` compacté éviterait le boxing mais impose une
logique de compaction et de remappage de positions. Pour un `k` typique (10 à 100), le boxing
est négligeable devant le calcul de distance lui-même. Si le profilage le contredit un jour, le
tas est isolé derrière une classe et remplaçable sans toucher au reste.

L'état groupé indexe ces tas par `groupId` via `ObjectBigArray` (artefact `io.trino:trino-array`,
la même dépendance qu'utilise `trino-ml`).

`getEstimatedSize()` est implémenté : Trino comptabilise alors la mémoire et interrompt
proprement la requête au lieu de tomber en `OutOfMemoryError`.

### Distribution

`@CombineFunction` fusionne deux tas et conserve les `k` meilleurs. Trino exécute toujours une
agrégation en deux passes (partielle par worker, puis finale) : sans `@CombineFunction`
correcte, l'agrégation renvoie des résultats faux dès que les données sont réparties sur
plusieurs splits.

`KnnStateSerializer` sérialise l'état vers `ROW(BIGINT, VARCHAR, ARRAY(ROW(K, DOUBLE)))`,
c'est-à-dire `k`, la métrique, puis le contenu du tas. Ce type est déclaré dans
`@AccumulatorStateMetadata(typeParameters = "K", serializedType = "...")`, et le constructeur
de la fabrique comme celui du sérialiseur reçoivent le type de clé par
`@TypeParameter("K") Type keyType`, sur le modèle de `ArrayAggregationState` en amont.

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

**Le tout premier test à écrire est celui du chargement du plugin** : installer `VectorPlugin`
dans un `StandaloneQueryRunner` doit réussir. Il échoue immédiatement si une signature entre en
collision avec une fonction native, ce qui est le mode de panne le plus coûteux du projet
(serveur qui refuse de démarrer). Il coûte cinq lignes et couvre toutes les fonctions ajoutées
par la suite.

## 9. Contrainte d'environnement

Trino 483 impose **Java 25**.

- **JDK 25 : installé et vérifié**, Temurin 25.0.1+8, dans
  `C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot`
- **Maven : absent du système**, résolu par le Maven Wrapper committé dans le dépôt, qui se
  télécharge seul au premier appel
- **`JAVA_HOME` pointe sur le JDK 8** et le `java` du `PATH` est le 8. Le wrapper Maven
  sélectionne son JDK via `JAVA_HOME` : chaque commande de build doit donc être préfixée par
  `JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot"`. La variable globale
  n'est pas modifiée, d'autres projets de la machine dépendant du JDK 8.
- Le cache Maven local est quasi vide : le premier build téléchargera l'intégralité des
  dépendances et sera long

## 10. Extensions prévues (hors périmètre)

Points d'accroche déjà en place pour la suite :

- `Metric` et `VectorMath` sont réutilisables tels quels par un index ANN
- l'ANN nécessitera un connecteur (les PTF sont déclarées par un connecteur, pas par un plugin
  de fonctions nu) : ce sera un projet distinct, avec son propre cycle spec / plan
- un type `vector(n)` en float32 ne se justifiera qu'à ce moment-là, quand la compacité de
  stockage et la vérification de dimension à l'analyse auront une contrepartie réelle
