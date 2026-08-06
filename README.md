# trino-search

Plugin [Trino](https://trino.io) de fonctions de recherche. La première famille couvre les
vecteurs : distances, normalisation et recherche des k plus proches voisins (KNN) exacte,
directement en SQL.

<https://github.com/JaaJSoft/trino-search>

Le plugin n'expose que des fonctions. Il ne définit ni catalogue ni connecteur : une fois le
JAR déposé dans le répertoire des plugins, les fonctions sont disponibles globalement, sur
n'importe quel catalogue.

## Fonctions

Les vecteurs sont des `array(double)` ou des `array(real)`. Aucun type custom, aucun `CAST`.

Trino fournit déjà nativement `euclidean_distance`, `dot_product`, `cosine_similarity` et
`cosine_distance`, mais uniquement sur `array(double)`. Le plugin ne les réimplémente pas : il
en ajoute les surcharges `array(real)` et complète ce qui manque.

### Nouvelles métriques

| Fonction | Description |
| --- | --- |
| `euclidean_squared_distance(x, y)` | carré de la distance euclidienne (même ordre de tri, sans `sqrt`) |
| `manhattan_distance(x, y)` | distance L1 |
| `l2_norm(x)` | norme euclidienne |
| `normalize_vector(x)` | vecteur normalise (norme 1), du meme type que l'entree |

### Surcharges `array(real)`

`euclidean_distance`, `dot_product`, `cosine_similarity` et `cosine_distance` deviennent
utilisables directement sur des colonnes `array(real)`, sans `CAST` vers `array(double)` qui
doublerait l'empreinte memoire.

#### Résolution de surcharge sur un littéral non typé

Le plugin réutilise volontairement les noms natifs (`euclidean_distance`, `dot_product`,
`cosine_similarity`, `cosine_distance`) plutôt que des noms distincts, pour qu'une requête écrite
pour une colonne `array(double)` fonctionne sans modification sur une colonne `array(real)`.

Une conséquence : un littéral décimal non typé comme `ARRAY[0.1, 0.2]` se lie désormais à la
surcharge `array(real)` du plugin, pas à l'implémentation native `array(double)` du moteur (Trino
préfère la coercition la plus étroite quand plusieurs surcharges sont applicables par coercition).
Cela affecte deux choses :

- **La précision** : les éléments sont lus en float32 avant le calcul, ce qui donne un résultat
  légèrement différent du calcul natif en double précision.
- **Le traitement des `NULL`** : la surcharge `array(real)` renvoie `NULL` si un élément du
  tableau est `NULL`, alors que l'implémentation native `array(double)` lit un `NULL` comme `0.0`.

Ce comportement ne concerne que les littéraux non typés, utilisés en pratique surtout pour des
tests ou des requêtes ad hoc. Une colonne réellement typée `array(double)`, un `CAST(... AS
array(double))` explicite, ou des littéraux `DOUBLE 'x'`, se lient tous à l'implémentation native
et conservent son comportement (précision double, `NULL` lu comme `0.0`).

### Agrégation

```sql
knn_agg(key, vector, query_vector, k, metric) -> array(row(key, distance))
```

Renvoie les `k` plus proches voisins de `query_vector` **par groupe**, triés du plus proche au
plus lointain. `metric` vaut `'euclidean'`, `'euclidean_squared'`, `'cosine'`, `'dot_product'`
ou `'manhattan'`.

`knn_agg` est elle aussi soumise à la résolution de surcharge décrite ci-dessus : un littéral
décimal non typé comme `ARRAY[0.1, 0.2]` passé en `vector` ou `query_vector` se lie à la
surcharge `array(real)`, pas `array(double)`.

`k` et `metric` doivent rester constants au sein d'un même groupe : une valeur qui varie d'une
ligne à l'autre est rejetée avec une erreur plutôt que d'être résolue silencieusement par
« première ligne gagnante ».

## Exemples

Top 10 global sur une colonne `array(real)`, grace a la surcharge ajoutee par le plugin :

```sql
SELECT id, euclidean_distance(embedding, ARRAY[REAL '0.1', REAL '0.2', REAL '0.3']) AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

Top 3 par catégorie, avec l'agrégation :

```sql
SELECT category,
       knn_agg(id, embedding, ARRAY[0.1, 0.2, 0.3], 3, 'cosine') AS neighbors
FROM documents
GROUP BY category;
```

## Compatibilité

- Trino 483
- Java 25

## Installation

```bash
./mvnw clean package
```

Copier le contenu de `target/trino-search-<version>/` dans `<trino>/plugin/search/`, puis
redémarrer le serveur.

## État du projet

La v1 implémente le KNN **exact**. La recherche approximative (ANN) est prévue, mais n'est pas
encore implémentée.

## Licence

Apache License 2.0
