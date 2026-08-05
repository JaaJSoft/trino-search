# trino-vector

Plugin [Trino](https://trino.io) de fonctions vectorielles : distances, normalisation et
recherche des k plus proches voisins (KNN) exacte, directement en SQL.

Le plugin n'expose que des fonctions. Il ne définit ni catalogue ni connecteur : une fois le
JAR déposé dans le répertoire des plugins, les fonctions sont disponibles globalement, sur
n'importe quel catalogue.

## Fonctions

Les vecteurs sont des `array(double)` ou des `array(real)`. Aucun type custom, aucun `CAST`.

### Distances et similarités

| Fonction | Description |
| --- | --- |
| `l2_distance(x, y)` | distance euclidienne |
| `l2_squared_distance(x, y)` | carré de la distance euclidienne (même ordre de tri, sans `sqrt`) |
| `cosine_distance(x, y)` | `1 - cosine_similarity(x, y)` |
| `dot_product(x, y)` | produit scalaire (similarite : plus grand = plus proche) |
| `l1_distance(x, y)` | distance de Manhattan |

### Utilitaires

| Fonction | Description |
| --- | --- |
| `l2_norm(x)` | norme euclidienne |
| `normalize_vector(x)` | vecteur normalise (norme 1) |

### Agrégation

```sql
knn_agg(key, vector, query_vector, k, metric) -> array(row(key, distance))
```

Renvoie les `k` plus proches voisins de `query_vector` **par groupe**, triés du plus proche au
plus lointain. `metric` vaut `'l2'`, `'l2_squared'`, `'cosine'`, `'dot_product'` ou `'l1'`.

## Exemples

Top 10 global, en SQL standard :

```sql
SELECT id, l2_distance(embedding, ARRAY[0.1, 0.2, 0.3]) AS distance
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

Copier le contenu de `target/trino-vector-<version>/` dans `<trino>/plugin/vector/`, puis
redémarrer le serveur.

## État du projet

La v1 implémente le KNN **exact**. La recherche approximative (ANN) est prévue, mais n'est pas
encore implémentée.

## Licence

Apache License 2.0
