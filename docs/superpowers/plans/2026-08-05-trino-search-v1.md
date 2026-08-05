# Plan d'implémentation : trino-search v1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Livrer un plugin Trino qui ajoute les métriques vectorielles absentes du moteur, leurs
surcharges `array(real)`, et une agrégation `knn_agg` de recherche des k plus proches voisins
exacte par groupe.

**Architecture:** Un unique `SearchPlugin` qui ne remplit que `getFunctions()`. Toutes les
boucles de calcul vivent dans `VectorMath`, paramétré par un `VectorReader` qui abstrait la
lecture d'un élément `double` ou `real`. `Metric` associe un nom SQL à un calcul et à un sens de
comparaison. L'agrégation s'appuie sur un tas borné à k, avec un état sérialisable pour la
fusion distribuée.

**Tech Stack:** Java 25, Maven (wrapper), Trino 483 SPI, JUnit 5, AssertJ,
`StandaloneQueryRunner` / `AbstractTestQueryFramework` pour les tests SQL de bout en bout.

**Spec de référence :**
[`docs/superpowers/specs/2026-08-05-trino-search-plugin-design.md`](../specs/2026-08-05-trino-search-plugin-design.md)

## Global Constraints

Ces contraintes s'appliquent à **toutes** les tâches.

- Cible **Trino 483**, **Java 25** (`air.java.version` = `25.0.1`).
- Maven n'est pas installé et `JAVA_HOME` pointe sur un JDK 8. **Toute** commande de build est
  préfixée par `JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot"`.
- Package racine `dev.jaaj.trino.search`. Jamais `io.trino.*`.
- **Aucune fonction ne doit dupliquer le couple nom + signature d'une fonction native.** Trino
  483 fournit déjà, sur `array(double)` uniquement : `euclidean_distance`, `dot_product`,
  `cosine_similarity`, `cosine_distance`. Une collision fait échouer le démarrage du serveur.
- Sémantique des cas de bord alignée sur `io.trino.operator.scalar.ArrayVectorFunctions` :
  longueurs différentes → `TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have
  the same length")` ; présence d'un élément `NULL` → résultat `NULL` ; norme nulle →
  `TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero")`.
- `io.trino.util.Failures.checkCondition` appartient à `trino-main` et n'est **pas** dans le
  SPI : lever `new TrinoException(...)` directement.
- En-tête de licence Apache 2.0 sur chaque fichier source (copier celui d'un fichier de
  `plugin/trino-ml` en amont).
- Pas d'imports avec joker, accolades obligatoires autour de tout corps de `if` / `for` /
  `while`, pas de `@author`.
- Code, commentaires et messages de commit en anglais. Les documents de `docs/` sont en français.
- Jamais de tiret cadratin (`—`) ni demi-cadratin (`–`), y compris dans les commentaires et les
  messages de commit. Utiliser le trait d'union.

## Structure des fichiers

| Fichier | Responsabilité |
| --- | --- |
| `pom.xml` | build, versions, packaging `trino-plugin` |
| `mvnw`, `.mvn/wrapper/maven-wrapper.properties` | wrapper Maven |
| `src/main/java/dev/jaaj/trino/search/SearchPlugin.java` | point d'entrée `ServiceLoader`, liste des classes de fonctions |
| `.../search/vector/VectorReader.java` | lecture d'un élément, `double` ou `real` |
| `.../search/vector/VectorMath.java` | toutes les boucles de calcul |
| `.../search/vector/Metric.java` | nom SQL → calcul + sens de comparaison |
| `.../search/vector/VectorDistanceFunctions.java` | `@ScalarFunction` de distance |
| `.../search/vector/VectorFunctions.java` | `l2_norm`, `normalize_vector` |
| `.../search/vector/knn/KnnHeap.java` | tas borné à k, sans dépendance Trino |
| `.../search/vector/knn/KnnState.java` | interface d'état + `@AccumulatorStateMetadata` |
| `.../search/vector/knn/KnnStateFactory.java` | états simple et groupé |
| `.../search/vector/knn/KnnStateSerializer.java` | sérialisation pour la fusion distribuée |
| `.../search/vector/knn/KnnAggregation.java` | `@AggregationFunction("knn_agg")` |

`VectorMath` et `KnnHeap` ne dépendent d'aucun mécanisme d'agrégation et sont testables sans
moteur. C'est délibéré : ce sont les deux endroits où une erreur produit des nombres plausibles
plutôt qu'une panne visible.

---

### Task 1: Squelette Maven et chargement du plugin

Cette tâche livre le garde-fou le plus important du projet : un test qui échoue si une signature
entre en collision avec le moteur.

**Files:**
- Create: `pom.xml`
- Create: `mvnw`, `.mvn/wrapper/maven-wrapper.properties`
- Create: `src/main/java/dev/jaaj/trino/search/SearchPlugin.java`
- Test: `src/test/java/dev/jaaj/trino/search/TestSearchPlugin.java`

**Interfaces:**
- Produces: `dev.jaaj.trino.search.SearchPlugin`, classe publique implémentant
  `io.trino.spi.Plugin`, avec `Set<Class<?>> getFunctions()`. Chaque tâche ultérieure y ajoute
  ses classes de fonctions.

- [ ] **Step 1: Installer le wrapper Maven**

Trino fournit un wrapper en mode `only-script` (aucun JAR à committer). Le copier :

```bash
cp /d/programmation/trino/trino/mvnw ./mvnw
mkdir -p .mvn/wrapper
cat > .mvn/wrapper/maven-wrapper.properties <<'EOF'
wrapperVersion=3.3.4
distributionType=only-script
distributionUrl=https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.9.16/apache-maven-3.9.16-bin.zip
EOF
chmod +x mvnw
```

- [ ] **Step 2: Écrire le `pom.xml`**

`io.airlift:bom` gère les versions de `slice`, JUnit et AssertJ ; seuls les artefacts Trino
doivent être épinglés.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>io.airlift</groupId>
        <artifactId>airbase</artifactId>
        <version>395</version>
    </parent>

    <groupId>dev.jaaj.trino</groupId>
    <artifactId>trino-search</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>trino-plugin</packaging>
    <description>Trino - Search functions</description>

    <properties>
        <air.java.version>25.0.1</air.java.version>
        <dep.trino.version>483</dep.trino.version>
        <dep.airlift.version>439</dep.airlift.version>
    </properties>

    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>io.airlift</groupId>
                <artifactId>bom</artifactId>
                <version>${dep.airlift.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <dependencies>
        <dependency>
            <groupId>io.airlift</groupId>
            <artifactId>slice</artifactId>
            <scope>provided</scope>
        </dependency>

        <dependency>
            <groupId>io.trino</groupId>
            <artifactId>trino-spi</artifactId>
            <version>${dep.trino.version}</version>
            <scope>provided</scope>
        </dependency>

        <dependency>
            <groupId>io.trino</groupId>
            <artifactId>trino-main</artifactId>
            <version>${dep.trino.version}</version>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>io.trino</groupId>
            <artifactId>trino-testing</artifactId>
            <version>${dep.trino.version}</version>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>org.assertj</groupId>
            <artifactId>assertj-core</artifactId>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter-api</artifactId>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter-engine</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>io.trino</groupId>
                <artifactId>trino-maven-plugin</artifactId>
                <version>20</version>
                <extensions>true</extensions>
            </plugin>
        </plugins>
    </build>
</project>
```

Le premier build télécharge tout le cache Maven : compter plusieurs minutes. Si airbase
refuse une vérification annexe (spotbugs, javadoc), la désactiver par une propriété plutôt que
de contourner le parent. **Ne pas** désactiver checkstyle ni modernizer.

- [ ] **Step 3: Écrire le test de chargement (il doit échouer)**

```java
package dev.jaaj.trino.search;

import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;
import static org.assertj.core.api.Assertions.assertThat;

public class TestSearchPlugin
{
    @Test
    public void testPluginLoads()
    {
        try (QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build())) {
            queryRunner.installPlugin(new SearchPlugin());
            assertThat(queryRunner.execute("SELECT 1").getOnlyValue()).isEqualTo(1L);
        }
    }
}
```

Ce test paraît trivial. Il ne l'est pas : `installPlugin` traverse
`GlobalFunctionCatalog.addFunctions`, qui lève une `IllegalArgumentException` si une signature
déclarée existe déjà dans le moteur. C'est le seul test qui attrape ce mode de panne, et il le
fait pour toutes les fonctions ajoutées par les tâches suivantes.

- [ ] **Step 4: Lancer le test et vérifier l'échec**

Run: `JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot" ./mvnw test -Dtest=TestSearchPlugin`
Expected: échec de compilation, `SearchPlugin` n'existe pas.

- [ ] **Step 5: Écrire `SearchPlugin`**

```java
package dev.jaaj.trino.search;

import io.trino.spi.Plugin;

import java.util.Set;

public class SearchPlugin
        implements Plugin
{
    @Override
    public Set<Class<?>> getFunctions()
    {
        return Set.of();
    }
}
```

- [ ] **Step 6: Lancer le test et vérifier le succès**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestSearchPlugin`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pom.xml mvnw .mvn src
git commit -m "Add the Maven skeleton and an empty SearchPlugin

The plugin loading test is the guard against signature collisions with the
engine's own functions, which would make the server fail to start."
```

---

### Task 2: Noyau de calcul `VectorMath`

**Files:**
- Create: `src/main/java/dev/jaaj/trino/search/vector/VectorReader.java`
- Create: `src/main/java/dev/jaaj/trino/search/vector/VectorMath.java`
- Test: `src/test/java/dev/jaaj/trino/search/vector/TestVectorMath.java`

**Interfaces:**
- Produces:
  - `VectorReader`, interface fonctionnelle `double read(Block vector, int position)`, avec les
    constantes `VectorReader.DOUBLE_READER` et `VectorReader.REAL_READER`.
  - `VectorMath` (package-private, méthodes statiques) :
    - `void checkSameLength(Block first, Block second)`
    - `boolean hasNulls(Block first, Block second)`
    - `double euclideanSquared(Block first, Block second, VectorReader reader)`
    - `double euclidean(Block first, Block second, VectorReader reader)`
    - `double manhattan(Block first, Block second, VectorReader reader)`
    - `double dotProduct(Block first, Block second, VectorReader reader)`
    - `double norm(Block vector, VectorReader reader)`
    - `double cosineSimilarity(Block first, Block second, VectorReader reader)`

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

```java
package dev.jaaj.trino.search.vector;

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestVectorMath
{
    private static Block doubles(Double... values)
    {
        BlockBuilder builder = DOUBLE.createBlockBuilder(null, values.length);
        for (Double value : values) {
            if (value == null) {
                builder.appendNull();
            }
            else {
                DOUBLE.writeDouble(builder, value);
            }
        }
        return builder.build();
    }

    @Test
    public void testEuclidean()
    {
        // 3-4-5 triangle
        assertThat(VectorMath.euclidean(doubles(0.0, 0.0), doubles(3.0, 4.0), DOUBLE_READER))
                .isCloseTo(5.0, within(1e-12));
    }

    @Test
    public void testEuclideanSquaredAvoidsSqrt()
    {
        assertThat(VectorMath.euclideanSquared(doubles(0.0, 0.0), doubles(3.0, 4.0), DOUBLE_READER))
                .isCloseTo(25.0, within(1e-12));
    }

    @Test
    public void testManhattan()
    {
        assertThat(VectorMath.manhattan(doubles(1.0, -2.0), doubles(4.0, 2.0), DOUBLE_READER))
                .isCloseTo(7.0, within(1e-12));
    }

    @Test
    public void testDotProduct()
    {
        assertThat(VectorMath.dotProduct(doubles(1.0, 2.0), doubles(3.0, 4.0), DOUBLE_READER))
                .isCloseTo(11.0, within(1e-12));
    }

    @Test
    public void testNorm()
    {
        assertThat(VectorMath.norm(doubles(3.0, 4.0), DOUBLE_READER)).isCloseTo(5.0, within(1e-12));
    }

    @Test
    public void testCosineSimilarityOfIdenticalVectorsIsOne()
    {
        assertThat(VectorMath.cosineSimilarity(doubles(1.0, 2.0), doubles(1.0, 2.0), DOUBLE_READER))
                .isCloseTo(1.0, within(1e-12));
    }

    @Test
    public void testCosineSimilarityOfOrthogonalVectorsIsZero()
    {
        assertThat(VectorMath.cosineSimilarity(doubles(1.0, 0.0), doubles(0.0, 1.0), DOUBLE_READER))
                .isCloseTo(0.0, within(1e-12));
    }

    @Test
    public void testDistancesAreSymmetric()
    {
        Block first = doubles(1.0, 5.0, -3.0);
        Block second = doubles(-2.0, 0.5, 4.0);
        assertThat(VectorMath.euclidean(first, second, DOUBLE_READER))
                .isCloseTo(VectorMath.euclidean(second, first, DOUBLE_READER), within(1e-12));
        assertThat(VectorMath.manhattan(first, second, DOUBLE_READER))
                .isCloseTo(VectorMath.manhattan(second, first, DOUBLE_READER), within(1e-12));
    }

    @Test
    public void testDistanceToSelfIsZero()
    {
        Block vector = doubles(1.0, 5.0, -3.0);
        assertThat(VectorMath.euclidean(vector, vector, DOUBLE_READER)).isCloseTo(0.0, within(1e-12));
        assertThat(VectorMath.manhattan(vector, vector, DOUBLE_READER)).isCloseTo(0.0, within(1e-12));
    }

    @Test
    public void testTriangleInequality()
    {
        Block a = doubles(0.0, 0.0);
        Block b = doubles(3.0, 4.0);
        Block c = doubles(1.0, 7.0);
        assertThat(VectorMath.euclidean(a, c, DOUBLE_READER))
                .isLessThanOrEqualTo(VectorMath.euclidean(a, b, DOUBLE_READER) + VectorMath.euclidean(b, c, DOUBLE_READER) + 1e-12);
    }

    @Test
    public void testEmptyVectorsGiveZero()
    {
        assertThat(VectorMath.euclideanSquared(doubles(), doubles(), DOUBLE_READER)).isEqualTo(0.0);
        assertThat(VectorMath.manhattan(doubles(), doubles(), DOUBLE_READER)).isEqualTo(0.0);
        assertThat(VectorMath.norm(doubles(), DOUBLE_READER)).isEqualTo(0.0);
    }

    @Test
    public void testLengthMismatchIsRejected()
    {
        assertThatThrownBy(() -> VectorMath.checkSameLength(doubles(1.0), doubles(1.0, 2.0)))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("The arguments must have the same length");
    }

    @Test
    public void testNullDetection()
    {
        assertThat(VectorMath.hasNulls(doubles(1.0, null), doubles(1.0, 2.0))).isTrue();
        assertThat(VectorMath.hasNulls(doubles(1.0, 2.0), doubles(1.0, 2.0))).isFalse();
    }
}
```

- [ ] **Step 2: Lancer les tests et vérifier l'échec**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestVectorMath`
Expected: échec de compilation, `VectorMath` et `VectorReader` n'existent pas.

- [ ] **Step 3: Écrire `VectorReader`**

```java
package dev.jaaj.trino.search.vector;

import io.trino.spi.block.Block;

import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.RealType.REAL;

@FunctionalInterface
public interface VectorReader
{
    VectorReader DOUBLE_READER = DOUBLE::getDouble;
    VectorReader REAL_READER = REAL::getFloat;

    double read(Block vector, int position);
}
```

- [ ] **Step 4: Écrire `VectorMath`**

```java
package dev.jaaj.trino.search.vector;

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

final class VectorMath
{
    private VectorMath() {}

    static void checkSameLength(Block first, Block second)
    {
        if (first.getPositionCount() != second.getPositionCount()) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
        }
    }

    static boolean hasNulls(Block first, Block second)
    {
        return first.hasNull() || second.hasNull();
    }

    static double euclideanSquared(Block first, Block second, VectorReader reader)
    {
        double sum = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            double difference = reader.read(first, i) - reader.read(second, i);
            sum += difference * difference;
        }
        return sum;
    }

    static double euclidean(Block first, Block second, VectorReader reader)
    {
        return Math.sqrt(euclideanSquared(first, second, reader));
    }

    static double manhattan(Block first, Block second, VectorReader reader)
    {
        double sum = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            sum += Math.abs(reader.read(first, i) - reader.read(second, i));
        }
        return sum;
    }

    static double dotProduct(Block first, Block second, VectorReader reader)
    {
        double sum = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            sum += reader.read(first, i) * reader.read(second, i);
        }
        return sum;
    }

    static double norm(Block vector, VectorReader reader)
    {
        double sum = 0.0;
        for (int i = 0; i < vector.getPositionCount(); i++) {
            double value = reader.read(vector, i);
            sum += value * value;
        }
        return Math.sqrt(sum);
    }

    static double cosineSimilarity(Block first, Block second, VectorReader reader)
    {
        double firstMagnitude = 0.0;
        double secondMagnitude = 0.0;
        double dotProduct = 0.0;
        for (int i = 0; i < first.getPositionCount(); i++) {
            double firstValue = reader.read(first, i);
            double secondValue = reader.read(second, i);
            firstMagnitude += firstValue * firstValue;
            secondMagnitude += secondValue * secondValue;
            dotProduct += firstValue * secondValue;
        }

        if (firstMagnitude == 0 || secondMagnitude == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }
        return dotProduct / Math.sqrt(firstMagnitude * secondMagnitude);
    }
}
```

Les méthodes ne vérifient ni les longueurs ni les `NULL` : ces contrôles appartiennent aux
fonctions SQL, qui décident d'un `NULL` ou d'une exception. `VectorMath` ne fait que calculer.

- [ ] **Step 5: Lancer les tests et vérifier le succès**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestVectorMath`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/main/java/dev/jaaj/trino/search/vector src/test/java/dev/jaaj/trino/search/vector
git commit -m "Add the vector computation core

VectorReader abstracts reading a double or real element so that every metric
has a single loop rather than one per element type."
```

---

### Task 3: L'énumération `Metric`

**Files:**
- Create: `src/main/java/dev/jaaj/trino/search/vector/Metric.java`
- Test: `src/test/java/dev/jaaj/trino/search/vector/TestMetric.java`

**Interfaces:**
- Consumes: `VectorMath`, `VectorReader` (Task 2).
- Produces: `public enum Metric` avec les constantes `EUCLIDEAN`, `EUCLIDEAN_SQUARED`, `COSINE`,
  `DOT_PRODUCT`, `MANHATTAN`, et l'API :
  - `static Metric fromName(String name)`
  - `double compute(Block first, Block second, VectorReader reader)`
  - `boolean higherIsCloser()`

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

```java
package dev.jaaj.trino.search.vector;

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import org.junit.jupiter.api.Test;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

public class TestMetric
{
    private static Block doubles(double... values)
    {
        BlockBuilder builder = DOUBLE.createBlockBuilder(null, values.length);
        for (double value : values) {
            DOUBLE.writeDouble(builder, value);
        }
        return builder.build();
    }

    @Test
    public void testNamesAreResolved()
    {
        assertThat(Metric.fromName("euclidean")).isEqualTo(Metric.EUCLIDEAN);
        assertThat(Metric.fromName("euclidean_squared")).isEqualTo(Metric.EUCLIDEAN_SQUARED);
        assertThat(Metric.fromName("cosine")).isEqualTo(Metric.COSINE);
        assertThat(Metric.fromName("dot_product")).isEqualTo(Metric.DOT_PRODUCT);
        assertThat(Metric.fromName("manhattan")).isEqualTo(Metric.MANHATTAN);
    }

    @Test
    public void testNameResolutionIsCaseInsensitive()
    {
        assertThat(Metric.fromName("EUCLIDEAN")).isEqualTo(Metric.EUCLIDEAN);
        assertThat(Metric.fromName("Dot_Product")).isEqualTo(Metric.DOT_PRODUCT);
    }

    @Test
    public void testUnknownNameListsTheValidOnes()
    {
        assertThatThrownBy(() -> Metric.fromName("hamming"))
                .isInstanceOf(TrinoException.class)
                .hasMessageContaining("hamming")
                .hasMessageContaining("euclidean")
                .hasMessageContaining("manhattan");
    }

    @Test
    public void testOnlyDotProductRanksHigherAsCloser()
    {
        assertThat(Metric.DOT_PRODUCT.higherIsCloser()).isTrue();
        assertThat(Metric.EUCLIDEAN.higherIsCloser()).isFalse();
        assertThat(Metric.EUCLIDEAN_SQUARED.higherIsCloser()).isFalse();
        assertThat(Metric.COSINE.higherIsCloser()).isFalse();
        assertThat(Metric.MANHATTAN.higherIsCloser()).isFalse();
    }

    @Test
    public void testComputeMatchesTheCore()
    {
        Block first = doubles(0.0, 0.0);
        Block second = doubles(3.0, 4.0);
        assertThat(Metric.EUCLIDEAN.compute(first, second, DOUBLE_READER)).isCloseTo(5.0, within(1e-12));
        assertThat(Metric.EUCLIDEAN_SQUARED.compute(first, second, DOUBLE_READER)).isCloseTo(25.0, within(1e-12));
        assertThat(Metric.MANHATTAN.compute(first, second, DOUBLE_READER)).isCloseTo(7.0, within(1e-12));
        assertThat(Metric.DOT_PRODUCT.compute(doubles(1.0, 2.0), doubles(3.0, 4.0), DOUBLE_READER)).isCloseTo(11.0, within(1e-12));
    }

    @Test
    public void testCosineMetricIsADistance()
    {
        // identical vectors are at distance 0, not similarity 1
        assertThat(Metric.COSINE.compute(doubles(1.0, 2.0), doubles(1.0, 2.0), DOUBLE_READER))
                .isCloseTo(0.0, within(1e-12));
    }
}
```

- [ ] **Step 2: Lancer les tests et vérifier l'échec**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestMetric`
Expected: échec de compilation, `Metric` n'existe pas.

- [ ] **Step 3: Écrire `Metric`**

```java
package dev.jaaj.trino.search.vector;

import io.trino.spi.TrinoException;
import io.trino.spi.block.Block;

import java.util.Locale;
import java.util.stream.Stream;

import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;

public enum Metric
{
    EUCLIDEAN("euclidean", false) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return VectorMath.euclidean(first, second, reader);
        }
    },
    EUCLIDEAN_SQUARED("euclidean_squared", false) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return VectorMath.euclideanSquared(first, second, reader);
        }
    },
    COSINE("cosine", false) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return 1.0 - VectorMath.cosineSimilarity(first, second, reader);
        }
    },
    DOT_PRODUCT("dot_product", true) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return VectorMath.dotProduct(first, second, reader);
        }
    },
    MANHATTAN("manhattan", false) {
        @Override
        public double compute(Block first, Block second, VectorReader reader)
        {
            return VectorMath.manhattan(first, second, reader);
        }
    };

    private final String sqlName;
    private final boolean higherIsCloser;

    Metric(String sqlName, boolean higherIsCloser)
    {
        this.sqlName = sqlName;
        this.higherIsCloser = higherIsCloser;
    }

    public abstract double compute(Block first, Block second, VectorReader reader);

    /**
     * Dot product is a similarity, every other metric is a distance. The heap needs the
     * direction; the value handed back to the user is always the raw metric value.
     */
    public boolean higherIsCloser()
    {
        return higherIsCloser;
    }

    public static Metric fromName(String name)
    {
        String normalized = name.toLowerCase(Locale.ROOT);
        for (Metric metric : values()) {
            if (metric.sqlName.equals(normalized)) {
                return metric;
            }
        }
        throw new TrinoException(INVALID_FUNCTION_ARGUMENT,
                "Unknown metric '%s', expected one of: %s".formatted(
                        name,
                        Stream.of(values()).map(metric -> metric.sqlName).toList()));
    }
}
```

- [ ] **Step 4: Lancer les tests et vérifier le succès**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestMetric`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/java/dev/jaaj/trino/search/vector/Metric.java src/test/java/dev/jaaj/trino/search/vector/TestMetric.java
git commit -m "Add the Metric enum

Each constant carries its own computation and whether a higher value means a
closer neighbour, which is what the KNN heap needs to order candidates."
```

---

### Task 4: Fonctions scalaires sur `array(double)`

Uniquement les fonctions **absentes** du moteur.

**Files:**
- Create: `src/main/java/dev/jaaj/trino/search/vector/VectorDistanceFunctions.java`
- Create: `src/main/java/dev/jaaj/trino/search/vector/VectorFunctions.java`
- Modify: `src/main/java/dev/jaaj/trino/search/SearchPlugin.java`
- Test: `src/test/java/dev/jaaj/trino/search/vector/TestVectorFunctionQueries.java`

**Interfaces:**
- Consumes: `VectorMath`, `VectorReader` (Task 2).
- Produces les fonctions SQL `euclidean_squared_distance`, `manhattan_distance`, `l2_norm`,
  `normalize_vector` sur `array(double)`.

- [ ] **Step 1: Écrire les tests SQL (ils doivent échouer)**

```java
package dev.jaaj.trino.search.vector;

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;

public class TestVectorFunctionQueries
        extends AbstractTestQueryFramework
{
    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testEuclideanSquaredDistance()
    {
        assertQuery("SELECT euclidean_squared_distance(ARRAY[0.0, 0.0], ARRAY[3.0, 4.0])", "SELECT 25.0");
    }

    @Test
    public void testManhattanDistance()
    {
        assertQuery("SELECT manhattan_distance(ARRAY[1.0, -2.0], ARRAY[4.0, 2.0])", "SELECT 7.0");
    }

    @Test
    public void testL2Norm()
    {
        assertQuery("SELECT l2_norm(ARRAY[3.0, 4.0])", "SELECT 5.0");
    }

    @Test
    public void testNormalizeVectorHasUnitNorm()
    {
        assertQuery("SELECT l2_norm(normalize_vector(ARRAY[3.0, 4.0]))", "SELECT 1.0");
    }

    @Test
    public void testNormalizeVectorValues()
    {
        assertQuery("SELECT normalize_vector(ARRAY[3.0, 4.0])", "SELECT ARRAY[0.6, 0.8]");
    }

    @Test
    public void testLengthMismatchIsRejected()
    {
        assertQueryFails(
                "SELECT manhattan_distance(ARRAY[1.0], ARRAY[1.0, 2.0])",
                ".*The arguments must have the same length.*");
    }

    @Test
    public void testNullElementGivesNull()
    {
        assertQuery("SELECT manhattan_distance(ARRAY[1.0, NULL], ARRAY[1.0, 2.0]) IS NULL", "SELECT true");
        assertQuery("SELECT l2_norm(ARRAY[1.0, NULL]) IS NULL", "SELECT true");
    }

    @Test
    public void testNullArgumentGivesNull()
    {
        assertQuery("SELECT manhattan_distance(NULL, ARRAY[1.0, 2.0]) IS NULL", "SELECT true");
    }

    @Test
    public void testNormalizeZeroVectorIsRejected()
    {
        assertQueryFails(
                "SELECT normalize_vector(ARRAY[0.0, 0.0])",
                ".*Vector magnitude cannot be zero.*");
    }

    @Test
    public void testEmptyVectors()
    {
        assertQuery("SELECT euclidean_squared_distance(ARRAY[], ARRAY[])", "SELECT 0.0");
        assertQuery("SELECT l2_norm(CAST(ARRAY[] AS array(double)))", "SELECT 0.0");
    }
}
```

- [ ] **Step 2: Lancer les tests et vérifier l'échec**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestVectorFunctionQueries`
Expected: FAIL, `Function 'euclidean_squared_distance' not registered`.

- [ ] **Step 3: Écrire `VectorDistanceFunctions`**

```java
package dev.jaaj.trino.search.vector;

import io.trino.spi.block.Block;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;

public final class VectorDistanceFunctions
{
    private VectorDistanceFunctions() {}

    @Description("Calculates the squared euclidean distance between two vectors")
    @ScalarFunction("euclidean_squared_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double euclideanSquaredDistance(@SqlType("array(double)") Block first, @SqlType("array(double)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.euclideanSquared(first, second, DOUBLE_READER);
    }

    @Description("Calculates the manhattan distance between two vectors")
    @ScalarFunction("manhattan_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double manhattanDistance(@SqlType("array(double)") Block first, @SqlType("array(double)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.manhattan(first, second, DOUBLE_READER);
    }
}
```

- [ ] **Step 4: Écrire `VectorFunctions`**

`normalize_vector` construit le bloc des **éléments** et le renvoie ; Trino l'enveloppe dans le
tableau. C'est le pattern de `QuantileDigestFunctions.valuesAtQuantilesDouble` en amont.

```java
package dev.jaaj.trino.search.vector;

import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.function.Description;
import io.trino.spi.function.ScalarFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.type.StandardTypes;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static io.trino.spi.type.DoubleType.DOUBLE;

public final class VectorFunctions
{
    private VectorFunctions() {}

    @Description("Calculates the euclidean norm of a vector")
    @ScalarFunction("l2_norm")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double l2Norm(@SqlType("array(double)") Block vector)
    {
        if (vector.hasNull()) {
            return null;
        }
        return VectorMath.norm(vector, DOUBLE_READER);
    }

    @Description("Scales a vector to unit norm")
    @ScalarFunction("normalize_vector")
    @SqlType("array(double)")
    @SqlNullable
    public static Block normalizeVector(@SqlType("array(double)") Block vector)
    {
        if (vector.hasNull()) {
            return null;
        }
        double norm = VectorMath.norm(vector, DOUBLE_READER);
        if (norm == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }

        BlockBuilder output = DOUBLE.createFixedSizeBlockBuilder(vector.getPositionCount());
        for (int i = 0; i < vector.getPositionCount(); i++) {
            DOUBLE.writeDouble(output, DOUBLE_READER.read(vector, i) / norm);
        }
        return output.build();
    }
}
```

Ajouter les imports `io.trino.spi.TrinoException` et
`static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT`.

Un vecteur vide a une norme de 0, donc `normalize_vector(ARRAY[])` lève. C'est cohérent :
il n'existe pas de vecteur unitaire de dimension 0.

- [ ] **Step 5: Enregistrer les fonctions dans `SearchPlugin`**

```java
    @Override
    public Set<Class<?>> getFunctions()
    {
        return Set.of(
                VectorDistanceFunctions.class,
                VectorFunctions.class);
    }
```

avec les imports `dev.jaaj.trino.search.vector.VectorDistanceFunctions` et
`dev.jaaj.trino.search.vector.VectorFunctions`.

- [ ] **Step 6: Lancer toute la suite et vérifier le succès**

Run: `JAVA_HOME="..." ./mvnw test`
Expected: PASS, y compris `TestSearchPlugin` qui valide l'absence de collision.

- [ ] **Step 7: Commit**

```bash
git add src
git commit -m "Add the vector metrics missing from the engine

euclidean_squared_distance orders like euclidean_distance without the square
root, manhattan_distance covers L1, and normalize_vector pairs with l2_norm so
cosine and dot product coincide on normalized data."
```

---

### Task 5: Surcharges `array(real)`

C'est ici que le plugin apporte le plus aux tables d'embeddings stockées en float32.

**Files:**
- Modify: `src/main/java/dev/jaaj/trino/search/vector/VectorDistanceFunctions.java`
- Modify: `src/main/java/dev/jaaj/trino/search/vector/VectorFunctions.java`
- Test: `src/test/java/dev/jaaj/trino/search/vector/TestVectorRealFunctionQueries.java`

**Interfaces:**
- Produces, sur `array(real)` : `euclidean_squared_distance`, `manhattan_distance`, `l2_norm`,
  `normalize_vector` (qui renvoie `array(real)`), **plus** les quatre noms natifs
  `euclidean_distance`, `dot_product`, `cosine_similarity`, `cosine_distance`.

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

```java
package dev.jaaj.trino.search.vector;

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;

public class TestVectorRealFunctionQueries
        extends AbstractTestQueryFramework
{
    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testEuclideanDistanceOnReal()
    {
        assertQuery("SELECT euclidean_distance(ARRAY[REAL '0.0', REAL '0.0'], ARRAY[REAL '3.0', REAL '4.0'])", "SELECT 5.0");
    }

    @Test
    public void testDotProductOnReal()
    {
        assertQuery("SELECT dot_product(ARRAY[REAL '1.0', REAL '2.0'], ARRAY[REAL '3.0', REAL '4.0'])", "SELECT 11.0");
    }

    @Test
    public void testCosineSimilarityOnReal()
    {
        assertQuery("SELECT cosine_similarity(ARRAY[REAL '1.0', REAL '0.0'], ARRAY[REAL '0.0', REAL '1.0'])", "SELECT 0.0");
    }

    @Test
    public void testCosineDistanceOnReal()
    {
        assertQuery("SELECT cosine_distance(ARRAY[REAL '1.0', REAL '2.0'], ARRAY[REAL '1.0', REAL '2.0'])", "SELECT 0.0");
    }

    @Test
    public void testNewMetricsOnReal()
    {
        assertQuery("SELECT euclidean_squared_distance(ARRAY[REAL '0.0', REAL '0.0'], ARRAY[REAL '3.0', REAL '4.0'])", "SELECT 25.0");
        assertQuery("SELECT manhattan_distance(ARRAY[REAL '1.0', REAL '-2.0'], ARRAY[REAL '4.0', REAL '2.0'])", "SELECT 7.0");
        assertQuery("SELECT l2_norm(ARRAY[REAL '3.0', REAL '4.0'])", "SELECT 5.0");
    }

    @Test
    public void testNormalizeVectorKeepsRealType()
    {
        assertQuery("SELECT normalize_vector(ARRAY[REAL '3.0', REAL '4.0'])", "SELECT ARRAY[REAL '0.6', REAL '0.8']");
    }

    @Test
    public void testDoubleOverloadStillResolves()
    {
        // adding real overloads must not shadow the double ones
        assertQuery("SELECT euclidean_distance(ARRAY[0.0, 0.0], ARRAY[3.0, 4.0])", "SELECT 5.0");
        assertQuery("SELECT manhattan_distance(ARRAY[1.0, -2.0], ARRAY[4.0, 2.0])", "SELECT 7.0");
    }

    @Test
    public void testRealLengthMismatchIsRejected()
    {
        assertQueryFails(
                "SELECT euclidean_distance(ARRAY[REAL '1.0'], ARRAY[REAL '1.0', REAL '2.0'])",
                ".*The arguments must have the same length.*");
    }
}
```

- [ ] **Step 2: Lancer les tests et vérifier l'échec**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestVectorRealFunctionQueries`
Expected: FAIL, aucune surcharge `array(real)` ne résout.

- [ ] **Step 3: Ajouter les surcharges `real` à `VectorDistanceFunctions`**

Ajouter dans la même classe, à la suite des méthodes existantes :

```java
    @Description("Calculates the squared euclidean distance between two vectors")
    @ScalarFunction("euclidean_squared_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double euclideanSquaredDistanceReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.euclideanSquared(first, second, REAL_READER);
    }

    @Description("Calculates the manhattan distance between two vectors")
    @ScalarFunction("manhattan_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double manhattanDistanceReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.manhattan(first, second, REAL_READER);
    }

    @Description("Calculates the euclidean distance between two vectors")
    @ScalarFunction("euclidean_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double euclideanDistanceReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.euclidean(first, second, REAL_READER);
    }

    @Description("Calculates the dot product between two vectors")
    @ScalarFunction("dot_product")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double dotProductReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.dotProduct(first, second, REAL_READER);
    }

    @Description("Calculates the cosine similarity between two vectors")
    @ScalarFunction("cosine_similarity")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double cosineSimilarityReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        VectorMath.checkSameLength(first, second);
        if (VectorMath.hasNulls(first, second)) {
            return null;
        }
        return VectorMath.cosineSimilarity(first, second, REAL_READER);
    }

    @Description("Calculates the cosine distance between two vectors")
    @ScalarFunction("cosine_distance")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double cosineDistanceReal(@SqlType("array(real)") Block first, @SqlType("array(real)") Block second)
    {
        Double similarity = cosineSimilarityReal(first, second);
        if (similarity == null) {
            return null;
        }
        return 1.0 - similarity;
    }
```

Ajouter l'import statique `dev.jaaj.trino.search.vector.VectorReader.REAL_READER`.

Les noms Java diffèrent (`...Real`) mais les noms SQL sont identiques aux natifs : c'est la
signature qui distingue les surcharges, et réutiliser le nom natif est ce qui rend une requête
portable d'une colonne `array(double)` vers une colonne `array(real)`.

- [ ] **Step 4: Ajouter les surcharges `real` à `VectorFunctions`**

```java
    @Description("Calculates the euclidean norm of a vector")
    @ScalarFunction("l2_norm")
    @SqlType(StandardTypes.DOUBLE)
    @SqlNullable
    public static Double l2NormReal(@SqlType("array(real)") Block vector)
    {
        if (vector.hasNull()) {
            return null;
        }
        return VectorMath.norm(vector, REAL_READER);
    }

    @Description("Scales a vector to unit norm")
    @ScalarFunction("normalize_vector")
    @SqlType("array(real)")
    @SqlNullable
    public static Block normalizeVectorReal(@SqlType("array(real)") Block vector)
    {
        if (vector.hasNull()) {
            return null;
        }
        double norm = VectorMath.norm(vector, REAL_READER);
        if (norm == 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "Vector magnitude cannot be zero");
        }

        BlockBuilder output = REAL.createFixedSizeBlockBuilder(vector.getPositionCount());
        for (int i = 0; i < vector.getPositionCount(); i++) {
            REAL.writeFloat(output, (float) (REAL_READER.read(vector, i) / norm));
        }
        return output.build();
    }
```

Ajouter les imports `static io.trino.spi.type.RealType.REAL` et
`static dev.jaaj.trino.search.vector.VectorReader.REAL_READER`.

- [ ] **Step 5: Lancer toute la suite**

Run: `JAVA_HOME="..." ./mvnw test`
Expected: PASS. Si `TestSearchPlugin` échoue sur « Function already registered », c'est qu'une
surcharge a été déclarée sur `array(double)` au lieu de `array(real)`.

- [ ] **Step 6: Commit**

```bash
git add src
git commit -m "Add array(real) overloads for every vector function

Embedding tables are commonly stored as float32. Casting them to array(double)
to reach the engine's vector functions doubles the memory read per row, so the
plugin registers the same function names against the real signature."
```

---

### Task 6: Le tas borné `KnnHeap`

Structure de données pure, sans dépendance à l'agrégation. C'est le composant où une erreur
donne un résultat plausible mais faux, donc celui qui mérite le plus de tests.

**Files:**
- Create: `src/main/java/dev/jaaj/trino/search/vector/knn/KnnHeap.java`
- Test: `src/test/java/dev/jaaj/trino/search/vector/knn/TestKnnHeap.java`

**Interfaces:**
- Produces `public final class KnnHeap` :
  - `KnnHeap(int k, boolean higherIsCloser)`
  - `void add(Object key, double distance)`
  - `void mergeFrom(KnnHeap other)`
  - `int size()`
  - `List<KnnHeap.Neighbour> drainSorted()` où
    `record Neighbour(Object key, double distance)`, trié du plus proche au plus lointain
  - `long estimatedSizeInBytes()`

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

```java
package dev.jaaj.trino.search.vector.knn;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

public class TestKnnHeap
{
    private static List<Object> keysOf(KnnHeap heap)
    {
        return heap.drainSorted().stream().map(KnnHeap.Neighbour::key).toList();
    }

    @Test
    public void testKeepsTheKSmallestDistances()
    {
        KnnHeap heap = new KnnHeap(3, false);
        heap.add("e", 5.0);
        heap.add("a", 1.0);
        heap.add("d", 4.0);
        heap.add("b", 2.0);
        heap.add("c", 3.0);

        assertThat(keysOf(heap)).containsExactly("a", "b", "c");
    }

    @Test
    public void testKeepsTheKLargestWhenHigherIsCloser()
    {
        KnnHeap heap = new KnnHeap(2, true);
        heap.add("a", 1.0);
        heap.add("e", 5.0);
        heap.add("c", 3.0);

        assertThat(keysOf(heap)).containsExactly("e", "c");
    }

    @Test
    public void testInsertionOrderDoesNotMatter()
    {
        KnnHeap ascending = new KnnHeap(3, false);
        KnnHeap descending = new KnnHeap(3, false);
        for (int i = 0; i < 20; i++) {
            ascending.add(i, i);
            descending.add(19 - i, 19 - i);
        }

        assertThat(keysOf(ascending)).isEqualTo(keysOf(descending));
    }

    @Test
    public void testFewerElementsThanK()
    {
        KnnHeap heap = new KnnHeap(10, false);
        heap.add("a", 1.0);
        heap.add("b", 2.0);

        assertThat(heap.size()).isEqualTo(2);
        assertThat(keysOf(heap)).containsExactly("a", "b");
    }

    @Test
    public void testEmptyHeap()
    {
        KnnHeap heap = new KnnHeap(5, false);

        assertThat(heap.size()).isZero();
        assertThat(heap.drainSorted()).isEmpty();
    }

    @Test
    public void testNullKeysAreKept()
    {
        KnnHeap heap = new KnnHeap(2, false);
        heap.add(null, 1.0);
        heap.add("b", 2.0);

        assertThat(keysOf(heap)).containsExactly(null, "b");
    }

    @Test
    public void testMergeKeepsTheBestAcrossBothHeaps()
    {
        KnnHeap left = new KnnHeap(3, false);
        left.add("a", 1.0);
        left.add("d", 4.0);
        left.add("f", 6.0);

        KnnHeap right = new KnnHeap(3, false);
        right.add("b", 2.0);
        right.add("c", 3.0);
        right.add("e", 5.0);

        left.mergeFrom(right);

        assertThat(keysOf(left)).containsExactly("a", "b", "c");
    }

    @Test
    public void testMergeWithAnEmptyHeapChangesNothing()
    {
        KnnHeap left = new KnnHeap(2, false);
        left.add("a", 1.0);
        left.mergeFrom(new KnnHeap(2, false));

        assertThat(keysOf(left)).containsExactly("a");
    }

    @Test
    public void testDrainDoesNotMutate()
    {
        KnnHeap heap = new KnnHeap(2, false);
        heap.add("a", 1.0);
        heap.add("b", 2.0);

        assertThat(keysOf(heap)).containsExactly("a", "b");
        assertThat(keysOf(heap)).containsExactly("a", "b");
    }

    @Test
    public void testMatchesABruteForceSortOnRandomData()
    {
        int k = 7;
        KnnHeap heap = new KnnHeap(k, false);
        java.util.Random random = new java.util.Random(42);
        java.util.List<Double> all = new java.util.ArrayList<>();
        for (int i = 0; i < 500; i++) {
            double distance = random.nextDouble() * 1000;
            all.add(distance);
            heap.add(i, distance);
        }

        List<Double> expected = all.stream().sorted().limit(k).toList();
        List<Double> actual = heap.drainSorted().stream().map(KnnHeap.Neighbour::distance).toList();

        assertThat(actual).isEqualTo(expected);
    }
}
```

- [ ] **Step 2: Lancer les tests et vérifier l'échec**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestKnnHeap`
Expected: échec de compilation, `KnnHeap` n'existe pas.

- [ ] **Step 3: Écrire `KnnHeap`**

```java
package dev.jaaj.trino.search.vector.knn;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;

/**
 * Bounded heap keeping the k nearest neighbours seen so far.
 * <p>
 * The root holds the <em>worst</em> retained candidate, so deciding whether a new candidate
 * belongs in the result is a single comparison against position 0.
 */
public final class KnnHeap
{
    private static final long INSTANCE_SIZE = instanceSize(KnnHeap.class);

    public record Neighbour(Object key, double distance) {}

    private final int k;
    private final boolean higherIsCloser;
    private final double[] distances;
    private final Object[] keys;
    private int size;

    public KnnHeap(int k, boolean higherIsCloser)
    {
        this.k = k;
        this.higherIsCloser = higherIsCloser;
        this.distances = new double[k];
        this.keys = new Object[k];
    }

    public int size()
    {
        return size;
    }

    public void add(Object key, double distance)
    {
        if (size < k) {
            distances[size] = distance;
            keys[size] = key;
            siftUp(size);
            size++;
            return;
        }
        if (isCloser(distance, distances[0])) {
            distances[0] = distance;
            keys[0] = key;
            siftDown(0);
        }
    }

    public void mergeFrom(KnnHeap other)
    {
        for (int i = 0; i < other.size; i++) {
            add(other.keys[i], other.distances[i]);
        }
    }

    public List<Neighbour> drainSorted()
    {
        List<Neighbour> neighbours = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            neighbours.add(new Neighbour(keys[i], distances[i]));
        }
        Comparator<Neighbour> byDistance = Comparator.comparingDouble(Neighbour::distance);
        neighbours.sort(higherIsCloser ? byDistance.reversed() : byDistance);
        return neighbours;
    }

    public long estimatedSizeInBytes()
    {
        return INSTANCE_SIZE + sizeOf(distances) + sizeOf(keys);
    }

    private boolean isCloser(double candidate, double incumbent)
    {
        return higherIsCloser ? candidate > incumbent : candidate < incumbent;
    }

    private void siftUp(int start)
    {
        int index = start;
        while (index > 0) {
            int parent = (index - 1) >>> 1;
            if (isCloser(distances[parent], distances[index])) {
                break;
            }
            swap(index, parent);
            index = parent;
        }
    }

    private void siftDown(int start)
    {
        int index = start;
        while (true) {
            int left = 2 * index + 1;
            int right = left + 1;
            int worst = index;
            if (left < size && isCloser(distances[worst], distances[left])) {
                worst = left;
            }
            if (right < size && isCloser(distances[worst], distances[right])) {
                worst = right;
            }
            if (worst == index) {
                return;
            }
            swap(index, worst);
            index = worst;
        }
    }

    private void swap(int first, int second)
    {
        double distance = distances[first];
        distances[first] = distances[second];
        distances[second] = distance;
        Object key = keys[first];
        keys[first] = keys[second];
        keys[second] = key;
    }
}
```

`estimatedSizeInBytes` ne compte pas la taille propre des clés : pour les types courants
(`bigint`, `varchar` court) c'est une sous-estimation modeste, et la comptabilité mémoire de
Trino n'a pas besoin d'être exacte, seulement de croître avec l'usage réel.

- [ ] **Step 4: Lancer les tests et vérifier le succès**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestKnnHeap`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/main/java/dev/jaaj/trino/search/vector/knn src/test/java/dev/jaaj/trino/search/vector/knn
git commit -m "Add the bounded KNN heap

Keeping the worst retained candidate at the root turns the per-row decision
into one comparison, which bounds memory to k per group instead of collecting
every row and sorting."
```

---

### Task 7: État de l'agrégation

**Files:**
- Create: `src/main/java/dev/jaaj/trino/search/vector/knn/KnnState.java`
- Create: `src/main/java/dev/jaaj/trino/search/vector/knn/KnnStateFactory.java`
- Create: `src/main/java/dev/jaaj/trino/search/vector/knn/KnnStateSerializer.java`

**Interfaces:**
- Consumes: `KnnHeap` (Task 6).
- Produces `public interface KnnState extends AccumulatorState` :
  - `KnnHeap getHeap()`
  - `void setHeap(KnnHeap heap)`
  - `int getK()` / `void setK(int k)`
  - `String getMetricName()` / `void setMetricName(String metricName)`

Cette tâche n'a pas de test propre : son comportement n'est observable qu'à travers
l'agrégation, testée en Task 8, et un test qui manipulerait l'état à la main ne prouverait rien
que Task 8 ne prouve mieux.

- [ ] **Step 1: Écrire `KnnState`**

```java
package dev.jaaj.trino.search.vector.knn;

import io.trino.spi.function.AccumulatorState;
import io.trino.spi.function.AccumulatorStateMetadata;

@AccumulatorStateMetadata(
        stateFactoryClass = KnnStateFactory.class,
        stateSerializerClass = KnnStateSerializer.class,
        typeParameters = "K",
        serializedType = "ROW(BIGINT, VARCHAR, ARRAY(ROW(K, DOUBLE)))")
public interface KnnState
        extends AccumulatorState
{
    KnnHeap getHeap();

    void setHeap(KnnHeap heap);

    int getK();

    void setK(int k);

    String getMetricName();

    void setMetricName(String metricName);
}
```

`k` et le nom de la métrique **doivent** figurer dans le type sérialisé. L'agrégation finale ne
voit jamais les lignes d'origine, seulement des états partiels désérialisés : sans ces deux
champs, elle ignorerait combien de voisins garder et dans quel sens comparer. Ce défaut ne se
manifeste qu'en exécution distribuée, jamais sur une partition unique.

- [ ] **Step 2: Écrire `KnnStateFactory`**

```java
package dev.jaaj.trino.search.vector.knn;

import io.trino.spi.function.AccumulatorStateFactory;
import io.trino.spi.function.GroupedAccumulatorState;
import io.trino.spi.function.TypeParameter;
import io.trino.spi.type.Type;

import java.util.Arrays;

import static io.airlift.slice.SizeOf.instanceSize;
import static io.airlift.slice.SizeOf.sizeOf;
import static java.util.Objects.requireNonNull;

public class KnnStateFactory
        implements AccumulatorStateFactory<KnnState>
{
    private final Type keyType;

    public KnnStateFactory(@TypeParameter("K") Type keyType)
    {
        this.keyType = requireNonNull(keyType, "keyType is null");
    }

    public Type getKeyType()
    {
        return keyType;
    }

    @Override
    public KnnState createSingleState()
    {
        return new SingleKnnState();
    }

    @Override
    public KnnState createGroupedState()
    {
        return new GroupedKnnState();
    }

    public static class SingleKnnState
            implements KnnState
    {
        private static final long INSTANCE_SIZE = instanceSize(SingleKnnState.class);

        private KnnHeap heap;
        private int k;
        private String metricName;

        @Override
        public KnnHeap getHeap()
        {
            return heap;
        }

        @Override
        public void setHeap(KnnHeap heap)
        {
            this.heap = heap;
        }

        @Override
        public int getK()
        {
            return k;
        }

        @Override
        public void setK(int k)
        {
            this.k = k;
        }

        @Override
        public String getMetricName()
        {
            return metricName;
        }

        @Override
        public void setMetricName(String metricName)
        {
            this.metricName = metricName;
        }

        @Override
        public long getEstimatedSize()
        {
            return INSTANCE_SIZE + (heap == null ? 0 : heap.estimatedSizeInBytes());
        }
    }

    public static class GroupedKnnState
            implements GroupedAccumulatorState, KnnState
    {
        private static final long INSTANCE_SIZE = instanceSize(GroupedKnnState.class);

        private KnnHeap[] heaps = new KnnHeap[0];
        private int[] ks = new int[0];
        private String[] metricNames = new String[0];
        private int groupId;

        @Override
        public void setGroupId(int groupId)
        {
            this.groupId = groupId;
        }

        @Override
        public void ensureCapacity(int size)
        {
            if (size > heaps.length) {
                heaps = Arrays.copyOf(heaps, size);
                ks = Arrays.copyOf(ks, size);
                metricNames = Arrays.copyOf(metricNames, size);
            }
        }

        @Override
        public KnnHeap getHeap()
        {
            return heaps[groupId];
        }

        @Override
        public void setHeap(KnnHeap heap)
        {
            heaps[groupId] = heap;
        }

        @Override
        public int getK()
        {
            return ks[groupId];
        }

        @Override
        public void setK(int k)
        {
            ks[groupId] = k;
        }

        @Override
        public String getMetricName()
        {
            return metricNames[groupId];
        }

        @Override
        public void setMetricName(String metricName)
        {
            metricNames[groupId] = metricName;
        }

        @Override
        public long getEstimatedSize()
        {
            long size = INSTANCE_SIZE + sizeOf(heaps) + sizeOf(ks) + sizeOf(metricNames);
            for (KnnHeap heap : heaps) {
                if (heap != null) {
                    size += heap.estimatedSizeInBytes();
                }
            }
            return size;
        }
    }
}
```

- [ ] **Step 3: Écrire `KnnStateSerializer`**

```java
package dev.jaaj.trino.search.vector.knn;

import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.RowBlockBuilder;
import io.trino.spi.block.SqlRow;
import io.trino.spi.function.AccumulatorStateSerializer;
import io.trino.spi.function.TypeParameter;
import io.trino.spi.type.ArrayType;
import io.trino.spi.type.RowType;
import io.trino.spi.type.Type;

import java.util.List;

import static io.trino.spi.type.BigintType.BIGINT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.TypeUtils.readNativeValue;
import static io.trino.spi.type.TypeUtils.writeNativeValue;
import static io.trino.spi.type.VarcharType.VARCHAR;
import static java.util.Objects.requireNonNull;

public class KnnStateSerializer
        implements AccumulatorStateSerializer<KnnState>
{
    private final Type keyType;
    private final RowType neighbourType;
    private final ArrayType neighbourArrayType;
    private final RowType serializedType;

    public KnnStateSerializer(@TypeParameter("K") Type keyType)
    {
        this.keyType = requireNonNull(keyType, "keyType is null");
        this.neighbourType = RowType.anonymous(List.of(keyType, DOUBLE));
        this.neighbourArrayType = new ArrayType(neighbourType);
        this.serializedType = RowType.anonymous(List.of(BIGINT, VARCHAR, neighbourArrayType));
    }

    @Override
    public Type getSerializedType()
    {
        return serializedType;
    }

    @Override
    public void serialize(KnnState state, BlockBuilder out)
    {
        KnnHeap heap = state.getHeap();
        if (heap == null) {
            out.appendNull();
            return;
        }

        ((RowBlockBuilder) out).buildEntry(fieldBuilders -> {
            BIGINT.writeLong(fieldBuilders.get(0), state.getK());
            VARCHAR.writeString(fieldBuilders.get(1), state.getMetricName());
            ((io.trino.spi.block.ArrayBlockBuilder) fieldBuilders.get(2)).buildEntry(elementBuilder -> {
                for (KnnHeap.Neighbour neighbour : heap.drainSorted()) {
                    ((RowBlockBuilder) elementBuilder).buildEntry(neighbourFields -> {
                        writeNativeValue(keyType, neighbourFields.get(0), neighbour.key());
                        DOUBLE.writeDouble(neighbourFields.get(1), neighbour.distance());
                    });
                }
            });
        });
    }

    @Override
    public void deserialize(Block block, int index, KnnState state)
    {
        SqlRow row = serializedType.getObject(block, index);
        int offset = row.getRawIndex();

        int k = (int) BIGINT.getLong(row.getRawFieldBlock(0), offset);
        String metricName = VARCHAR.getSlice(row.getRawFieldBlock(1), offset).toStringUtf8();
        state.setK(k);
        state.setMetricName(metricName);

        KnnHeap heap = state.getHeap();
        if (heap == null) {
            heap = new KnnHeap(k, dev.jaaj.trino.search.vector.Metric.fromName(metricName).higherIsCloser());
            state.setHeap(heap);
        }

        Block neighbours = neighbourArrayType.getObject(row.getRawFieldBlock(2), offset);
        for (int i = 0; i < neighbours.getPositionCount(); i++) {
            SqlRow neighbour = neighbourType.getObject(neighbours, i);
            int neighbourOffset = neighbour.getRawIndex();
            Object key = readNativeValue(keyType, neighbour.getRawFieldBlock(0), neighbourOffset);
            double distance = DOUBLE.getDouble(neighbour.getRawFieldBlock(1), neighbourOffset);
            heap.add(key, distance);
        }
    }
}
```

Remplacer les noms qualifiés en ligne par de vrais imports (`io.trino.spi.block.ArrayBlockBuilder`,
`dev.jaaj.trino.search.vector.Metric`) : ils sont écrits ainsi ici pour rendre l'origine
explicite, pas parce que c'est le style attendu.

Les API `SqlRow` / `getRawFieldBlock` / `getRawIndex` sont celles de Trino 483. Si la
compilation échoue, lire `io.trino.spi.block.SqlRow` dans le SPI installé et adapter : c'est un
détail d'API, pas un changement de conception.

- [ ] **Step 4: Vérifier la compilation**

Run: `JAVA_HOME="..." ./mvnw test-compile`
Expected: succès. Aucun test ne s'exécute encore sur ces classes ; Task 8 les met sous tension.

- [ ] **Step 5: Commit**

```bash
git add src/main/java/dev/jaaj/trino/search/vector/knn
git commit -m "Add the KNN aggregation state

k and the metric name travel with the serialized heap because the final
aggregation only ever sees deserialized partial states, never the input rows."
```

---

### Task 8: L'agrégation `knn_agg`

**Files:**
- Create: `src/main/java/dev/jaaj/trino/search/vector/knn/KnnAggregation.java`
- Modify: `src/main/java/dev/jaaj/trino/search/SearchPlugin.java`
- Test: `src/test/java/dev/jaaj/trino/search/vector/knn/TestKnnAggregation.java`

**Interfaces:**
- Consumes: `Metric`, `VectorReader` (Tasks 2-3), `KnnHeap`, `KnnState` (Tasks 6-7).
- Produces la fonction SQL
  `knn_agg(K key, array(double) vector, array(double) query, bigint k, varchar metric) -> array(row(K, double))`
  et sa surcharge `array(real)`.

- [ ] **Step 1: Écrire les tests (ils doivent échouer)**

```java
package dev.jaaj.trino.search.vector.knn;

import dev.jaaj.trino.search.SearchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;

public class TestKnnAggregation
        extends AbstractTestQueryFramework
{
    private static final String POINTS = """
            (VALUES
                ('a', ARRAY[0.0, 0.0]),
                ('b', ARRAY[1.0, 0.0]),
                ('c', ARRAY[2.0, 0.0]),
                ('d', ARRAY[3.0, 0.0])) AS t(id, v)
            """;

    @Override
    protected QueryRunner createQueryRunner()
    {
        QueryRunner queryRunner = new StandaloneQueryRunner(testSessionBuilder().build());
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testNearestNeighboursInOrder()
    {
        assertQuery(
                "SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'), x -> x[1]) FROM " + POINTS,
                "SELECT ARRAY['a', 'b']");
    }

    @Test
    public void testDistancesAreTheRawMetricValues()
    {
        assertQuery(
                "SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'), x -> x[2]) FROM " + POINTS,
                "SELECT ARRAY[0.0, 1.0]");
    }

    @Test
    public void testDotProductRanksHigherAsCloser()
    {
        assertQuery(
                "SELECT transform(knn_agg(id, v, ARRAY[1.0, 0.0], 2, 'dot_product'), x -> x[1]) FROM " + POINTS,
                "SELECT ARRAY['d', 'c']");
    }

    @Test
    public void testGroupsAreIndependent()
    {
        assertQuery(
                """
                SELECT g, transform(knn_agg(id, v, ARRAY[0.0, 0.0], 1, 'euclidean'), x -> x[1])
                FROM (VALUES
                    (1, 'a', ARRAY[5.0, 0.0]),
                    (1, 'b', ARRAY[1.0, 0.0]),
                    (2, 'c', ARRAY[9.0, 0.0]),
                    (2, 'd', ARRAY[7.0, 0.0])) AS t(g, id, v)
                GROUP BY g ORDER BY g
                """,
                "VALUES (1, ARRAY['b']), (2, ARRAY['d'])");
    }

    @Test
    public void testKLargerThanTheGroupReturnsEverything()
    {
        assertQuery(
                "SELECT cardinality(knn_agg(id, v, ARRAY[0.0, 0.0], 100, 'euclidean')) FROM " + POINTS,
                "SELECT 4");
    }

    @Test
    public void testZeroKIsRejected()
    {
        assertQueryFails(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 0, 'euclidean') FROM " + POINTS,
                ".*k must be greater than zero.*");
    }

    @Test
    public void testUnknownMetricIsRejected()
    {
        assertQueryFails(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'hamming') FROM " + POINTS,
                ".*Unknown metric 'hamming'.*");
    }

    @Test
    public void testEmptyGroupGivesNull()
    {
        assertQuery(
                "SELECT knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean') IS NULL FROM " + POINTS + " WHERE id = 'zzz'",
                "SELECT true");
    }

    @Test
    public void testNullVectorRowsAreIgnored()
    {
        assertQuery(
                """
                SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'), x -> x[1])
                FROM (VALUES
                    ('a', ARRAY[9.0, 0.0]),
                    ('b', CAST(NULL AS array(double))),
                    ('c', ARRAY[1.0, 0.0])) AS t(id, v)
                """,
                "SELECT ARRAY['c', 'a']");
    }

    @Test
    public void testNullKeysAreKept()
    {
        assertQuery(
                """
                SELECT cardinality(knn_agg(id, v, ARRAY[0.0, 0.0], 2, 'euclidean'))
                FROM (VALUES
                    (CAST(NULL AS varchar), ARRAY[1.0, 0.0]),
                    ('b', ARRAY[2.0, 0.0])) AS t(id, v)
                """,
                "SELECT 2");
    }

    @Test
    public void testBigintKeys()
    {
        assertQuery(
                """
                SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 1, 'euclidean'), x -> x[1])
                FROM (VALUES
                    (10, ARRAY[5.0, 0.0]),
                    (20, ARRAY[1.0, 0.0])) AS t(id, v)
                """,
                "SELECT ARRAY[20]");
    }

    @Test
    public void testRealOverload()
    {
        assertQuery(
                """
                SELECT transform(knn_agg(id, v, ARRAY[REAL '0.0', REAL '0.0'], 1, 'euclidean'), x -> x[1])
                FROM (VALUES
                    ('a', ARRAY[REAL '5.0', REAL '0.0']),
                    ('b', ARRAY[REAL '1.0', REAL '0.0'])) AS t(id, v)
                """,
                "SELECT ARRAY['b']");
    }
}
```

- [ ] **Step 2: Lancer les tests et vérifier l'échec**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestKnnAggregation`
Expected: FAIL, `Function 'knn_agg' not registered`.

- [ ] **Step 3: Écrire `KnnAggregation`**

```java
package dev.jaaj.trino.search.vector.knn;

import dev.jaaj.trino.search.vector.Metric;
import dev.jaaj.trino.search.vector.VectorReader;
import io.airlift.slice.Slice;
import io.trino.spi.TrinoException;
import io.trino.spi.block.ArrayBlockBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.block.BlockBuilder;
import io.trino.spi.block.RowBlockBuilder;
import io.trino.spi.block.ValueBlock;
import io.trino.spi.function.AggregationFunction;
import io.trino.spi.function.AggregationState;
import io.trino.spi.function.BlockIndex;
import io.trino.spi.function.BlockPosition;
import io.trino.spi.function.CombineFunction;
import io.trino.spi.function.Description;
import io.trino.spi.function.InputFunction;
import io.trino.spi.function.OutputFunction;
import io.trino.spi.function.SqlNullable;
import io.trino.spi.function.SqlType;
import io.trino.spi.function.TypeParameter;
import io.trino.spi.type.StandardTypes;
import io.trino.spi.type.Type;

import static dev.jaaj.trino.search.vector.VectorReader.DOUBLE_READER;
import static dev.jaaj.trino.search.vector.VectorReader.REAL_READER;
import static io.trino.spi.StandardErrorCode.INVALID_FUNCTION_ARGUMENT;
import static io.trino.spi.type.DoubleType.DOUBLE;
import static io.trino.spi.type.TypeUtils.readNativeValue;
import static io.trino.spi.type.TypeUtils.writeNativeValue;

@AggregationFunction("knn_agg")
@Description("Returns the k nearest neighbours of a query vector within each group")
public final class KnnAggregation
{
    private KnnAggregation() {}

    @InputFunction
    @TypeParameter("K")
    public static void inputDouble(
            @AggregationState("K") KnnState state,
            @TypeParameter("K") Type keyType,
            @SqlNullable @BlockPosition @SqlType("K") ValueBlock key,
            @BlockIndex int position,
            @SqlType("array(double)") Block vector,
            @SqlType("array(double)") Block queryVector,
            @SqlType(StandardTypes.BIGINT) long k,
            @SqlType(StandardTypes.VARCHAR) Slice metricName)
    {
        input(state, keyType, key, position, vector, queryVector, k, metricName, DOUBLE_READER);
    }

    @InputFunction
    @TypeParameter("K")
    public static void inputReal(
            @AggregationState("K") KnnState state,
            @TypeParameter("K") Type keyType,
            @SqlNullable @BlockPosition @SqlType("K") ValueBlock key,
            @BlockIndex int position,
            @SqlType("array(real)") Block vector,
            @SqlType("array(real)") Block queryVector,
            @SqlType(StandardTypes.BIGINT) long k,
            @SqlType(StandardTypes.VARCHAR) Slice metricName)
    {
        input(state, keyType, key, position, vector, queryVector, k, metricName, REAL_READER);
    }

    private static void input(
            KnnState state,
            Type keyType,
            ValueBlock key,
            int position,
            Block vector,
            Block queryVector,
            long k,
            Slice metricName,
            VectorReader reader)
    {
        if (k <= 0) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "k must be greater than zero, got " + k);
        }
        if (k > Integer.MAX_VALUE) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "k is too large: " + k);
        }

        Metric metric = Metric.fromName(metricName.toStringUtf8());
        KnnHeap heap = state.getHeap();
        if (heap == null) {
            heap = new KnnHeap((int) k, metric.higherIsCloser());
            state.setHeap(heap);
            state.setK((int) k);
            state.setMetricName(metricName.toStringUtf8());
        }

        if (vector.getPositionCount() != queryVector.getPositionCount()) {
            throw new TrinoException(INVALID_FUNCTION_ARGUMENT, "The arguments must have the same length");
        }
        if (vector.hasNull() || queryVector.hasNull()) {
            return;
        }

        heap.add(readNativeValue(keyType, key, position), metric.compute(vector, queryVector, reader));
    }

    @CombineFunction
    public static void combine(
            @AggregationState("K") KnnState state,
            @AggregationState("K") KnnState otherState)
    {
        KnnHeap other = otherState.getHeap();
        if (other == null) {
            return;
        }
        KnnHeap heap = state.getHeap();
        if (heap == null) {
            state.setHeap(other);
            state.setK(otherState.getK());
            state.setMetricName(otherState.getMetricName());
            return;
        }
        heap.mergeFrom(other);
    }

    @SqlNullable
    @OutputFunction("array(row(K, double))")
    public static void output(
            @TypeParameter("K") Type keyType,
            @AggregationState("K") KnnState state,
            BlockBuilder out)
    {
        KnnHeap heap = state.getHeap();
        if (heap == null || heap.size() == 0) {
            out.appendNull();
            return;
        }

        ((ArrayBlockBuilder) out).buildEntry(elementBuilder -> {
            for (KnnHeap.Neighbour neighbour : heap.drainSorted()) {
                ((RowBlockBuilder) elementBuilder).buildEntry(fieldBuilders -> {
                    writeNativeValue(keyType, fieldBuilders.get(0), neighbour.key());
                    DOUBLE.writeDouble(fieldBuilders.get(1), neighbour.distance());
                });
            }
        });
    }
}
```

Une ligne dont le vecteur contient un `NULL` est **ignorée** plutôt que de rendre tout le
groupe `NULL` : sa distance est indéfinie, donc elle n'est classable ni parmi les plus proches
ni parmi les plus lointaines, et annuler le groupe entier ferait qu'une seule ligne corrompue
détruirait un résultat par ailleurs valide.

- [ ] **Step 4: Enregistrer l'agrégation dans `SearchPlugin`**

```java
        return Set.of(
                VectorDistanceFunctions.class,
                VectorFunctions.class,
                KnnAggregation.class);
```

- [ ] **Step 5: Lancer toute la suite**

Run: `JAVA_HOME="..." ./mvnw test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src
git commit -m "Add the knn_agg aggregation

Returns the k nearest neighbours per group from a bounded heap, so memory
stays proportional to k rather than to the group size."
```

---

### Task 9: Fusion distribuée et test différentiel

Les tests précédents s'exécutent presque tous sur une partition unique, où Trino court-circuite
le cycle sérialiser / fusionner. Cette tâche est celle qui met `@CombineFunction` et
`KnnStateSerializer` sous tension.

**Files:**
- Test: `src/test/java/dev/jaaj/trino/search/vector/knn/TestKnnAggregationDistributed.java`

**Interfaces:**
- Consumes: la fonction `knn_agg` (Task 8).

- [ ] **Step 1: Écrire les tests (ils doivent échouer si l'état est incomplet)**

```java
package dev.jaaj.trino.search.vector.knn;

import com.google.common.collect.ImmutableMap;
import dev.jaaj.trino.search.SearchPlugin;
import io.trino.Session;
import io.trino.plugin.tpch.TpchPlugin;
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import io.trino.testing.StandaloneQueryRunner;
import org.junit.jupiter.api.Test;

import static io.trino.testing.TestingSession.testSessionBuilder;

public class TestKnnAggregationDistributed
        extends AbstractTestQueryFramework
{
    /**
     * A vector built from orderkey so that the nearest neighbours of the origin are the smallest
     * keys. Using a TPCH table spreads the rows across splits, which is what forces Trino to run
     * the partial and final aggregation steps and therefore to serialize the state.
     */
    private static final String VECTORS = """
            (SELECT orderkey AS id, ARRAY[CAST(orderkey AS double), 0.0] AS v FROM tpch.tiny.orders)
            """;

    @Override
    protected QueryRunner createQueryRunner()
    {
        Session session = testSessionBuilder()
                .setCatalog("tpch")
                .setSchema("tiny")
                .build();
        QueryRunner queryRunner = new StandaloneQueryRunner(session);
        queryRunner.installPlugin(new TpchPlugin());
        queryRunner.createCatalog("tpch", "tpch", ImmutableMap.of("tpch.splits-per-node", "4"));
        queryRunner.installPlugin(new SearchPlugin());
        return queryRunner;
    }

    @Test
    public void testMatchesOrderByLimitAcrossSplits()
    {
        assertQuery(
                "SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 10, 'euclidean'), x -> x[1]) FROM " + VECTORS,
                "SELECT array_agg(id ORDER BY id) FROM (SELECT orderkey AS id FROM tpch.tiny.orders ORDER BY orderkey LIMIT 10)");
    }

    @Test
    public void testDistancesMatchTheNativeFunctionAcrossSplits()
    {
        assertQuery(
                "SELECT transform(knn_agg(id, v, ARRAY[0.0, 0.0], 5, 'euclidean'), x -> x[2]) FROM " + VECTORS,
                """
                SELECT array_agg(d ORDER BY d) FROM (
                    SELECT euclidean_distance(ARRAY[CAST(orderkey AS double), 0.0], ARRAY[0.0, 0.0]) AS d
                    FROM tpch.tiny.orders ORDER BY d LIMIT 5)
                """);
    }

    @Test
    public void testGroupedAcrossSplits()
    {
        assertQuery(
                """
                SELECT orderstatus, cardinality(knn_agg(orderkey, ARRAY[CAST(orderkey AS double), 0.0], ARRAY[0.0, 0.0], 3, 'euclidean'))
                FROM tpch.tiny.orders
                GROUP BY orderstatus
                ORDER BY orderstatus
                """,
                "SELECT orderstatus, 3 FROM tpch.tiny.orders GROUP BY orderstatus ORDER BY orderstatus");
    }

    @Test
    public void testCosineMetricMatchesTheNativeFunctionAcrossSplits()
    {
        assertQuery(
                """
                SELECT transform(knn_agg(orderkey, ARRAY[CAST(orderkey AS double), 1.0], ARRAY[1.0, 1.0], 5, 'cosine'), x -> x[1])
                FROM tpch.tiny.orders
                """,
                """
                SELECT array_agg(orderkey ORDER BY d, orderkey) FROM (
                    SELECT orderkey, cosine_distance(ARRAY[CAST(orderkey AS double), 1.0], ARRAY[1.0, 1.0]) AS d
                    FROM tpch.tiny.orders ORDER BY d, orderkey LIMIT 5)
                """);
    }
}
```

Ajouter `io.trino:trino-tpch` en scope `test` dans le `pom.xml`, avec
`<version>${dep.trino.version}</version>`. Guava arrive transitivement via `trino-testing` ;
si l'import `ImmutableMap` ne résout pas, utiliser `java.util.Map.of` à la place.

Le test `testCosineMetricMatchesTheNativeFunctionAcrossSplits` départage les ex aequo par
`orderkey` dans la requête de référence. Si `knn_agg` produit un ordre différent sur des
distances égales, le test devient instable : dans ce cas, choisir un jeu de données sans ex
aequo plutôt que d'affaiblir l'assertion, l'ordre entre ex aequo n'étant pas une garantie du
contrat.

- [ ] **Step 2: Lancer les tests**

Run: `JAVA_HOME="..." ./mvnw test -Dtest=TestKnnAggregationDistributed`
Expected: PASS. Un échec de type « k vaut 0 » ou « résultat tronqué » à ce stade signale un
état sérialisé incomplet (Task 7) et non un problème de tas.

- [ ] **Step 3: Vérifier que le test attrape bien la panne visée**

Preuve que le test n'est pas décoratif : retirer temporairement `k` du type sérialisé (ou faire
retourner `1` à `getK()` après désérialisation), relancer, constater l'échec, puis rétablir.

```bash
git stash
# appliquer la régression à la main, lancer le test, constater l'échec, annuler
git checkout -- src/main/java/dev/jaaj/trino/search/vector/knn/KnnStateSerializer.java
git stash pop
```

- [ ] **Step 4: Commit**

```bash
git add src pom.xml
git commit -m "Cover the distributed aggregation path

Small local datasets never exercise serialize and combine, so these tests read
from a TPCH table split across several ranges and compare knn_agg against the
engine's own ORDER BY plus LIMIT."
```

---

### Task 10: Documentation et vérification finale

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-05-trino-search-plugin-design.md` (section 7, cas de bord)

- [ ] **Step 1: Aligner le spec sur le comportement implémenté**

Ajouter à la table des cas de bord de la section 7 la ligne décidée en Task 8 :

| `vector` contenant un élément `NULL` | ligne ignorée, comme un vecteur `NULL` |

- [ ] **Step 2: Vérifier que le README décrit les fonctions réellement livrées**

Comparer la table du README avec les `@ScalarFunction` effectivement enregistrées. Corriger le
README, pas le code.

- [ ] **Step 3: Build complet**

Run: `JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot" ./mvnw clean verify`
Expected: BUILD SUCCESS, checkstyle et modernizer inclus.

- [ ] **Step 4: Vérifier l'artefact de déploiement**

```bash
ls target/trino-search-1.0-SNAPSHOT/
```
Expected: le JAR du plugin et ses dépendances non-`provided`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs
git commit -m "Align the documentation with the shipped functions"
```

---

## Auto-revue du plan

**Couverture du spec.** Section 3 (build) → Task 1. Section 4 (architecture) → Tasks 1 à 8.
Section 5 (noyau et cas de bord) → Tasks 2 et 4. Section 6 (scalaires) → Tasks 4 et 5.
Section 7 (agrégation) → Tasks 6 à 8. Section 8 (tests) → présente dans chaque tâche, avec le
test de chargement en Task 1 et le niveau distribué en Task 9. Section 9 (environnement) →
contrainte globale. Section 10 (extensions) → hors périmètre, rien à implémenter.

**Cohérence des noms.** `VectorReader.DOUBLE_READER` / `REAL_READER` (Task 2) sont utilisés
tels quels en Tasks 3, 4, 5 et 8. `Metric.compute` / `Metric.higherIsCloser` / `Metric.fromName`
(Task 3) sont appelés en Tasks 7 et 8. `KnnHeap.Neighbour` avec ses accesseurs `key()` et
`distance()` (Task 6) est consommé en Tasks 7 et 8. `KnnState` (Task 7) est le type d'état
annoté en Task 8.

**Points où le plan peut demander une adaptation.** Deux endroits reposent sur des détails
d'API que je n'ai pas pu compiler : les noms exacts de `SqlRow` / `getRawFieldBlock` dans
`KnnStateSerializer` (Task 7), et l'acceptation par airbase 395 du `pom.xml` tel quel (Task 1).
Les deux sont des ajustements locaux, pas des remises en cause de conception, et ils sont
signalés dans les tâches concernées.
