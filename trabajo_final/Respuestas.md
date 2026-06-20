# Consigna QDA

**Notación**: en general notamos

* $k$ la cantidad de clases
* $n$ la cantidad de observaciones
* $p$ la cantidad de features/variables/predictores

**Sugerencia:** combinaciones adecuadas de `transpose`, `stack`, `reshape` y, ocasionalmente, `flatten` y `diagonal` suele ser más que suficiente. Se recomienda *fuertemente* explorar la dimensionalidad de cada elemento antes de implementar las clases.

# Respuestas

Para todas las respuestas vamos a respetar la notación propuesta. El dataset que utilizaremos será el de vinos, via la función `get_wine_dataset()`. El mismo tiene la siguiente distribución:
- $k$ clases: 3
- $p$ features: 13
- $n$ observaciones: 178

## Tensorización

En esta sección nos vamos a ocupar de hacer que el modelo sea más rápido para generar predicciones, observando que incurre en un doble `for` dado que predice en forma individual un escalar para cada observación, para cada clase. Paralelizar ambos vía tensorización suena como una gran vía de mejora de tiempos.

### 1) Diferencias entre `QDA`y `TensorizedQDA`

1. ¿Sobre qué paraleliza `TensorizedQDA`? ¿Sobre las $k$ clases, las $n$ observaciones a predecir, o ambas?

TensorizedQDA paraleliza sobre las k clases. Ambos recorren las observaciones de la misma forma. Pero una vez que se obtienen las matrices de covarianzas, las apila para todas las clases a la vez usando `np.stack`
Por ejemplo, usando el dataset de vinos, queda (3, 13, 13) donde:

- 3 es por las k clases
- 13, 13 es por la matriz de covarianza de los features


En conclusión, en `QDA` las $k$ matrices de covarianza viven separadas en una lista, entonces no queda otra que visitarlas una por una. `TensorizedQDA` las "junta" en un único objeto 3D, y eso le permite a numpy hacer todos los productos en paralelo en lugar de un loop.

2. Analizar los shapes de `tensor_inv_covs` y `tensor_means` y explicar paso a paso cómo es que `TensorizedQDA` llega a predecir lo mismo que `QDA`.

`tensor_inv_cov` tiene shape $(k, p, p)$ y `tensor_means` shape $(k, p, 1)$, para el caso del dataset de vinos los shapes son:

```py
self.tensor_inv_cov.shape
(3, 13, 13)
self.tensor_means.shape
(3, 13, 1)
```
A la hora de calcular `_predict_log_conditionals` toma cada observación $x$ con sus features y le resta los sesgos tensorizados antes de hacer el producto matricial con la matrices de covarianza tensorizadas. Por broadcasting, el tensor resuelve en paralelo el mismo vector de observación para las $k$ categorías. Por lo tanto, evita tener que predecir en loop. El producto matricial termina siendo de `(3,1,1)`: un escalar para cada clase.

### 2) Optimización

Debido a la forma cuadrática de QDA, no se puede predecir para $n$ observaciones en una sola pasada (utilizar $X \in \mathbb{R}^{p \times n}$ en vez de $x \in \mathbb{R}^p$) sin pasar por una matriz de $n \times n$ en donde se computan todas las interacciones entre observaciones. Se puede acceder al resultado recuperando sólo la diagonal de dicha matriz, pero resulta ineficiente en tiempo y (especialmente) en memoria. Aún así, es *posible* que el modelo funcione más rápido.

> Para esta sección usamos el dataset de cartas dado que, por su magnitud, era el único donde podíamos observar diferencias:
- $k$ clases: 26
- $p$ features: 16
- $n$ observaciones: 20000

3. Implementar el modelo `FasterQDA` (se recomienda heredarlo de `TensorizedQDA`) de manera de eliminar el ciclo for en el método predict.

```py
class FasterQDA(TensorizedQDA):
    def predict(self, X):
            unbiased_X = X[np.newaxis, :, :] - self.tensor_means
            temp1 = unbiased_X.transpose(0, 2, 1) @ self.tensor_inv_cov
            
            full_matrix = temp1 @ unbiased_X
            
            # np.diagonal extrae la diagonal con axis1 y axis2
            quadratic = np.diagonal(full_matrix, axis1=1, axis2=2)
            
            log_det = 0.5 * np.log(LA.det(self.tensor_inv_cov))[:, np.newaxis]
            log_post = self.log_a_priori[:, np.newaxis] + log_det - 0.5 * quadratic
            
            return np.argmax(log_post, axis=0).reshape(1, -1)
```

4. Mostrar dónde aparece la mencionada matriz de $n \times n$, donde $n$ es la cantidad de observaciones a predecir.

La matriz de $n\times n$ es `full_matrix = temp1 @ unbiased_X`

5. Demostrar que
$$
diag(A \cdot B) = \sum_{cols} A \odot B^T = np.sum(A \odot B^T, axis=1)
$$ es decir, que se puede "esquivar" la matriz de $n \times n$ usando matrices de $n \times p$. También se puede usar, de forma equivalente,
$$
np.sum(A^T \odot B, axis=0).T
$$
queda a preferencia del alumno cuál usar.

---
Dadas dos matrices de 2x2:

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \quad B = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$$

*Lado izquierdo:* $diag(A \cdot B)$

$$A \cdot B = \begin{pmatrix} 1\cdot5+2\cdot7 & \cdots \\ \cdots & 3\cdot6+4\cdot8 \end{pmatrix} = \begin{pmatrix} 19 & \cdots \\ \cdots & 50 \end{pmatrix}$$

$$diag(A \cdot B) = \begin{pmatrix} 19 \\ 50 \end{pmatrix}$$

*Lado derecho:* $\sum_{cols} A \odot B^T$

$$B^T = \begin{pmatrix} 5 & 7 \\ 6 & 8 \end{pmatrix}$$

$$A \odot B^T = \begin{pmatrix} 1\cdot5 & 2\cdot7 \\ 3\cdot6 & 4\cdot8 \end{pmatrix} = \begin{pmatrix} 5 & 14 \\ 18 & 32 \end{pmatrix}$$

$$\sum_{cols} = \begin{pmatrix} 5+14 \\ 18+32 \end{pmatrix} = \begin{pmatrix} 19 \\ 50 \end{pmatrix}$$
---

6. Utilizar la propiedad antes demostrada para reimplementar la predicción del modelo `FasterQDA` de forma eficiente en un nuevo modelo `EfficientQDA`.

```py
class EfficientQDA(TensorizedQDA):
    def predict(self, X):
        unbiased_X = X - self.tensor_means 
        temp = self.tensor_inv_cov @ unbiased_X 
        quadratic_terms = (unbiased_X * temp).sum(axis=1) 
        
        log_det = 0.5 * np.log(LA.det(self.tensor_inv_cov))[:, np.newaxis]
        log_post = self.log_a_priori[:, np.newaxis] + log_det - 0.5 * quadratic_terms
        
        return np.argmax(log_post, axis=0).reshape(1, -1)
```
7. Comparar la performance de las 4 variantes de QDA implementadas hasta ahora (no Cholesky) ¿Qué se observa? A modo de opinión ¿Se condice con lo esperado?

![results](results_benchmark_1.png)

La mejora del `TensorizedQDA` respecto a `QDA` es clara. Sin embargo, resulta que el `FasterQDA` es en efecto mucho más pesado en memoria pero también lo es en tiempo. El `EfficientQDA` es mucho más eficiente en ambos casos. Se condice con lo esperado porque FasterQDA a pesar de  eliminar el ciclo for se sigue construyendo la matriz n×n, generando un cuello de botella en memoria que impacta negativamente en el tiempo de ejecución.

## Cholesky

Hasta ahora todos los esfuerzos fueron enfocados en realizar una predicción más rápida. Los tiempos de entrenamiento (teóricos al menos) siguen siendo los mismos o hasta (minúsculamente) peores, dado que todas las mejoras siguen llamando al método `_fit_params` original de `QDA`.

La descomposición/factorización de [Cholesky](https://en.wikipedia.org/wiki/Cholesky_decomposition#Statement) permite factorizar una matriz definida positiva $A = LL^T$ donde $L$ es una matriz triangular inferior. En particular, si bien se asume que $p \ll n$, invertir la matriz de covarianzas $\Sigma$ para cada clase impone un cuello de botella que podría alivianarse. Teniendo en cuenta que las matrices de covarianza son simétricas y salvo degeneración, definidas positivas, Cholesky como mínimo debería permitir invertir la matriz más rápido.

*Nota: observar que calcular* $A^{-1}b$ *equivale a resolver el sistema* $Ax=b$.

### 3) Diferencias entre implementaciones de `QDA_Chol`

8. Si una matriz $A$ tiene fact. de Cholesky $A=LL^T$, expresar $A^{-1}$ en términos de $L$. ¿Cómo podría esto ser útil en la forma cuadrática de QDA?

$A^{-1}$ en términos de $L$ se puede expresar como:

$$A^{-1} = (LL^T)^{-1} = (L^T)^{-1}L^{-1} = (L^{-1})^T L^{-1}$$

Ya que la inversa se distribuye en el producto entre dos matrices. Como $L$ es triangular inferior, su inversa también lo es. Entonces la inversa de $A$ termina siendo el producto entre dos matrices triangulares inferiores, de la cual una es la traspuesta de otra.

Esto sería útil a la hora de calcular la inversa de la matriz de covarianzas porque invertir una matriz triangular es más barato que invertir una matriz no triangular de $k \times k$. Y, a su vez, calcular la traspuesta de una matriz es barato.

9. Explicar las diferencias entre `QDA_Chol1`y `QDA` y cómo `QDA_Chol1` llega, paso a paso, hasta las predicciones.

Primero que nada, `QDA_Chol1` trabaja con las trianguladas de las matrices de covarianza. Por lo tanto, el entrenamiento es más eficiente. Una vez que obtiene el array de las inversas de las matrices de covarianza para cada clase y le resta los sesgos al vector entrada $x$, obtiene una matriz $L^{-1}$ de $p \times p$ y hace el producto matricial con el vector $x$ de tamaño $p \times 1$. Lo que da como resultado un vector de predicciones $y$ de $p \times 1$.

Una vez que calcula eso, resuelve el determinante multiplicando la diagonal de la matriz triangulada. También, como $y$ es un vector, hace el cuadrado del mismo y lo suma. Lo que es equivalente a hacer $y^T * y$ elemento a elemento. De esta manera, simplifica el cálculo de la probabilidad a posteriori

10. ¿Cuáles son las diferencias entre `QDA_Chol1`, `QDA_Chol2` y `QDA_Chol3`?
Las diferencias están en cómo implementan el entrenamiento. `QDA_Chol1` junta el array de las inversas de las matrices de covarianzas factorizadas con Cholesky de forma genérica mientras que `QDA_Chol3` utiliza un algoritmo específico para invertir matrices triangulares. Por otro lado, `QDA_Chol2`, al entrenar, se queda con las matrices factorizadas y no las invierte. Para predecir, resuelve el sistema $Lx = y$ a través de `solve_triangular`, dado que el sistema es triangular. En consecuencia, al no tener el determinante de una matriz inversa que toma la forma $det(L^{-1}) = det(L)^{-1}$, debe multiplicar el logaritmo por $-1$ para conservar el signo y obtener resultado equivalente a las otras implementaciones

11. Comparar la performance de las 7 variantes de QDA implementadas hasta ahora ¿Qué se observa?¿Hay alguna de las implementaciones de `QDA_Chol` que sea claramente mejor que las demás?¿Alguna que sea peor?

![results](results_benchmark_2.png)

| model         |   train_median_ms |   train_std_ms |   test_median_ms |   test_std_ms |   mean_accuracy |   train_mem_median_mb |   train_mem_std_mb |   test_mem_median_mb |   test_mem_std_mb |   train_speedup |   test_speedup |   train_mem_reduction |   test_mem_reduction |
|:--------------|------------------:|---------------:|-----------------:|--------------:|----------------:|----------------------:|-------------------:|---------------------:|------------------:|----------------:|---------------:|----------------------:|---------------------:|
| QDA           |           5.3709  |       0.671311 |        1853.09   |      28.0497  |        0.88427  |              0.251595 |         0.00219294 |             0.116177 |       0.0701678   |        1        |       1        |               1       |          1           |
| TensorizedQDA |           5.29225 |       0.606143 |         324.814  |       8.42425 |        0.884498 |              0.250076 |         0.00226286 |             0.170273 |       0.000215291 |        1.01486  |       5.70509  |               1.00607 |          0.682297    |
| FasterQDA     |           6.19625 |       2.85118  |        2521.82   |     145.735   |        0.885152 |              0.249771 |         0.00333285 |          7179.32     |       0.042637    |        0.866798 |       0.734824 |               1.0073  |          1.61821e-05 |
| EfficientQDA  |           5.2565  |       0.808637 |          23.7725 |       2.04119 |        0.88472  |              0.249283 |         0.0023721  |            58.4361   |       0.0002427   |        1.02176  |      77.9508   |               1.00927 |          0.0019881   |
| QDA_Chol1     |           6.2964  |       1.25168  |        1264.37   |      73.38    |        0.885082 |              0.250926 |         0.00218775 |             0.111112 |       0.0564661   |        0.853011 |       1.46562  |               1.00266 |          1.04558     |
| QDA_Chol2     |           5.7079  |       1.06278  |        3448.08   |     248.033   |        0.88546  |              0.249499 |         0.00219015 |             0.252956 |       0.0424177   |        0.940959 |       0.537427 |               1.0084  |          0.459275    |
| QDA_Chol3     |           5.3528  |       0.675186 |        1060.15   |      18.9203  |        0.884858 |              0.250573 |         0.00195457 |             0.111068 |       0.0564986   |        1.00338  |       1.74796  |               1.00408 |          1.046       |

La mejor sigue siendo el `EfficientQDA`. Las `QDA_Chol` optimizan el entrenamiento pero pagan el costo en cada predict, lo que escala con el tamaño del dataset. `QDA_Chol2` resulta ser la peor, incluso peor que QDA original, lo cual también es esperable ya que mueve el costo de invertir al predict. De las tres, `QDA_Chol3` es la que mejor funciona porque el `lapack.dtrtri` aprovecha la triangularidad.

### 4) Optimización

12. Implementar el modelo `TensorizedChol` paralelizando sobre clases/observaciones según corresponda. Se recomienda heredarlo de alguna de las implementaciones de `QDA_Chol`, aunque la elección de cuál de ellas queda a cargo del alumno según lo observado en los benchmarks de puntos anteriores.
13. Implementar el modelo `EfficientChol` combinando los insights de `EfficientQDA` y `TensorizedChol`. Si se desea, se puede implementar `FasterChol` como ayuda, pero no se contempla para el punto.
13. Comparar la performance de las 9 variantes de QDA implementadas ¿Qué se observa? A modo de opinión ¿Se condice con lo esperado?