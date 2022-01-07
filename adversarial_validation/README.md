# Adversarial Validation

<img src="https://user-images.githubusercontent.com/39204075/148427390-368bdb57-8374-4842-9528-12fa35288b8a.png" alt="drawing" width="500"/>

## Постановка задачи
Пусть дана исходная задача классификации (регрессии), которая включает в себя тренировочную и тестовые выборки. 

Наша задача заключается в том, чтобы проверить, 
насколько совпадают распределения на объектах, используемых для обучения и теста. 

## Алгоритм

Алгоритм проверки в следующем:
* Создадим новый бинарный признак, равный 0 на тестовой выборке и 1 на обучающей
* Сделаем этот признак целевым
* Попробуем решить полученную задачу

Если это удалось сделать хорошо (например __ROC AUC__ значительно отличается от 0.5), то эти выборки не имеют разные распределения.
При этом виновные в этом признаки можно определить посмотрев на feature imiportance.

## Ноутбуки
Ноутбуки имеют следующую структуру:
* [\[0\]adversarial_validation](https://github.com/AlexGrunt/MLValidation/blob/main/adversarial_validation/%5B0%5Dadversarial_validation.ipynb) - простой пример применения описанного подхода. Показано, что тестовая и обучающая California Housing Dataset
неразличимы моделью CatBoostClassifier. 

## Литература

* [Adversarial Validation Approach to Concept Drift Problem in User Targeting Automation Systems at Uber](https://arxiv.org/pdf/2004.03045.pdf)
* [How to Assess Similarity between Two Datasets? — Adversarial Validation](https://towardsdatascience.com/how-to-assess-similarity-between-two-datasets-adversarial-validation-246710eba387) - здесь можно найти имплементацию с LightGBM
