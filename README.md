## Запуск

```powershell
python -m compileall src scripts
pytest
ruff check .
mypy src
```

### FedAvg на MNIST

```powershell
python scripts\run_experiment.py --config configs\mnist_fedavg.toml
```

### FedAvgM на MNIST

```powershell
python scripts\run_experiment.py --config configs\mnist_fedavgm.toml
```

### FedProx на MNIST

```powershell
python scripts\run_experiment.py --config configs\mnist_fedprox.toml
```

### smoke-тест

```powershell
python scripts\run_experiment.py --config configs\mnist_smoke.toml --synthetic-smoke
```

### Ограниченный запуск на реальном датасете

```powershell
python scripts\run_experiment.py --config configs\mnist_fedavg.toml --train-limit 1000 --test-limit 200
```

- `--train-limit 1000` первые 1000 примеров исходной обучающей выборки
- `--test-limit 200` первые 200 примеров тестовой выборки;

## Разбиения данных

- `iid`
- `dirichlet`
- `pathological`
- `quantity`

## Инспекция разбиения

```powershell
python scripts\inspect_partitions.py --config configs\mnist_fedavg.toml
```

Команда создаёт `partition_statistics.csv`, `partition_matrix.csv`

## Серия экспериментов

Показать первые комбинации grid без запуска:

```powershell
python scripts\run_grid.py --config configs\experiment_grid.toml --max-experiments 3 --dry-run
```

Запустить ограниченную серию:

```powershell
python scripts\run_grid.py --config configs\experiment_grid.toml --max-experiments 3
```

grid-конфигурация содержит только MNIST

## Результаты и графики

По умолчанию каждый запуск создаёт каталог `results\<experiment_id>\`:

- `config.toml` — использованная конфигурация
- `metadata.json` — сведения об окружении и статус запуска
- `round_metrics.csv` — глобальные метрики по раундам
- `client_metrics.csv` — метрики локального обучения клиентов
- `partition_statistics.csv`, `partition_matrix.csv` — статистика разбиения
- `attack_manifest.json` — параметры data-poisoning-атаки и ground truth симулятора
- `summary.json`
- `final_confusion_matrix.csv`
- `final_per_class_metrics.csv` — accuracy, cross-entropy loss и число тестовых примеров по истинным классам
- `final_model.pt`

Агрегировать summary-файлы (собрать нессколько summary.json в сводные таблицы):

```powershell
python scripts\aggregate_results.py --results-dir results
```

Построить графики:

```powershell
python scripts\plot_results.py --results-dir results
```

## Label-flipping data poisoning

Конфигурация `configs\mnist_label_flip.toml` запускает воспроизводимую атаку на данные: 20% клиентов выбираются по seed, а у 50% доступных им примеров класса `1` метка изменяется на `7`. Исходный датасет не меняется: атака применяется только в локальной обёртке выборки вредоносного клиента.

```powershell
python scripts\run_experiment.py --config configs\mnist_label_flip.toml
```

Файл `attack_manifest.json` содержит список выбранных клиентов и индексов отравлённых примеров. Он является ground truth экспериментальной симуляции и не моделирует сведения, доступные серверу в реальной федеративной системе.

### Два независимых label flip

Конфигурация `configs\mnist_multi_label_flip.toml` задаёт две одновременные непересекающиеся замены меток: `1 -> 7` и `2 -> 8`. Для каждого выбранного вредоносного клиента доля `poison_fraction` выбирается отдельно среди доступных примеров каждого исходного класса.

```powershell
python scripts\run_experiment.py --config configs\mnist_multi_label_flip.toml
```

Для типа `multi_label_flip` отображения задаются в TOML так: `label_flips = [[source_1, target_1], [source_2, target_2]]`. Стенд проверяет, что классы отображений не пересекаются, поэтому один пример не может быть отравлен двумя правилами. В `attack_manifest.json` сохраняется число реально изменённых меток для каждого отображения.

### Усиленный label flip (stress test)

`configs\mnist_strong_label_flip.toml` задаёт намеренно жёсткий сценарий: 80% клиентов изменяют все доступные им метки `1 -> 7`. Он предназначен для демонстрации предельного эффекта data poisoning и не должен интерпретироваться как реалистичная распространённость вредоносных клиентов.

## Конфигурация

Эксперимент описывается TOML-файлом с разделами:

- `[dataset]` — датасет, каталог данных и загрузка
- `[partition]` — тип разбиения, число клиентов, `alpha`, seed
- `[model]` — модель и число классов
- `[training]` — batch size, локальные эпохи, SGD-параметры, устройство
- `[federation]` — число раундов и доли участников
- `[strategy]` — FedAvg/FedAvgM/FedProx и серверные параметры
- `[tracking]` — каталог результатов и сохранение модели

## Расширение стенда

Новый датасет добавляется в `src/fl_benchmark/datasets/loader.py` и должен возвращать `DatasetBundle` с `train`, `test` и `num_classes`. 

Новый тип модели регистрируется в `src/fl_benchmark/models/factory.py`. 

Новая стратегия добавляется в `src/fl_benchmark/strategies/` и регистрируется в `strategies/factory.py` и типе `StrategyName` в `config.py`.
