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
- `summary.json`
- `final_confusion_matrix.csv`
- `final_model.pt`

Агрегировать summary-файлы (собрать нессколько summary.json в сводные таблицы):

```powershell
python scripts\aggregate_results.py --results-dir results
```

Построить графики:

```powershell
python scripts\plot_results.py --results-dir results
```

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
