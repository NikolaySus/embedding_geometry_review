# Межзадачные компромиссы в моделях текстовых эмбеддингов

Самостоятельный репозиторий исследования связи прикладного качества с геометрией
пространства представлений. Здесь находятся экспериментальный код, исходные
метрики, конфигурации, две рукописи и галерея вариантов рисунков.

История редакций ведётся в Git. **full** и **compact** являются двумя намеренно
сохранёнными вариантами статьи, а не последовательными версиями. Галерея одна;
её альтернативы предназначены для выбора оформления рисунков.

## Установка

```bash
cd /home/nop/Documents/m2/research/embedding_geometry_review
uv sync --locked
```

Требуется Python 3.12. Версии научных библиотек перенесены без обновления;
`uv.lock` фиксирует зависимости, в том числе PyTorch из CUDA 13.0 index.
Для документов дополнительно нужны LibreOffice (`libreoffice` или `soffice`),
Poppler (`pdftoppm`) и Chrome/Chromium для экспорта Plotly через Kaleido.
При необходимости путь к браузеру задаётся переменной `BROWSER_PATH`.
Сборка документов и галереи не запускает обучение и не требует GPU.

## Статья и галерея

```bash
uv run python src/build_article.py --variant full
uv run python src/build_article.py --variant compact
uv run python src/build_gallery.py
uv run python src/validate_publication.py --all
```

| Материал | Редактируемый источник | Результаты сборки |
|---|---|---|
| Полная статья | `manuscript/full.md` | `build/full/` |
| Компактная статья | `manuscript/compact.md` | `build/compact/` |
| Галерея и интерактивные панели | `src/gallery/`, `assets/gallery/` | `build/gallery/` |

В Git входят исходники, числовые таблицы и выбранные настройки визуализаций.
Рисунки, интерактивный HTML, PDF, DOCX и рендеры страниц являются локальными
результатами сборки и игнорируются. Не храните вручную выбранный ракурс только
в `build/`: сохраните его в исходных настройках галереи согласно её инструкции.

Новая редакция изменяет текущие файлы. Дополнительную параллельную рукопись или
галерею следует создавать только по явному решению автора.

## Экспериментальные данные

| Каталог | Содержание | Хранение |
|---|---|---|
| `src/embedding_geometry/` | Обучение, оценка, геометрические диагностики | Git |
| `configs/` | Конфигурации проведённых экспериментов | Git |
| `runs/` | Первичные метрики, журналы обучения, состояния и манифесты | Git, кроме тяжёлых артефактов |
| `reports/` | Сводная статистика, аудиты и исторические пояснения | Git |
| `references/` | Библиография и литературная матрица | Git |
| `assets/` | Числовые данные и настройки публикационных материалов | Git |
| `data/processed/` | Подготовленные обучающие и диагностические корпуса | Локально; манифест в Git |
| `runs/**/final_model/` | Сохранённые модели | Только локально |
| `.venv/`, `build/`, кэши | Окружение и производные файлы | Только локально |

Идентификаторы `english_geometry_pilot_v1` и `english_geometry_pilot_v2_mixtures`
не переименовываются: это идентификаторы опытов. Сохранены все исходные результаты,
в том числе исторические ветви, не включённые в текущие рукописи. Наличие результата
в архиве не означает его включения в доказательную базу статьи; используйте
существующие аудиты и критерии отбора наблюдений.

Некоторые веса смесей удалялись исходным пайплайном после оценки. Перенос сохраняет
только действительно имеющиеся веса и не восстанавливает отсутствующие checkpoints.
Свежий Git clone содержит всё необходимое для сборки статьи из сохранённой
статистики, но не содержит корпуса и веса для повторного обучения.

## Запуски исследования

Команды ниже приведены для будущих запусков, **не как обязательный этап переноса**.
Они могут загружать данные, занимать GPU и изменять файлы результатов.

```bash
uv run python src/experiment.py --help
uv run python src/experiment.py prepare-data
uv run python src/experiment.py audit
uv run python src/experiment.py audit-evaluation
uv run python src/experiment.py run --phase smoke
uv run python src/experiment.py run --phase seed1
uv run python src/experiment.py evaluate --suite gate --seeds 42
uv run python src/experiment.py run --phase seeds3
uv run python src/experiment.py evaluate --suite full
uv run python src/experiment.py evaluate-clean-sts
uv run python src/experiment.py report
```

Смешивание целей:

```bash
uv run python src/experiment.py --config configs/pilot_v2_mixtures.json mixtures --phase smoke
uv run python src/experiment.py --config configs/pilot_v2_mixtures.json mixtures --phase full
uv run python src/experiment.py --config configs/pilot_v2_mixtures.json mixtures --phase report
```

Для нового исследовательского опыта задавайте новый `experiment_id`, чтобы не
переписывать завершённые наблюдения. Это не относится к редакциям документов,
история которых ведётся Git.

## Проверки и происхождение

```bash
uv run pytest
uv run python scripts/verify_migration.py
# Дополнительно проверить локальные веса, корпуса и массивы:
uv run python scripts/verify_migration.py --include-local
uv run python scripts/check_source_checkout.py
git status --short
git add --dry-run .
```

Проверка переноса сверяет SHA-256 с исходным снимком, не требует доступа к старому
репозиторию и не изменяет результаты. После осознанного изменения исследовательского
кода она будет сообщать отклонение от исходного снимка; не перезаписывайте этот
снимок для сокрытия изменений.

`check_source_checkout.py` копирует только доступные Git файлы в служебный
каталог `build/source-checkout-check/` и собирает там обе статьи и галерею с
отключённым доступом к Hugging Face и скрытым GPU. Веса, корпуса и старые
выходы сборки в эту проверку не попадают. Каталог проверки пересоздаётся при
каждом запуске; не храните в нём ручные изменения.

Подробности: [правила работы](AGENTS.md), [происхождение и перенос](docs/MIGRATION.md),
[результаты проверок](docs/VALIDATION.md).
Старые абсолютные пути в исторических метаданных являются свидетельством
происхождения, а не зависимостью сборки от прежнего каталога.
