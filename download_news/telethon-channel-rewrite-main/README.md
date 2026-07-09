# telethon-channel-rewrite

Простой скрипт на Python, который при помощи Telethon достает текстовые сообщения и сообщения с одним изображением. Поддержка
остальных форматов сообщений ведется опционально.

[Написан для статьи](https://mrtstg.ru/posts/telethon-channel-rewrite?source=telethon-channel-rewrite-repo)

## Запуск

Если есть poetry:

```bash
mkdir -p images
poetry install --no-root
# отредактируйте скрипт до этого
poetry run python src/main.py
```

Если нет:

```bash
mkdir -p images
pip install telethon==1.38.0
python src/main.py
```
