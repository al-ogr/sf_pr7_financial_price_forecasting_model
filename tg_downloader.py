import asyncio
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from telethon import TelegramClient

# Необязательная полоска прогресса (tqdm). Если пакет не установлен — используем no-op совместимый класс.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class _DummyTqdm:
        def __init__(self, *args, total=None, **kwargs) -> None:
            self.total = int(total or 0)
            self.n = 0

        def update(self, n: int = 1) -> None:
            self.n += int(n)

        def set_postfix(self, **kwargs) -> None:  # noqa: D401
            return None

        def close(self) -> None:
            return None

    def tqdm(*args, **kwargs):  # type: ignore
        return _DummyTqdm(*args, **kwargs)

# Цветной вывод: сначала пытаемся подключить colorama, иначе используем ANSI-коды (macOS/Linux)
try:  # pragma: no cover
    from colorama import Fore, Style, init as colorama_init  # type: ignore

    colorama_init()  # enable colors on Windows; harmless on Unix

    def color_text(text: str, color: str) -> str:
        return f"{color}{text}{Style.RESET_ALL}"

    COLOR_INFO = Fore.CYAN
    COLOR_OK = Fore.GREEN
    COLOR_WARN = Fore.YELLOW
    COLOR_ERR = Fore.RED
except Exception:  # pragma: no cover
    ANSI_RESET = "\033[0m"
    ANSI_CYAN = "\033[36m"
    ANSI_GREEN = "\033[32m"
    ANSI_YELLOW = "\033[33m"
    ANSI_RED = "\033[31m"

    def color_text(text: str, color: str) -> str:
        # color here should be an ANSI code; append reset
        return f"{color}{text}{ANSI_RESET}"

    COLOR_INFO = ANSI_CYAN
    COLOR_OK = ANSI_GREEN
    COLOR_WARN = ANSI_YELLOW
    COLOR_ERR = ANSI_RED

def parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)

async def main():
    #-----------------------------------
    dotenv_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    # Подгрузка данных для подключения из файла .env
    api_id   = os.environ["TLG_API_ID"]
    api_hash = os.environ["TLG_API_HASH"]
    username = os.environ["TLG_USER_NAME"]
    #-----------------------------------
    ap = argparse.ArgumentParser(description="tg_downloader: экспорт постов из публичных Telegram-каналов за период")
    ap.add_argument("channel", nargs="?", default=None,
                    help="Публичный канал: @channel | channel | https://t.me/channel (если не указан — используем --channels-file)")
    ap.add_argument("--channels-file", default=None,
                    help="Путь к файлу со списком каналов (по одному в строке; поддерживаются @name, name, https://t.me/name)")
    ap.add_argument("--date-from", required=True, help="Начало периода (YYYY-MM-DD)")
    ap.add_argument("--date-to", required=True, help="Конец периода (YYYY-MM-DD)")
    # ap.add_argument("--api-id", type=int, default=os.getenv("TG_API_ID"), help="Telegram api_id (или TG_API_ID в env)")
    # ap.add_argument("--api-hash", default=os.getenv("TG_API_HASH"), help="Telegram api_hash (или TG_API_HASH в env)")
    ap.add_argument("--api-id", type=int, default=api_id, help="Telegram api_id (или TG_API_ID в env)")
    ap.add_argument("--api-hash", default=api_hash, help="Telegram api_hash (или TG_API_HASH в env)")
    ap.add_argument("--out-dir", default=".", help="Базовый каталог для выгрузки (будет создан подкаталог периода)")
    args = ap.parse_args()

    if not args.api_id or not args.api_hash:
        raise SystemExit("Нужны api_id и api_hash (аргументы или переменные окружения TG_API_ID, TG_API_HASH)")

    def normalize_channel(raw: str) -> str:
        ch = (raw or "").strip()
        if not ch:
            return ch
        if ch.startswith("https://t.me/"):
            ch = ch.split("https://t.me/")[-1].strip("/")
        if ch.startswith("@"):
            ch = ch[1:]
        return ch

    def sanitize_for_filename(s: str) -> str:
        # Разрешим буквы, цифры, дефис и подчёркивание; остальное заменим на '_'
        return re.sub(r"[^0-9A-Za-z_\-]", "_", s)

    # Список каналов: либо из аргумента, либо из файла
    channels: list[str] = []
    if args.channel and args.channels_file:
        raise SystemExit("Укажите либо одиночный канал, либо --channels-file, но не оба сразу")
    if not args.channel and not args.channels_file:
        raise SystemExit("Нужно указать канал или --channels-file")
    if args.channel:
        channels = [normalize_channel(args.channel)]
    else:
        file_path = args.channels_file
        if not os.path.isfile(file_path):
            raise SystemExit(f"Файл не найден: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                channels.append(normalize_channel(line))
        if not channels:
            raise SystemExit("В файле каналов нет валидных записей")

    date_from = parse_date(args.date_from)
    date_to = parse_date(args.date_to)
    if date_to < date_from:
        raise SystemExit("date-to не может быть меньше date-from")

    # Источник (для имени каталога): файл каналов либо одиночный канал
    if args.channels_file:
        src_base = os.path.splitext(os.path.basename(args.channels_file))[0]
        source_label = re.sub(r"[^0-9A-Za-z_\-]", "_", src_base)
    else:
        # одиночный канал
        single_ch = sanitize_for_filename(normalize_channel(args.channel))
        source_label = single_ch or "channel"

    # Директория периода
    period_from = date_from.strftime("%Y-%m-%d")
    period_to = date_to.strftime("%Y-%m-%d")
    period_dirname = f"{source_label}_posts_{period_from}_{period_to}"
    output_dir = os.path.join(args.out_dir, period_dirname)
    os.makedirs(output_dir, exist_ok=True)

    # Берём на день больше для корректной выборки верхней границы
    offset_date = date_to + timedelta(days=1)

    client = TelegramClient("tg_downloader_session", args.api_id, args.api_hash)
    await client.start()

    # Глобальный прогресс по списку каналов
    channels_pbar = tqdm(total=len(channels), desc="Каналы", unit="кан", dynamic_ncols=True, leave=True)

    for raw_channel in channels:
        ch = normalize_channel(raw_channel)
        if not ch:
            tqdm.write(color_text("[пропуск] Пустое имя канала", COLOR_WARN))
            channels_pbar.update(1)
            continue

        try:
            entity = await client.get_entity(ch)
        except Exception as e:
            tqdm.write(color_text(f"[ошибка] Не удалось получить канал '{raw_channel}': {e}", COLOR_ERR))
            channels_pbar.update(1)
            continue

        chan_username = getattr(entity, "username", None)

        out: list[dict] = []
        processed_msgs = 0
        tqdm.write(color_text(f"→ Обработка канала: {ch}", COLOR_INFO))
        async for msg in client.iter_messages(entity, offset_date=offset_date):
            # iter_messages по умолчанию идёт от новых к старым
            if msg.date is None:
                continue
            if msg.date < date_from:
                break
            # Верхняя граница обеспечена offset_date

            link = None
            if chan_username:
                link = f"https://t.me/{chan_username}/{msg.id}"

            out.append({
                "id": msg.id,
                "date_utc": msg.date.astimezone(timezone.utc).isoformat(),
                "message": msg.message or "",
                "views": getattr(msg, "views", None),
                "forwards": getattr(msg, "forwards", None),
                "replies": getattr(getattr(msg, "replies", None), "replies", None),
                "has_media": msg.media is not None,
                "link": link,
            })
            processed_msgs += 1
            if processed_msgs % 100 == 0:
                tqdm.write(color_text(f"   … извлечено {processed_msgs} сообщений из {ch}", COLOR_INFO))

        # Если сообщений нет — файл не создаём
        if len(out) == 0:
            tqdm.write(color_text(f"[пропуск] Нет сообщений у {ch} за {period_from}..{period_to}", COLOR_WARN))
            channels_pbar.update(1)
            continue

        # Имя файла
        safe_name = sanitize_for_filename(ch)
        file_name = f"posts_{safe_name}_{period_from}_{period_to}.json"
        file_path = os.path.join(output_dir, file_name)

        data = json.dumps(out, ensure_ascii=False, indent=2)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data)
        tqdm.write(color_text(f"[успех] Сохранено {len(out)} сообщений → {file_path}", COLOR_OK))
        channels_pbar.set_postfix_str(f"последний={ch}")
        channels_pbar.update(1)

    channels_pbar.close()

if __name__ == "__main__":
    asyncio.run(main())