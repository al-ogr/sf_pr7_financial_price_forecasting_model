from telethon import TelegramClient
from telethon.helpers import TotalList
import asyncio
import json
from telethon.types import Message, Photo, MessageMediaPhoto
from telethon.tl.types import PeerChannel
import logging

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

logging.basicConfig(level=logging.INFO)

app = TelegramClient(
    session="userbot",
    api_id=0,  # INSERT API_ID
    api_hash="YOUR_HASH",
)

CHANNEL_ID = 0  # YOUR CHANNEL
payload: list[dict] = []


async def main(client: TelegramClient):
    min_id = 0
    max_id = 100

    while True:
        logging.info("Getting messages from {} to {}".format(min_id, max_id))
        res: TotalList | None | Message = await client.get_messages(
            PeerChannel(CHANNEL_ID),
            limit=(max_id - min_id),
            max_id=max_id,
            min_id=min_id,
        )
        assert isinstance(res, TotalList)
        logging.info("Got {} messages".format(len(res)))

        if len(res) == 0:
            break

        message: Message
        for message in reversed(res):
            media: str | None = None
            date: float | None = None
            message_text = ""
            if message.date is not None:
                date = message.date.timestamp()

            if isinstance(message.message, str) and len(message.message) > 0:
                message_text += message.message

            if (
                isinstance(message.media, MessageMediaPhoto)
                and message.media.photo is not None
                and isinstance(message.media.photo, Photo)
            ):
                path = "./images/{}.jpg".format(message.id)
                await client.download_media(message.media, file=path)  # type: ignore
                media = path
            elif message.media is not None:
                pass

            payload.append(
                {"id": message.id, "text": message_text, "media": media, "date": date}
            )
        with open("messages.json", "w", encoding="utf-8") as file:
            file.write(json.dumps(payload))
        max_id += 100
        min_id += 100


if __name__ == "__main__":
    with app as client:
        loop.run_until_complete(main(client))
