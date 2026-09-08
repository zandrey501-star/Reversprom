import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


BASE_PROMPT = """Нарисуй логотип компании «Реверспром» (слово написано русскими буквами, чётко, читаемо). Сфера деятельности — реверс-инжиниринг и производство металлических деталей (восстановление чертежей и изготовление по образцу).

Стиль: современный, технологичный, инженерный. Буквы должны быть сделаны из металла (холодная сталь, титан или алюминий) с лёгкими следами токарной обработки или фрезеровки. В начертании букв можно обыграть элементы шестерни, резьбы, контура детали (например, буква «Р» может содержать стилизованную шестерню или разрез подшипника).
Цветовая гамма: стальной серый, тёмно-синий (добавляет доверия), акцент — графитовый или бронзовый. Без ярких кислотных цветов.
Композиция: горизонтальная или квадратная. Логотип строгий, без излишних теней и гранжей, хорошо смотрится как на белом фоне, так и на тёмном (инвертированная версия).
Требования: русский текст «РЕВЕРСПРОМ» (все заглавные или как обычно, главное — разборчиво). Не добавляй несуществующие детали вроде гаек и болтов внутри букв там, где их быть не может. Буквы должны быть единым шрифтом, но слегка «механизированным».
"""


OUT_DIR = Path(__file__).resolve().parent

JOBS = [
    {
        "filename": "logo_reversprom_horizontal_white.png",
        "size": "1536x1024",
        "prompt_suffix": "Сделай горизонтальную композицию. Фон: чистый белый (#FFFFFF).",
    },
    {
        "filename": "logo_reversprom_horizontal_dark.png",
        "size": "1536x1024",
        "prompt_suffix": "Сделай горизонтальную композицию. Фон: тёмный (графитовый/почти чёрный, например #0B1220). Сделай инвертированную версию, чтобы логотип был контрастным и читаемым на тёмном фоне.",
    },
    {
        "filename": "logo_reversprom_square_white.png",
        "size": "1024x1024",
        "prompt_suffix": "Сделай квадратную композицию (1:1). Фон: чистый белый (#FFFFFF).",
    },
    {
        "filename": "logo_reversprom_square_dark.png",
        "size": "1024x1024",
        "prompt_suffix": "Сделай квадратную композицию (1:1). Фон: тёмный (графитовый/почти чёрный, например #0B1220). Сделай инвертированную версию, чтобы логотип был контрастным и читаемым на тёмном фоне.",
    },
]


def _build_prompt(suffix: str) -> str:
    return f"{BASE_PROMPT}\n\nДополнительно:\n{suffix}\n"


def main() -> None:
    # Load OPENAI_API_KEY from .env in the project folder
    load_dotenv(OUT_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not found. Create /Users/andrey/Reversprom/.env with:\n"
            "OPENAI_API_KEY=YOUR_KEY\n"
        )

    client = OpenAI(api_key=api_key)

    for job in JOBS:
        out_path = OUT_DIR / job["filename"]
        prompt = _build_prompt(job["prompt_suffix"])

        print(f"Generating: {job['filename']} ({job['size']})")
        res = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=job["size"],
            quality="high",
            output_format="png",
        )

        b64 = res.data[0].b64_json
        out_path.write_bytes(base64.b64decode(b64))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

