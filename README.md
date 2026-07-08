# Telegram background remover bot

Бот принимает фото или изображение файлом и возвращает PNG с прозрачным фоном.

## Что нужно

- Python 3.11, 3.12 или 3.13
- Токен Telegram-бота от BotFather
- Интернет при первом запуске: `rembg` скачает модель в папку `~/.u2net`

## Запуск на Windows PowerShell

```powershell
cd C:\Users\user\Documents\Codex\2026-07-03\new-chat\outputs\telegram-bg-remover-bot
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:BOT_TOKEN="ТОКЕН_ОТ_BOTFATHER"
python bot.py
```

## Как пользоваться

1. Напиши своему боту `/start`.
2. Отправь фото или картинку файлом.
3. Бот вернет `transparent-background.png`.

Для лучшего качества отправляй изображение именно как файл, а не как обычное фото:
так Telegram меньше портит исходник сжатием.

## Полезные настройки

- `REMBG_MODEL=u2net` - стандартная модель общего назначения.
- `REMBG_MODEL=isnet-general-use` - часто лучше для сложных объектов, но может быть тяжелее.
- `ALPHA_MATTING=true` - иногда аккуратнее края волос/шерсти/полупрозрачных областей, но медленнее.
- `MAX_DOWNLOAD_BYTES=20971520` - лимит входного файла, по умолчанию 20 МБ.

## Docker

```powershell
docker build -t telegram-bg-remover-bot .
docker run --rm -e BOT_TOKEN="ТОКЕН_ОТ_BOTFATHER" telegram-bg-remover-bot
```
