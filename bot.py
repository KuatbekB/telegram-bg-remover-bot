import asyncio
import logging
import os
import threading
from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter
from rembg import new_session, remove
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger(__name__)


def env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


BOT_NAME = os.getenv("BOT_NAME", "PNG Cutout Bot")
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(20 * 1024 * 1024)))
REMBG_MODEL = os.getenv("REMBG_MODEL", "u2net")
ALPHA_MATTING = env_bool("ALPHA_MATTING", "false")
POST_PROCESS_MASK = env_bool("POST_PROCESS_MASK", "true")
ALPHA_MATTING_FOREGROUND_THRESHOLD = int(os.getenv("ALPHA_MATTING_FOREGROUND_THRESHOLD", "240"))
ALPHA_MATTING_BACKGROUND_THRESHOLD = int(os.getenv("ALPHA_MATTING_BACKGROUND_THRESHOLD", "10"))
ALPHA_MATTING_ERODE_SIZE = int(os.getenv("ALPHA_MATTING_ERODE_SIZE", "10"))
SAVE_DEBUG_IMAGES = env_bool("SAVE_DEBUG_IMAGES", "false")
BOT_DIR = Path(__file__).resolve().parent
DEBUG_DIR = Path(os.getenv("DEBUG_DIR", str(BOT_DIR / "debug_images")))
REMOVE_WHITE_REMNANTS = env_bool("REMOVE_WHITE_REMNANTS", "true")
WHITE_REMNANT_THRESHOLD = int(os.getenv("WHITE_REMNANT_THRESHOLD", "245"))
EDGE_FEATHER_RADIUS = float(os.getenv("EDGE_FEATHER_RADIUS", "0.4"))
REQUIRE_CHANNEL_SUBSCRIPTION = env_bool("REQUIRE_CHANNEL_SUBSCRIPTION", "false")
REQUIRED_CHANNEL = os.getenv("REQUIRED_CHANNEL", "@superski9")
REQUIRED_CHANNEL_URL = os.getenv("REQUIRED_CHANNEL_URL", "https://t.me/superski9")
CHECK_SUBSCRIPTION_CALLBACK = "check_subscription"

BUTTON_HELP = "Как пользоваться"
BUTTON_QUALITY = "Лучшее качество"
BUTTON_SETTINGS = "Текущие настройки"
BUTTON_HIDE = "Скрыть меню"

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [BUTTON_HELP, BUTTON_QUALITY],
        [BUTTON_SETTINGS, BUTTON_HIDE],
    ],
    resize_keyboard=True,
    input_field_placeholder="Отправь фото или картинку файлом",
)


def subscription_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Подписаться на канал", url=REQUIRED_CHANNEL_URL)],
            [InlineKeyboardButton("Проверить подписку", callback_data=CHECK_SUBSCRIPTION_CALLBACK)],
        ]
    )


def subscription_intro_text() -> str:
    if not REQUIRE_CHANNEL_SUBSCRIPTION:
        return ""
    return f"<b>Доступ только после подписки:</b> <a href=\"{REQUIRED_CHANNEL_URL}\">{REQUIRED_CHANNEL}</a>\n\n"


def subscription_settings_text() -> str:
    if not REQUIRE_CHANNEL_SUBSCRIPTION:
        return "Проверка подписки: <code>выключена</code>\n"
    return (
        "Проверка подписки: <code>включена</code>\n"
        f"Канал доступа: <code>{REQUIRED_CHANNEL}</code>\n"
    )

_session = None
_session_lock = threading.Lock()


def get_rembg_session():
    global _session
    with _session_lock:
        if _session is None:
            logger.info("Loading rembg model: %s", REMBG_MODEL)
            _session = new_session(REMBG_MODEL)
        return _session


def remove_background(image_bytes: bytes) -> bytes:
    output_bytes = remove(
        image_bytes,
        session=get_rembg_session(),
        alpha_matting=ALPHA_MATTING,
        alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,
        alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD,
        alpha_matting_erode_size=ALPHA_MATTING_ERODE_SIZE,
        post_process_mask=POST_PROCESS_MASK,
        force_return_bytes=True,
    )
    return polish_output_png(output_bytes)


def polish_output_png(output_bytes: bytes) -> bytes:
    if not REMOVE_WHITE_REMNANTS and EDGE_FEATHER_RADIUS <= 0:
        return output_bytes

    image = Image.open(BytesIO(output_bytes)).convert("RGBA")

    if REMOVE_WHITE_REMNANTS:
        pixels = image.load()
        width, height = image.size
        for y in range(height):
            for x in range(width):
                red, green, blue, alpha = pixels[x, y]
                if alpha and min(red, green, blue) >= WHITE_REMNANT_THRESHOLD:
                    pixels[x, y] = (red, green, blue, 0)

    if EDGE_FEATHER_RADIUS > 0:
        red, green, blue, alpha = image.split()
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=EDGE_FEATHER_RADIUS))
        image = Image.merge("RGBA", (red, green, blue, alpha))

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def normalize_source_suffix(source_suffix: str) -> str:
    suffix = source_suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg"
    return suffix


def save_debug_input(input_bytes: bytes, source_suffix: str, name: str) -> None:
    if not SAVE_DEBUG_IMAGES:
        return

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = normalize_source_suffix(source_suffix)
    (DEBUG_DIR / f"{name}{suffix}").write_bytes(input_bytes)


def save_debug_images(input_bytes: bytes, output_bytes: bytes, source_suffix: str) -> None:
    if not SAVE_DEBUG_IMAGES:
        return

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_debug_input(input_bytes, source_suffix, f"{stamp}-input")
    (DEBUG_DIR / f"{stamp}-result.png").write_bytes(output_bytes)


def progress_bar(percent: int) -> str:
    percent = max(0, min(100, percent))
    filled = round(percent / 10)
    empty = 10 - filled
    return f"[{'#' * filled}{'-' * empty}] {percent}%"


def progress_text(title: str, percent: int, detail: str) -> str:
    return (
        f"<b>{title}</b>\n\n"
        f"<code>{progress_bar(percent)}</code>\n"
        f"{detail}"
    )


async def safe_edit_progress(status_message, title: str, percent: int, detail: str) -> None:
    try:
        await status_message.edit_text(
            progress_text(title, percent, detail),
            parse_mode=ParseMode.HTML,
        )
    except BadRequest as exc:
        if "Message is not modified" not in str(exc):
            raise


async def progress_animation(status_message, stop_event: asyncio.Event) -> None:
    steps = [
        (35, "Нейросеть ищет границы объекта..."),
        (45, "Отделяю объект от фона..."),
        (58, "Проверяю внутренние белые области..."),
        (70, "Очищаю остатки фона..."),
        (82, "Смягчаю край PNG..."),
        (88, "Готовлю файл без сжатия..."),
    ]

    index = 0
    while not stop_event.is_set():
        percent, detail = steps[min(index, len(steps) - 1)]
        await safe_edit_progress(status_message, "Обрабатываю фото", percent, detail)
        index += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.8)
        except asyncio.TimeoutError:
            pass


async def setup_commands(app) -> None:
    await app.bot.set_my_commands(
        [
            BotCommand("start", "открыть меню"),
            BotCommand("help", "как пользоваться"),
            BotCommand("quality", "как получить лучший результат"),
            BotCommand("settings", "текущие настройки обработки"),
        ]
    )


async def get_subscription_check_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[bool, str]:
    user = update.effective_user
    if not user:
        return False, "Не удалось определить пользователя Telegram."

    try:
        member = await context.bot.get_chat_member(chat_id=REQUIRED_CHANNEL, user_id=user.id)
    except TelegramError as exc:
        logger.exception("Failed to check subscription for channel %s", REQUIRED_CHANNEL)
        return (
            False,
            "Не удалось проверить подписку. Убедись, что бот добавлен в канал "
            f"{REQUIRED_CHANNEL} как администратор. Ошибка Telegram: {exc}",
        )

    if member.status in {"creator", "administrator", "member"} or bool(getattr(member, "is_member", False)):
        return True, "Подписка подтверждена."

    return False, "Подписку пока не вижу."


async def is_user_subscribed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    is_subscribed, _reason = await get_subscription_check_result(update, context)
    return is_subscribed


async def send_subscription_required(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "<b>Нужна подписка на канал</b>\n\n"
        "Чтобы пользоваться удалением фона, подпишись на канал "
        f"<a href=\"{REQUIRED_CHANNEL_URL}\">{REQUIRED_CHANNEL}</a>, "
        "а потом нажми <b>Проверить подписку</b>."
    )

    if update.callback_query:
        await update.callback_query.message.reply_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=subscription_keyboard(),
            disable_web_page_preview=True,
        )
    else:
        await update.effective_message.reply_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=subscription_keyboard(),
            disable_web_page_preview=True,
        )


async def require_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if not REQUIRE_CHANNEL_SUBSCRIPTION:
        return True

    if await is_user_subscribed(update, context):
        return True

    await send_subscription_required(update, context)
    return False


async def check_subscription_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer("Проверяю подписку...")

    is_subscribed, reason = await get_subscription_check_result(update, context)
    if is_subscribed:
        try:
            await query.edit_message_text(
                "<b>Подписка подтверждена</b>\n\n"
                "Теперь можешь отправить фото для удаления фона.",
                parse_mode=ParseMode.HTML,
            )
        except BadRequest:
            pass
        await query.message.reply_text(
            "Готово. Отправь фото или картинку файлом.",
            reply_markup=MAIN_KEYBOARD,
        )
        return

    text = (
        "<b>Подписка не подтверждена</b>\n\n"
        f"{reason}\n\n"
        f"Канал: <a href=\"{REQUIRED_CHANNEL_URL}\">{REQUIRED_CHANNEL}</a>\n\n"
        "После подписки нажми кнопку проверки еще раз."
    )
    try:
        await query.edit_message_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=subscription_keyboard(),
            disable_web_page_preview=True,
        )
    except BadRequest:
        await query.message.reply_text(
            text,
            parse_mode=ParseMode.HTML,
            reply_markup=subscription_keyboard(),
            disable_web_page_preview=True,
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        f"<b>{BOT_NAME}</b>\n\n"
        f"{subscription_intro_text()}"
        "Удаляю фон с фото и возвращаю PNG с прозрачностью. "
        "Такой файл можно вставлять в Photoshop, Canva, Figma или на сайт.\n\n"
        "<b>Как начать</b>\n"
        "1. Отправь фото или изображение файлом.\n"
        "2. Дождись обработки.\n"
        "3. Получи PNG без фона.",
        parse_mode=ParseMode.HTML,
        reply_markup=MAIN_KEYBOARD,
    )
    if REQUIRE_CHANNEL_SUBSCRIPTION and not await is_user_subscribed(update, context):
        await send_subscription_required(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "<b>Как пользоваться</b>\n\n"
        "Отправь изображение в чат. Подойдут фото, PNG, JPG, JPEG или WebP.\n\n"
        "Для лучшего качества отправляй картинку как <b>файл</b>, "
        "а не как обычное фото. Telegram меньше сжимает такие изображения.",
        parse_mode=ParseMode.HTML,
        reply_markup=MAIN_KEYBOARD,
    )


async def quality_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "<b>Лучшее качество</b>\n\n"
        "Лучше всего обрабатываются фото, где объект четкий, хорошо освещен "
        "и отделен от фона по цвету.\n\n"
        "Для товарных фото на белом фоне включена дополнительная очистка белых остатков. "
        "Если объект сам белый или очень светлый, эту настройку лучше отключать.",
        parse_mode=ParseMode.HTML,
        reply_markup=MAIN_KEYBOARD,
    )


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "<b>Текущие настройки</b>\n\n"
        f"Модель: <code>{REMBG_MODEL}</code>\n"
        f"Alpha matting: <code>{ALPHA_MATTING}</code>\n"
        f"Очистка маски: <code>{POST_PROCESS_MASK}</code>\n"
        f"Удаление белых остатков: <code>{REMOVE_WHITE_REMNANTS}</code>\n"
        f"Смягчение края: <code>{EDGE_FEATHER_RADIUS}</code>\n"
        f"{subscription_settings_text()}"
        f"Лимит файла: <code>{MAX_DOWNLOAD_BYTES // (1024 * 1024)} МБ</code>",
        parse_mode=ParseMode.HTML,
        reply_markup=MAIN_KEYBOARD,
    )


async def hide_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Меню скрыто. Чтобы вернуть его, отправь /start.",
        reply_markup=ReplyKeyboardRemove(),
    )


def get_input_file_id_and_size(update: Update) -> tuple[str | None, int | None]:
    message = update.effective_message
    if message.photo:
        photo = message.photo[-1]
        return photo.file_id, photo.file_size

    document = message.document
    if document and document.mime_type and document.mime_type.startswith("image/"):
        return document.file_id, document.file_size

    return None, None


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    file_id, file_size = get_input_file_id_and_size(update)
    input_bytes = None
    source_suffix = ".jpg"

    if not file_id:
        await message.reply_text(
            "Пришли изображение: фото или файл PNG/JPG/WebP.",
            reply_markup=MAIN_KEYBOARD,
        )
        return

    if not await require_subscription(update, context):
        return

    if file_size and file_size > MAX_DOWNLOAD_BYTES:
        limit_mb = MAX_DOWNLOAD_BYTES // (1024 * 1024)
        await message.reply_text(
            f"Файл слишком большой. Максимум: {limit_mb} МБ.",
            reply_markup=MAIN_KEYBOARD,
        )
        return

    status_message = await message.reply_text(
        progress_text("Старт обработки", 5, "Получаю изображение из Telegram..."),
        parse_mode=ParseMode.HTML,
    )

    stop_progress = asyncio.Event()
    progress_task = None

    try:
        await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
        telegram_file = await context.bot.get_file(file_id)

        await safe_edit_progress(status_message, "Загрузка", 18, "Скачиваю исходное изображение...")
        input_buffer = BytesIO()
        await telegram_file.download_to_memory(input_buffer)
        input_bytes = input_buffer.getvalue()
        source_suffix = Path(telegram_file.file_path or "").suffix

        await safe_edit_progress(status_message, "Обрабатываю фото", 28, "Запускаю удаление фона...")
        await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.UPLOAD_DOCUMENT)

        progress_task = asyncio.create_task(progress_animation(status_message, stop_progress))
        output_bytes = await asyncio.to_thread(remove_background, input_bytes)
        stop_progress.set()
        await progress_task

        await safe_edit_progress(status_message, "Финальная сборка", 94, "Сохраняю PNG с прозрачностью...")
        await asyncio.to_thread(save_debug_images, input_bytes, output_bytes, source_suffix)

        output_buffer = BytesIO(output_bytes)
        output_buffer.name = "transparent-background.png"

        await safe_edit_progress(status_message, "Готово", 100, "Отправляю файл без сжатия...")
        await message.reply_document(
            document=output_buffer,
            filename="transparent-background.png",
            caption=(
                "Готово: PNG с прозрачным фоном.\n"
                "Файл отправлен как документ, чтобы Telegram не испортил качество."
            ),
            reply_markup=MAIN_KEYBOARD,
        )
        await status_message.delete()
    except Exception:
        logger.exception("Failed to process image")
        stop_progress.set()
        if progress_task:
            await asyncio.gather(progress_task, return_exceptions=True)
        if input_bytes:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            await asyncio.to_thread(save_debug_input, input_bytes, source_suffix, f"{stamp}-failed-input")
        await status_message.edit_text(
            "Не получилось обработать картинку. Попробуй другое фото или файл поменьше."
        )
        await message.reply_text(
            "Можешь отправить новое изображение.",
            reply_markup=MAIN_KEYBOARD,
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.effective_message.text
    if text == BUTTON_HELP:
        await help_command(update, context)
    elif text == BUTTON_QUALITY:
        await quality_command(update, context)
    elif text == BUTTON_SETTINGS:
        await settings_command(update, context)
    elif text == BUTTON_HIDE:
        await hide_menu(update, context)
    else:
        await update.effective_message.reply_text(
            "Отправь фото или изображение файлом. Меню можно открыть командой /start.",
            reply_markup=MAIN_KEYBOARD,
        )


def main() -> None:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("Set BOT_TOKEN environment variable first.")

    if SAVE_DEBUG_IMAGES:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Debug images will be saved to: %s", DEBUG_DIR)

    app = ApplicationBuilder().token(token).post_init(setup_commands).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("quality", quality_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CallbackQueryHandler(check_subscription_callback, pattern=f"^{CHECK_SUBSCRIPTION_CALLBACK}$"))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
