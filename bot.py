import asyncio
import html
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar

from PIL import Image, ImageFilter, UnidentifiedImageError
from rembg import new_session, remove
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, NetworkError, TelegramError
from telegram.ext import (
    AIORateLimiter,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger(__name__)
# HTTP client INFO logs may contain Bot API URLs. Keep credentials and request
# metadata out of normal production logs even when the application logs at INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


BOT_NAME = os.getenv("BOT_NAME", "ФонOFF")
MAX_DOWNLOAD_BYTES = max(1, int(os.getenv("MAX_DOWNLOAD_BYTES", str(20 * 1024 * 1024))))
MAX_OUTPUT_BYTES = max(1, int(os.getenv("MAX_OUTPUT_BYTES", str(49 * 1024 * 1024))))
MAX_IMAGE_PIXELS = max(1, int(os.getenv("MAX_IMAGE_PIXELS", "12000000")))
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2")))
MAX_QUEUE_SIZE = max(0, int(os.getenv("MAX_QUEUE_SIZE", "20")))
UPDATE_CONCURRENCY = max(1, int(os.getenv("UPDATE_CONCURRENCY", "32")))
USER_COOLDOWN_SECONDS = max(0.0, float(os.getenv("USER_COOLDOWN_SECONDS", "3")))
USER_RATE_LIMIT_COUNT = max(1, int(os.getenv("USER_RATE_LIMIT_COUNT", "5")))
USER_RATE_LIMIT_WINDOW_SECONDS = max(1.0, float(os.getenv("USER_RATE_LIMIT_WINDOW_SECONDS", "60")))
PROCESSING_TIMEOUT_SECONDS = max(1.0, float(os.getenv("PROCESSING_TIMEOUT_SECONDS", "120")))
PROGRESS_UPDATE_SECONDS = max(2.0, float(os.getenv("PROGRESS_UPDATE_SECONDS", "4")))
TELEGRAM_MAX_RETRIES = max(0, int(os.getenv("TELEGRAM_MAX_RETRIES", "2")))
TELEGRAM_RETRY_BASE_SECONDS = max(0.1, float(os.getenv("TELEGRAM_RETRY_BASE_SECONDS", "1")))
RETRY_DATA_TTL_SECONDS = max(60.0, float(os.getenv("RETRY_DATA_TTL_SECONDS", "3600")))
PRIVACY_CLEANUP_INTERVAL_SECONDS = max(5.0, float(os.getenv("PRIVACY_CLEANUP_INTERVAL_SECONDS", "60")))
REMBG_MODEL = os.getenv("REMBG_MODEL", "u2net")
RETRY_REMBG_MODEL = os.getenv("RETRY_REMBG_MODEL", "isnet-general-use")
PRELOAD_MODELS = env_bool("PRELOAD_MODELS", "false")
PRELOAD_PRIMARY_MODEL = env_bool("PRELOAD_PRIMARY_MODEL", "true")
ALPHA_MATTING = env_bool("ALPHA_MATTING", "false")
POST_PROCESS_MASK = env_bool("POST_PROCESS_MASK", "true")
ALPHA_MATTING_FOREGROUND_THRESHOLD = int(os.getenv("ALPHA_MATTING_FOREGROUND_THRESHOLD", "240"))
ALPHA_MATTING_BACKGROUND_THRESHOLD = int(os.getenv("ALPHA_MATTING_BACKGROUND_THRESHOLD", "10"))
ALPHA_MATTING_ERODE_SIZE = int(os.getenv("ALPHA_MATTING_ERODE_SIZE", "10"))
SAVE_DEBUG_IMAGES = env_bool("SAVE_DEBUG_IMAGES", "false")
READY_FILE = Path(os.getenv("READY_FILE", str(Path(os.getenv("TEMP", "/tmp")) / "telegram-bg-remover-ready")))
HEARTBEAT_FILE = Path(
    os.getenv("HEARTBEAT_FILE", str(Path(os.getenv("TEMP", "/tmp")) / "telegram-bg-remover-heartbeat"))
)
HEALTH_HEARTBEAT_SECONDS = max(1.0, float(os.getenv("HEALTH_HEARTBEAT_SECONDS", "10")))
INSTANCE_LOCK_FILE = Path(
    os.getenv("INSTANCE_LOCK_FILE", str(Path(os.getenv("TEMP", "/tmp")) / "telegram-bg-remover.lock"))
)
REMOVE_WHITE_REMNANTS = env_bool("REMOVE_WHITE_REMNANTS", "false")
WHITE_REMNANT_THRESHOLD = int(os.getenv("WHITE_REMNANT_THRESHOLD", "245"))
EDGE_FEATHER_RADIUS = float(os.getenv("EDGE_FEATHER_RADIUS", "0.4"))
REQUIRE_CHANNEL_SUBSCRIPTION = env_bool("REQUIRE_CHANNEL_SUBSCRIPTION", "false")
REQUIRED_CHANNEL = os.getenv("REQUIRED_CHANNEL", "@superski9")
REQUIRED_CHANNEL_URL = os.getenv("REQUIRED_CHANNEL_URL", "https://t.me/superski9")
PRIVACY_CONTACT = os.getenv("PRIVACY_CONTACT", REQUIRED_CHANNEL)
PRIVACY_POLICY_URL = os.getenv("PRIVACY_POLICY_URL", "https://telegram.org/privacy-tpa")
CHECK_SUBSCRIPTION_CALLBACK = "check_subscription"
RETRY_PROCESSING_CALLBACK = "retry_processing"

BUTTON_HELP = "Как пользоваться"
BUTTON_HIDE = "Скрыть меню"

ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImageError(ValueError):
    pass


class OutputTooLargeError(ValueError):
    pass


@dataclass(slots=True)
class ProcessingJob:
    user_id: int = field(repr=False)
    file_id: str = field(repr=False)
    message: Any = field(repr=False)
    bot: Any = field(repr=False)
    user_data: dict = field(repr=False)
    status_message: Any = field(repr=False)
    is_retry: bool = False
    source_suffix: str = ".jpg"


T = TypeVar("T")

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [[BUTTON_HELP, BUTTON_HIDE]],
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


def retry_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔄 Попробовать ещё раз", callback_data=RETRY_PROCESSING_CALLBACK)]]
    )


def subscription_intro_text() -> str:
    if not REQUIRE_CHANNEL_SUBSCRIPTION:
        return ""
    return f"<b>Доступ только после подписки:</b> <a href=\"{REQUIRED_CHANNEL_URL}\">{REQUIRED_CHANNEL}</a>\n\n"


_sessions = {}
_session_lock = threading.Lock()
_instance_lock_handle = None


def acquire_instance_lock() -> None:
    global _instance_lock_handle
    if _instance_lock_handle is not None:
        raise SystemExit("Another bot instance is already active in this process.")

    INSTANCE_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = INSTANCE_LOCK_FILE.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise SystemExit("Another bot instance is already running.") from exc

    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()).encode("ascii"))
    handle.flush()
    _instance_lock_handle = handle
    logger.info("Acquired single-instance lock: %s", INSTANCE_LOCK_FILE)


def release_instance_lock() -> None:
    global _instance_lock_handle
    handle = _instance_lock_handle
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        logger.exception("Failed to release instance lock cleanly")
    finally:
        handle.close()
        _instance_lock_handle = None


def get_rembg_session(model_name: str = REMBG_MODEL):
    with _session_lock:
        if model_name not in _sessions:
            logger.info("Loading rembg model: %s", model_name)
            _sessions[model_name] = new_session(model_name)
        return _sessions[model_name]


def preload_models() -> None:
    for model_name in dict.fromkeys((REMBG_MODEL, RETRY_REMBG_MODEL)):
        get_rembg_session(model_name)
    logger.info("Background-removal models are ready")


def validate_image_bytes(image_bytes: bytes) -> tuple[int, int, str]:
    if not image_bytes:
        raise InvalidImageError("Файл изображения пуст.")
    if len(image_bytes) > MAX_DOWNLOAD_BYTES:
        raise InvalidImageError(
            f"Файл превышает лимит {MAX_DOWNLOAD_BYTES // (1024 * 1024)} МБ."
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise InvalidImageError("Поддерживаются только PNG, JPG, JPEG и WebP.")
            if width <= 0 or height <= 0:
                raise InvalidImageError("У изображения некорректный размер.")
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError(
                    f"Слишком большое разрешение: максимум {MAX_IMAGE_PIXELS:,} пикселей."
                )
            image.verify()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Файл повреждён или не является допустимым изображением.") from exc

    return width, height, image_format


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


def remove_background_retry(image_bytes: bytes) -> bytes:
    output_bytes = remove(
        image_bytes,
        session=get_rembg_session(RETRY_REMBG_MODEL),
        alpha_matting=False,
        alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,
        alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD,
        alpha_matting_erode_size=ALPHA_MATTING_ERODE_SIZE,
        post_process_mask=True,
        force_return_bytes=True,
    )
    return polish_output_png(output_bytes, remove_white_remnants=False, feather_radius=0)


def new_processing_state(queue: asyncio.Queue | None = None) -> dict[str, Any]:
    return {
        "active_users": set(),
        "active_jobs": 0,
        "queue": queue,
        "worker_tasks": [],
        "heartbeat_task": None,
        "privacy_cleanup_task": None,
    }


def get_processing_state(application) -> dict[str, Any]:
    return application.bot_data.setdefault("processing_state", new_processing_state())


def try_begin_processing(context: ContextTypes.DEFAULT_TYPE, user_id: int, enforce_cooldown: bool) -> str | None:
    state = get_processing_state(context.application)

    if user_id in state["active_users"]:
        return "Предыдущее изображение уже в очереди или обрабатывается. Дождись результата."

    now = time.monotonic()
    request_times = [
        timestamp
        for timestamp in context.user_data.get("processing_request_times", [])
        if now - timestamp < USER_RATE_LIMIT_WINDOW_SECONDS
    ]
    if len(request_times) >= USER_RATE_LIMIT_COUNT:
        oldest = min(request_times)
        remaining = USER_RATE_LIMIT_WINDOW_SECONDS - (now - oldest)
        return f"Достигнут лимит запросов. Попробуй снова через {max(1, round(remaining))} сек."

    if enforce_cooldown and USER_COOLDOWN_SECONDS > 0:
        last_request_at = context.user_data.get("last_processing_request_at", 0.0)
        remaining = USER_COOLDOWN_SECONDS - (now - last_request_at)
        if remaining > 0:
            return f"Слишком быстро. Попробуй снова через {max(1, round(remaining))} сек."

    capacity = MAX_CONCURRENT_JOBS + MAX_QUEUE_SIZE
    if len(state["active_users"]) >= capacity:
        return "Очередь обработки заполнена. Попробуй ещё раз через несколько минут."

    request_times.append(now)
    context.user_data["processing_request_times"] = request_times
    context.user_data["last_processing_request_at"] = now
    state["active_users"].add(user_id)
    return None


def mark_processing_started(application) -> None:
    state = get_processing_state(application)
    state["active_jobs"] += 1


def finish_processing(application_or_context, user_id: int, was_active: bool = True) -> None:
    application = getattr(application_or_context, "application", application_or_context)
    state = application.bot_data.get("processing_state")
    if not state:
        return
    state["active_users"].discard(user_id)
    if was_active:
        state["active_jobs"] = max(0, state["active_jobs"] - 1)


def prune_expired_user_data(user_data: dict, now: float | None = None) -> bool:
    """Remove expired retry and rate-limit metadata. Returns True if retry data expired."""
    current_time = time.monotonic() if now is None else now
    retry_expired = False
    retry_data = user_data.get("retry_processing")
    if retry_data and current_time >= retry_data.get("expires_at", 0):
        user_data.pop("retry_processing", None)
        retry_expired = True

    request_times = [
        timestamp
        for timestamp in user_data.get("processing_request_times", [])
        if current_time - timestamp < USER_RATE_LIMIT_WINDOW_SECONDS
    ]
    if request_times:
        user_data["processing_request_times"] = request_times
    else:
        user_data.pop("processing_request_times", None)
        last_request_at = user_data.get("last_processing_request_at")
        if last_request_at is None or current_time - last_request_at >= max(
            USER_COOLDOWN_SECONDS,
            USER_RATE_LIMIT_WINDOW_SECONDS,
        ):
            user_data.pop("last_processing_request_at", None)

    return retry_expired


async def telegram_request(operation: Callable[[], Awaitable[T]], operation_name: str) -> T:
    for attempt in range(TELEGRAM_MAX_RETRIES + 1):
        try:
            return await operation()
        except NetworkError:
            if attempt >= TELEGRAM_MAX_RETRIES:
                raise
            delay = TELEGRAM_RETRY_BASE_SECONDS * (2**attempt)
        logger.warning(
            "Telegram operation %s failed; retrying in %.1f seconds (attempt %s/%s)",
            operation_name,
            delay,
            attempt + 1,
            TELEGRAM_MAX_RETRIES,
        )
        await asyncio.sleep(delay)

    raise RuntimeError(f"Telegram operation {operation_name} exhausted retries")


def polish_output_png(
    output_bytes: bytes,
    remove_white_remnants: bool = REMOVE_WHITE_REMNANTS,
    feather_radius: float = EDGE_FEATHER_RADIUS,
) -> bytes:
    if not remove_white_remnants and feather_radius <= 0:
        return output_bytes

    image = Image.open(BytesIO(output_bytes)).convert("RGBA")

    if remove_white_remnants:
        pixels = image.load()
        width, height = image.size
        for y in range(height):
            for x in range(width):
                red, green, blue, alpha = pixels[x, y]
                if alpha and min(red, green, blue) >= WHITE_REMNANT_THRESHOLD:
                    pixels[x, y] = (red, green, blue, 0)

    if feather_radius > 0:
        red, green, blue, alpha = image.split()
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        image = Image.merge("RGBA", (red, green, blue, alpha))

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


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
        await telegram_request(
            lambda: status_message.edit_text(
                progress_text(title, percent, detail),
                parse_mode=ParseMode.HTML,
            ),
            "edit progress",
        )
    except BadRequest as exc:
        if "Message is not modified" not in str(exc):
            logger.warning("Could not edit progress message: %s", exc)
    except TelegramError as exc:
        logger.warning("Could not edit progress message after retries: %s", exc)


async def safe_delete_message(message) -> None:
    if not message:
        return
    try:
        await telegram_request(message.delete, "delete status message")
    except TelegramError as exc:
        logger.warning("Could not delete status message: %s", exc)


async def safe_answer_query(query, text: str | None = None, show_alert: bool = False) -> None:
    try:
        await query.answer(text=text, show_alert=show_alert)
    except TelegramError as exc:
        logger.warning("Could not answer callback query: %s", exc)


async def safe_send_chat_action(bot, chat_id: int, action: str) -> None:
    try:
        await telegram_request(
            lambda: bot.send_chat_action(chat_id=chat_id, action=action),
            "send chat action",
        )
    except TelegramError as exc:
        logger.warning("Could not send chat action: %s", exc)


async def progress_animation(status_message, stop_event: asyncio.Event, title: str = "Обрабатываю фото") -> None:
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
        await safe_edit_progress(status_message, title, percent, detail)
        index += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=PROGRESS_UPDATE_SECONDS)
        except asyncio.TimeoutError:
            pass


async def heartbeat_loop() -> None:
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    while True:
        HEARTBEAT_FILE.write_text(f"{time.time()}\n", encoding="utf-8")
        await asyncio.sleep(HEALTH_HEARTBEAT_SECONDS)


async def privacy_cleanup_loop(app) -> None:
    while True:
        await asyncio.sleep(PRIVACY_CLEANUP_INTERVAL_SECONDS)
        try:
            now = time.monotonic()
            state = get_processing_state(app)
            active_users = state["active_users"]
            for user_id, user_data in list(getattr(app, "user_data", {}).items()):
                prune_expired_user_data(user_data, now)
                if not user_data and user_id not in active_users:
                    app.drop_user_data(user_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Temporary privacy data cleanup failed")


async def setup_commands(app) -> None:
    queue_capacity = MAX_CONCURRENT_JOBS + MAX_QUEUE_SIZE
    state = new_processing_state(asyncio.Queue(maxsize=queue_capacity))
    app.bot_data["processing_state"] = state

    if PRELOAD_MODELS:
        logger.info("Preloading configured models before accepting updates")
        await asyncio.to_thread(preload_models)
    elif PRELOAD_PRIMARY_MODEL:
        logger.info("Preloading primary model before accepting updates")
        await asyncio.to_thread(get_rembg_session, REMBG_MODEL)

    await telegram_request(
        lambda: app.bot.set_my_commands(
            [
                BotCommand("start", "открыть меню"),
                BotCommand("help", "как пользоваться"),
                BotCommand("privacy", "конфиденциальность"),
                BotCommand("delete_me", "удалить временные данные"),
            ]
        ),
        "set bot commands",
    )

    state["worker_tasks"] = [
        asyncio.create_task(processing_worker(app, worker_index), name=f"image-worker-{worker_index}")
        for worker_index in range(MAX_CONCURRENT_JOBS)
    ]
    state["heartbeat_task"] = asyncio.create_task(heartbeat_loop(), name="health-heartbeat")
    state["privacy_cleanup_task"] = asyncio.create_task(
        privacy_cleanup_loop(app),
        name="privacy-cleanup",
    )

    READY_FILE.parent.mkdir(parents=True, exist_ok=True)
    READY_FILE.write_text("ready\n", encoding="utf-8")
    logger.info(
        "Bot ready: %s worker(s), queue capacity %s, update concurrency %s",
        MAX_CONCURRENT_JOBS,
        MAX_QUEUE_SIZE,
        UPDATE_CONCURRENCY,
    )


async def shutdown_app(app) -> None:
    state = app.bot_data.get("processing_state", {})
    tasks = [*state.get("worker_tasks", [])]
    heartbeat_task = state.get("heartbeat_task")
    if heartbeat_task:
        tasks.append(heartbeat_task)
    privacy_cleanup_task = state.get("privacy_cleanup_task")
    if privacy_cleanup_task:
        tasks.append(privacy_cleanup_task)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    READY_FILE.unlink(missing_ok=True)
    HEARTBEAT_FILE.unlink(missing_ok=True)


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


async def retry_processing_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    retry_expired = prune_expired_user_data(context.user_data)
    retry_data = context.user_data.get("retry_processing")
    user_id = update.effective_user.id

    if retry_expired:
        await safe_answer_query(query, "Срок повторной обработки истёк. Отправь изображение заново.", show_alert=True)
        return

    if not retry_data or retry_data.get("used"):
        await safe_answer_query(query, "Повторная попытка уже использована.", show_alert=True)
        return

    if retry_data.get("result_message_id") != query.message.message_id:
        await safe_answer_query(query, "Эта кнопка больше не активна.", show_alert=True)
        return

    await safe_answer_query(query, "Добавляю повторную обработку в очередь...")
    if not await require_subscription(update, context):
        return

    busy_reason = try_begin_processing(context, user_id, enforce_cooldown=False)
    if busy_reason:
        await query.message.reply_text(busy_reason, reply_markup=MAIN_KEYBOARD)
        return

    retry_data["used"] = True
    status_message = None
    try:
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except TelegramError as exc:
            logger.warning("Could not remove retry button: %s", exc)

        state = get_processing_state(context.application)
        queue_position = state["queue"].qsize() + 1
        status_message = await telegram_request(
            lambda: query.message.reply_text(
                progress_text(
                    "Задание принято",
                    5,
                    f"Позиция в очереди: {queue_position}. Ожидаю свободный обработчик...",
                ),
                parse_mode=ParseMode.HTML,
            ),
            "send retry queue status",
        )
        state["queue"].put_nowait(
            ProcessingJob(
                user_id=user_id,
                file_id=retry_data["file_id"],
                message=query.message,
                bot=context.bot,
                user_data=context.user_data,
                status_message=status_message,
                is_retry=True,
                source_suffix=retry_data.get("source_suffix", ".jpg"),
            )
        )
    except Exception:
        logger.exception("Failed to enqueue retry processing")
        retry_data["used"] = False
        finish_processing(context, user_id, was_active=False)
        if status_message:
            await safe_edit_progress(
                status_message,
                "Ошибка очереди",
                0,
                "Не удалось добавить задание. Попробуй ещё раз.",
            )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        f"<b>{BOT_NAME}</b>\n\n"
        f"{subscription_intro_text()}"
        "🪄 <b>Удаляй фон с фотографий за несколько секунд!</b>\n\n"
        "Просто отправь изображение — бот автоматически аккуратно вырежет объект "
        "и вернёт готовую картинку с прозрачным фоном.\n\n"
        "⚡ Быстро\n"
        "✨ Качественно\n"
        "📱 Без сложных настроек\n\n"
        "Обработка данных и удаление временных сведений: /privacy",
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
        "а не как обычное фото. Telegram меньше сжимает такие изображения.\n\n"
        "Конфиденциальность: /privacy",
        parse_mode=ParseMode.HTML,
        reply_markup=MAIN_KEYBOARD,
    )


async def privacy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    retry_minutes = max(1, round(RETRY_DATA_TTL_SECONDS / 60))
    contact = html.escape(PRIVACY_CONTACT)
    policy_url = html.escape(PRIVACY_POLICY_URL, quote=True)
    subscription_note = (
        "При включённом ограничении доступа бот проверяет статус подписки, но не сохраняет его.\n\n"
        if REQUIRE_CHANNEL_SUBSCRIPTION
        else ""
    )
    await update.effective_message.reply_text(
        "<b>Конфиденциальность</b>\n\n"
        "Бот получает Telegram ID и отправленное изображение только для выполнения запроса, "
        "ограничения нагрузки и защиты от злоупотреблений. Изображение обрабатывается локальной "
        "моделью и не передаётся сторонним AI-сервисам.\n\n"
        "Изображение обычно находится только в оперативной памяти. Telegram file_id для кнопки "
        f"повторной обработки хранится до {retry_minutes} мин. Уже запущенная обработка может "
        "удерживать ссылку до своего завершения. Метаданные rate limit "
        "автоматически удаляются после окончания их срока. Telegram ID не записывается в журналы бота.\n\n"
        f"{subscription_note}"
        "Бот не создаёт базу пользователей, не продаёт данные и не использует их для рекламы. "
        "Команда /delete_me удаляет временные данные, которыми управляет бот. Сообщения в самом "
        "Telegram регулируются политикой Telegram.\n\n"
        f"Контакт оператора: {contact}\n"
        f'<a href="{policy_url}">Стандартная политика конфиденциальности Telegram для ботов</a>',
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def delete_my_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        await update.effective_message.reply_text("Не удалось определить пользователя Telegram.")
        return

    state = get_processing_state(context.application)
    is_processing = user.id in state["active_users"]
    context.user_data.clear()
    if is_processing:
        context.user_data["delete_after_processing"] = True
        text = (
            "Временные данные удалены. Текущее задание будет завершено, после чего оставшиеся "
            "служебные данные также будут удалены автоматически."
        )
    else:
        context.application.drop_user_data(user.id)
        text = "Все временные данные, которыми управляет бот, удалены."

    await update.effective_message.reply_text(text, reply_markup=MAIN_KEYBOARD)


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
    user_id = update.effective_user.id if update.effective_user else message.chat_id
    file_id, file_size = get_input_file_id_and_size(update)

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

    busy_reason = try_begin_processing(context, user_id, enforce_cooldown=True)
    if busy_reason:
        await message.reply_text(busy_reason, reply_markup=MAIN_KEYBOARD)
        return

    status_message = None
    try:
        state = get_processing_state(context.application)
        queue_position = state["queue"].qsize() + 1
        status_message = await telegram_request(
            lambda: message.reply_text(
                progress_text(
                    "Задание принято",
                    5,
                    f"Позиция в очереди: {queue_position}. Ожидаю свободный обработчик...",
                ),
                parse_mode=ParseMode.HTML,
            ),
            "send queue status",
        )
        state["queue"].put_nowait(
            ProcessingJob(
                user_id=user_id,
                file_id=file_id,
                message=message,
                bot=context.bot,
                user_data=context.user_data,
                status_message=status_message,
            )
        )
    except Exception:
        logger.exception("Failed to enqueue image processing")
        finish_processing(context, user_id, was_active=False)
        if status_message:
            await safe_edit_progress(
                status_message,
                "Ошибка очереди",
                0,
                "Не удалось добавить задание. Попробуй ещё раз.",
            )


async def download_job_input(job: ProcessingJob) -> tuple[bytes, str]:
    telegram_file = await telegram_request(
        lambda: job.bot.get_file(job.file_id),
        "get Telegram file",
    )
    input_buffer = BytesIO()

    async def download() -> None:
        input_buffer.seek(0)
        input_buffer.truncate(0)
        await telegram_file.download_to_memory(input_buffer)

    await telegram_request(download, "download Telegram file")
    source_suffix = job.source_suffix if job.is_retry else Path(telegram_file.file_path or "").suffix
    return input_buffer.getvalue(), source_suffix


async def send_result_document(
    job: ProcessingJob,
    output_bytes: bytes,
    filename: str,
    caption: str,
    reply_markup=None,
):
    async def send_document():
        output_buffer = BytesIO(output_bytes)
        output_buffer.name = filename
        return await job.message.reply_document(
            document=output_buffer,
            filename=filename,
            caption=caption,
            reply_markup=reply_markup,
            write_timeout=60,
            read_timeout=60,
        )

    return await telegram_request(send_document, "send result document")


async def process_image_job(job: ProcessingJob) -> None:
    status_message = job.status_message
    input_bytes = None
    source_suffix = job.source_suffix
    stop_progress = asyncio.Event()
    progress_task = None

    try:
        title = "Повторная обработка" if job.is_retry else "Загрузка"
        await safe_edit_progress(status_message, title, 12, "Скачиваю исходное изображение...")
        await safe_send_chat_action(job.bot, job.message.chat_id, ChatAction.TYPING)
        input_bytes, source_suffix = await download_job_input(job)
        await asyncio.to_thread(validate_image_bytes, input_bytes)

        processing_title = "Повторная обработка" if job.is_retry else "Обрабатываю фото"
        await safe_edit_progress(status_message, processing_title, 28, "Запускаю нейросеть...")
        await safe_send_chat_action(job.bot, job.message.chat_id, ChatAction.UPLOAD_DOCUMENT)
        progress_task = asyncio.create_task(
            progress_animation(status_message, stop_progress, processing_title),
            name="image-progress",
        )

        processor = remove_background_retry if job.is_retry else remove_background
        processing_task = asyncio.create_task(asyncio.to_thread(processor, input_bytes))
        try:
            output_bytes = await asyncio.wait_for(
                asyncio.shield(processing_task),
                timeout=PROCESSING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            stop_progress.set()
            if progress_task:
                await asyncio.gather(progress_task, return_exceptions=True)
                progress_task = None
            await safe_edit_progress(
                status_message,
                "Превышено время обработки",
                0,
                "Изображение обрабатывается слишком долго. Попробуй файл меньшего разрешения.",
            )
            # Native ONNX inference cannot be cancelled safely. Keep this worker occupied
            # until it really exits so another job cannot overcommit memory in the meantime.
            await asyncio.gather(processing_task, return_exceptions=True)
            raise
        if len(output_bytes) > MAX_OUTPUT_BYTES:
            raise OutputTooLargeError(
                f"Получившийся PNG больше {MAX_OUTPUT_BYTES // (1024 * 1024)} МБ. "
                "Отправь изображение с меньшим разрешением."
            )

        stop_progress.set()
        if progress_task:
            await asyncio.gather(progress_task, return_exceptions=True)
            progress_task = None

        await safe_edit_progress(status_message, "Финальная сборка", 94, "Готовлю PNG к отправке...")
        if job.is_retry:
            await send_result_document(
                job,
                output_bytes,
                "transparent-background-retry.png",
                "Готово: повторная обработка с более чёткими краями.\nСравни результат с первым вариантом.",
            )
            job.user_data.pop("retry_processing", None)
        else:
            result_message = await send_result_document(
                job,
                output_bytes,
                "transparent-background.png",
                (
                    "Готово: PNG с прозрачным фоном.\n"
                    "Файл отправлен как документ, чтобы Telegram не испортил качество.\n\n"
                    "Если фон удалён неудачно, используй дополнительную попытку."
                ),
                retry_keyboard(),
            )
            job.user_data["retry_processing"] = {
                "file_id": job.file_id,
                "source_suffix": source_suffix,
                "result_message_id": result_message.message_id,
                "used": False,
                "expires_at": time.monotonic() + RETRY_DATA_TTL_SECONDS,
            }

        await safe_delete_message(status_message)
        logger.info(
            "Image job completed: retry=%s input_bytes=%s output_bytes=%s",
            job.is_retry,
            len(input_bytes),
            len(output_bytes),
        )
    except InvalidImageError as exc:
        logger.warning("Rejected invalid image: %s", exc)
        await safe_edit_progress(status_message, "Изображение отклонено", 0, str(exc))
        if job.is_retry:
            job.user_data.pop("retry_processing", None)
    except OutputTooLargeError as exc:
        logger.warning("Image result is too large: %s", exc)
        await safe_edit_progress(status_message, "Результат слишком большой", 100, str(exc))
        if job.is_retry:
            job.user_data.pop("retry_processing", None)
    except asyncio.TimeoutError:
        logger.error("Image processing timed out")
        if job.is_retry:
            job.user_data.pop("retry_processing", None)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Failed to process image job")
        await safe_edit_progress(
            status_message,
            "Ошибка обработки",
            0,
            "Не получилось обработать изображение. Попробуй другой файл или файл меньшего размера.",
        )
        if job.is_retry:
            job.user_data.pop("retry_processing", None)
    finally:
        stop_progress.set()
        if progress_task:
            await asyncio.gather(progress_task, return_exceptions=True)


async def processing_worker(app, worker_index: int) -> None:
    state = get_processing_state(app)
    queue = state["queue"]
    logger.info("Image worker %s started", worker_index)
    while True:
        job = await queue.get()
        mark_processing_started(app)
        try:
            await process_image_job(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unhandled error in image worker %s", worker_index)
        finally:
            finish_processing(app, job.user_id, was_active=True)
            job_user_data = getattr(job, "user_data", {})
            if job_user_data.pop("delete_after_processing", False):
                job_user_data.clear()
                app.drop_user_data(job.user_id)
            queue.task_done()


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.effective_message.text
    if text == BUTTON_HELP:
        await help_command(update, context)
    elif text == BUTTON_HIDE:
        await hide_menu(update, context)
    else:
        await update.effective_message.reply_text(
            "Отправь фото или изображение файлом. Меню можно открыть командой /start.",
            reply_markup=MAIN_KEYBOARD,
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error_type = type(context.error).__name__ if context.error else "UnknownError"
    logger.error("Unhandled Telegram handler error: error_type=%s", error_type)


def validate_privacy_configuration() -> None:
    if SAVE_DEBUG_IMAGES:
        raise SystemExit(
            "SAVE_DEBUG_IMAGES must remain false: this build does not persist user images without encryption."
        )
    if not PRIVACY_CONTACT.strip():
        raise SystemExit("Set PRIVACY_CONTACT before starting the bot.")
    if not PRIVACY_POLICY_URL.startswith("https://"):
        raise SystemExit("PRIVACY_POLICY_URL must use HTTPS.")


def main() -> None:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("Set BOT_TOKEN environment variable first.")
    validate_privacy_configuration()

    acquire_instance_lock()
    try:
        READY_FILE.unlink(missing_ok=True)
        HEARTBEAT_FILE.unlink(missing_ok=True)

        app = (
            ApplicationBuilder()
            .token(token)
            .concurrent_updates(UPDATE_CONCURRENCY)
            .rate_limiter(AIORateLimiter(max_retries=TELEGRAM_MAX_RETRIES))
            .connect_timeout(10)
            .read_timeout(30)
            .write_timeout(30)
            .media_write_timeout(60)
            .pool_timeout(5)
            .post_init(setup_commands)
            .post_shutdown(shutdown_app)
            .build()
        )
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("privacy", privacy_command))
        app.add_handler(CommandHandler("delete_me", delete_my_data))
        app.add_handler(CallbackQueryHandler(check_subscription_callback, pattern=f"^{CHECK_SUBSCRIPTION_CALLBACK}$"))
        app.add_handler(CallbackQueryHandler(retry_processing_callback, pattern=f"^{RETRY_PROCESSING_CALLBACK}$"))
        app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
        app.add_error_handler(error_handler)

        logger.info("Bot started")
        app.run_polling(allowed_updates=[Update.MESSAGE, Update.CALLBACK_QUERY])
    finally:
        READY_FILE.unlink(missing_ok=True)
        HEARTBEAT_FILE.unlink(missing_ok=True)
        release_instance_lock()


if __name__ == "__main__":
    main()
