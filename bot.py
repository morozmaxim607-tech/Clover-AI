import base64
import logging
import os
import sqlite3
import time
from collections import defaultdict
from datetime import date, datetime, timezone

from anthropic import AsyncAnthropic
from telegram import LabeledPrice, Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PreCheckoutQueryHandler,
    filters,
)

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("claude-bot")

# ---------- настройки ----------
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # нужен только для голосовых
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))  # ваш Telegram ID: безлимит + /refund
DB_PATH = os.getenv("DB_PATH", "bot.db")

FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", "10"))
PREMIUM_DAILY_LIMIT = int(os.getenv("PREMIUM_DAILY_LIMIT", "200"))
PREMIUM_PRICE_STARS = int(os.getenv("PREMIUM_PRICE_STARS", "250"))
PREMIUM_DAYS = int(os.getenv("PREMIUM_DAYS", "30"))
MAX_HISTORY = 30

TOOLS = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]

client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
stt = AsyncOpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and AsyncOpenAI) else None
histories: dict[int, list[dict]] = defaultdict(list)

# ---------- база: премиум и лимиты хранятся на диске ----------
db = sqlite3.connect(DB_PATH)
db.executescript(
    """
    CREATE TABLE IF NOT EXISTS users(
        user_id INTEGER PRIMARY KEY,
        premium_until INTEGER DEFAULT 0,
        day TEXT DEFAULT '',
        used INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS payments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        stars INTEGER,
        charge_id TEXT,
        ts INTEGER
    );
    """
)
db.commit()


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_user(uid: int):
    row = db.execute(
        "SELECT premium_until, day, used FROM users WHERE user_id=?", (uid,)
    ).fetchone()
    if row is None:
        db.execute("INSERT INTO users(user_id) VALUES(?)", (uid,))
        db.commit()
        return 0, today(), 0
    premium_until, day, used = row
    if day != today():
        used = 0
    return premium_until, today(), used


def is_premium(uid: int) -> bool:
    return get_user(uid)[0] > time.time()


def daily_limit(uid: int):
    if uid == ADMIN_ID:
        return None  # без лимита
    return PREMIUM_DAILY_LIMIT if is_premium(uid) else FREE_DAILY_LIMIT


def can_use(uid: int) -> bool:
    limit = daily_limit(uid)
    return limit is None or get_user(uid)[2] < limit


def add_use(uid: int):
    _, day, used = get_user(uid)
    db.execute("UPDATE users SET day=?, used=? WHERE user_id=?", (day, used + 1, uid))
    db.commit()


def grant_premium(uid: int, days: int):
    until, _, _ = get_user(uid)
    start = max(int(time.time()), until)  # продление прибавляется к остатку
    new_until = start + days * 86400
    db.execute("UPDATE users SET premium_until=? WHERE user_id=?", (new_until, uid))
    db.commit()
    return new_until


def fmt_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%d.%m.%Y")


async def deny_if_limit(update: Update) -> bool:
    uid = update.effective_user.id
    if can_use(uid):
        return False
    if is_premium(uid):
        text = f"Дневной лимит Premium ({PREMIUM_DAILY_LIMIT}) исчерпан. Он обновится завтра (по UTC)."
    else:
        text = (
            f"Бесплатный лимит на сегодня ({FREE_DAILY_LIMIT} запросов) исчерпан.\n"
            f"/premium: {PREMIUM_DAILY_LIMIT} запросов в день на {PREMIUM_DAYS} дней "
            f"за {PREMIUM_PRICE_STARS} ⭐. Или подождите до завтра."
        )
    await update.message.reply_text(text)
    return True


# ---------- Claude ----------
def system_prompt() -> str:
    return (
        "Ты Claude, ИИ-ассистент от Anthropic. Ты общаешься с человеком в Telegram. "
        "Отвечай на том языке, на котором пишет человек. Будь тёплым, честным и полезным. "
        "Начинай сразу с ответа, без воды. Ответы делай компактными: экран телефона маленький. "
        "Не используй markdown-разметку: Telegram показывает её как обычный текст. "
        "Если вопрос касается свежих событий, цен, погоды или того, что могло измениться, "
        "сначала поищи в интернете. "
        f"Сегодняшняя дата: {date.today().isoformat()}."
    )


def split_text(text: str, limit: int = 4000):
    while text:
        if len(text) <= limit:
            yield text
            return
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        yield text[:cut]
        text = text[cut:].lstrip("\n")


async def ask_claude(update: Update, context: ContextTypes.DEFAULT_TYPE, content, note: str):
    chat_id = update.effective_chat.id
    uid = update.effective_user.id
    history = histories[chat_id]
    history.append({"role": "user", "content": content})
    del history[:-MAX_HISTORY]
    while history and history[0]["role"] != "user":
        history.pop(0)

    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        messages = list(history)
        for _ in range(3):
            response = await client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=system_prompt(),
                messages=messages,
                tools=TOOLS,
            )
            if response.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": response.content})
        answer = "".join(b.text for b in response.content if b.type == "text").strip()
        if not answer:
            answer = "Не получилось сформулировать ответ, попробуйте ещё раз."
    except Exception:
        log.exception("Ошибка API")
        history.pop()
        await update.message.reply_text("Что-то пошло не так, попробуйте ещё раз чуть позже.")
        return

    add_use(uid)  # лимит списываем только за успешный ответ
    history[-1] = {"role": "user", "content": note}
    history.append({"role": "assistant", "content": answer})
    for part in split_text(answer):
        await update.message.reply_text(part)


# ---------- команды ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я Claude. Пиши текстом, присылай фото или голосовые.\n"
        "/status: твой тариф и лимит\n"
        "/premium: купить Premium за звёзды\n"
        "/reset: очистить историю диалога"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    histories.pop(update.effective_chat.id, None)
    await update.message.reply_text("История очищена.")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    until, _, used = get_user(uid)
    limit = daily_limit(uid)
    if uid == ADMIN_ID:
        text = "Тариф: админ, без лимита."
    elif until > time.time():
        text = f"Тариф: Premium до {fmt_date(until)}.\nСегодня: {used}/{limit}."
    else:
        text = f"Тариф: бесплатный.\nСегодня: {used}/{limit}.\nУвеличить лимит: /premium"
    await update.message.reply_text(text)


async def premium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_invoice(
        chat_id=update.effective_chat.id,
        title="Claude Premium",
        description=(
            f"{PREMIUM_DAYS} дней, до {PREMIUM_DAILY_LIMIT} запросов в день. "
            "Повторная покупка продлевает срок."
        ),
        payload=f"premium_{PREMIUM_DAYS}d",
        provider_token="",  # для Telegram Stars токен провайдера пустой
        currency="XTR",
        prices=[LabeledPrice(f"Premium на {PREMIUM_DAYS} дней", PREMIUM_PRICE_STARS)],
    )


async def paysupport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Telegram требует, чтобы бот с платежами отвечал на /paysupport
    await update.message.reply_text(
        "Вопросы по оплате и возврату звёзд: напишите администратору бота, "
        "укажите дату платежа и сумму."
    )


async def pre_checkout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.pre_checkout_query
    ok = query.invoice_payload == f"premium_{PREMIUM_DAYS}d"
    await query.answer(ok=ok, error_message=None if ok else "Неизвестный товар.")


async def on_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pay = update.message.successful_payment
    uid = update.effective_user.id
    db.execute(
        "INSERT INTO payments(user_id, stars, charge_id, ts) VALUES(?,?,?,?)",
        (uid, pay.total_amount, pay.telegram_payment_charge_id, int(time.time())),
    )
    db.commit()
    until = grant_premium(uid, PREMIUM_DAYS)
    await update.message.reply_text(
        f"Спасибо! Premium активен до {fmt_date(until)}. Лимит: {PREMIUM_DAILY_LIMIT} запросов в день."
    )


async def refund(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Только для админа: /refund <user_id> <charge_id>"""
    if update.effective_user.id != ADMIN_ID:
        return
    try:
        user_id, charge_id = int(context.args[0]), context.args[1]
        await context.bot.refund_star_payment(user_id, charge_id)
        db.execute("UPDATE users SET premium_until=? WHERE user_id=?", (int(time.time()), user_id))
        db.commit()
        await update.message.reply_text("Возврат выполнен, Premium снят.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}\nФормат: /refund <user_id> <charge_id>")


# ---------- сообщения ----------
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await deny_if_limit(update):
        return
    text = update.message.text
    await ask_claude(update, context, text, text)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await deny_if_limit(update):
        return
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    photo = update.message.photo[-1]
    tg_file = await photo.get_file()
    data = bytes(await tg_file.download_as_bytearray())
    caption = update.message.caption or "Что на этом фото?"
    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(data).decode(),
            },
        },
        {"type": "text", "text": caption},
    ]
    await ask_claude(update, context, content, f"[пользователь прислал фото] {caption}")


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if stt is None:
        await update.message.reply_text(
            "Голосовые не настроены: нужен OPENAI_API_KEY и пакет openai для распознавания речи."
        )
        return
    if await deny_if_limit(update):
        return
    voice = update.message.voice
    if voice.duration and voice.duration > 300:
        await update.message.reply_text("Голосовое длиннее 5 минут, пришлите покороче.")
        return
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        tg_file = await voice.get_file()
        data = bytes(await tg_file.download_as_bytearray())
        result = await stt.audio.transcriptions.create(model="whisper-1", file=("voice.ogg", data))
        text = (result.text or "").strip()
    except Exception:
        log.exception("Ошибка распознавания")
        await update.message.reply_text("Не удалось распознать голосовое, попробуйте ещё раз.")
        return
    if not text:
        await update.message.reply_text("Не расслышал ничего в голосовом.")
        return
    await update.message.reply_text(f"Распознано: {text}")
    await ask_claude(update, context, text, text)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("premium", premium))
    app.add_handler(CommandHandler("paysupport", paysupport))
    app.add_handler(CommandHandler("refund", refund))
    app.add_handler(PreCheckoutQueryHandler(pre_checkout))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, on_payment))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.run_polling()


if __name__ == "__main__":
    main()
