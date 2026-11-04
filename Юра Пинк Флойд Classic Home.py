import sys
import os
import sounddevice as sd
import speech_recognition as sr
import json
import datetime
import time
import numpy as np
import scipy.io.wavfile as wav
import sqlite3
import winsound
import re
import threading
import requests
import urllib.request
import urllib.parse
import numexpr as ne
from datetime import datetime as dt_datetime, timedelta
from deep_translator import GoogleTranslator
from gigachat import GigaChat
from silero_tts.silero_tts import SileroTTS
from dotenv import load_dotenv

# --- НАСТРОЙКА GIGACHAT ---
load_dotenv()

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

try:
    giga_client = GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS"
    )
except Exception as e:
    print(f"[КРИТИЧЕСКАЯ ОШИБКА] Не удалось настроить GigaChat: {e}")
    sys.exit()

# ИМЯ ПОЛЬЗОВАТЕЛЯ
USER_NAME = "Юрочка"

# --- СПИСОК ИМЕН АССИСТЕНТА (ТОЛЬКО АСТРАЛ) ---
NAME_VARIANTS = [
    "астрал", "астрала", "астралу", "астралы", "астрале", "астр", "астра"
]

CMD_VARIATIONS = {
    "time": ["время", "час", "сколько времени", "который час", "текущее время", "время сейчас"],
    "weather": ["погода", "погоду", "какая погода", "что с погодой", "прогноз погоды", "погода на улице"],
    "clear_history": ["забудь историю", "очисти чаты", "удали историю", "сотри наши разговоры", "очисти память"],
    "show_facts": ["что ты помнишь", "что ты знаешь обо мне", "расскажи обо мне", "моя память", "что помнишь"],
    "exit": ["пока", "выключись", "отключись", "до свидания", "заверши работу", "стоп", "спокойной ночи"],
    "timer": ["таймер", "поставь таймер", "засеки", "таймер на", "засеки время"],
    "stopwatch": ["секундомер", "запусти секундомер", "включи секундомер"],
    "stopwatch_stop": ["останови секундомер", "стоп секундомер", "выключи секундомер"],
    "cancel_timer": ["отмени таймер", "убери таймер", "отмена таймера", "сбрось таймер"],
    "alarm": ["будильник", "поставь будильник", "разбуди меня", "будильник на", "поставь будильник на"],
    "radio": ["включи радио", "радио", "включи волну", "включи станцию", "радиостанция"],
    "joke": ["расскажи анекдот", "расскажи шутку", "пошути", "анекдот", "рассмеши меня"],
    "fact": ["расскажи факт", "интересный факт", "расскажи интересный факт", "факт", "что-нибудь интересное"]
}

_active_timers = []
_stopwatch_start = None
_stopwatch_running = False
_alarm_playing = False

# --- УСКОРЕНИЕ РЕАКЦИИ ---
SILENCE_LIMIT = 0.8  # Было 1.5, теперь он быстрее понимает, что ты замолчал
THRESHOLD = 0.06
MIN_SPEECH_DURATION = 0.3


def check_command(text, command_key):
    for variation in CMD_VARIATIONS.get(command_key, []):
        if variation in text:
            return True
    return False


def contains_name(text):
    for variant in NAME_VARIANTS:
        if variant in text:
            return True
    return False


def strip_name(text):
    for variant in NAME_VARIANTS:
        text = text.replace(variant, "")
    return text.strip()


def num_to_words_text(n):
    units = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    teens = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
             "семнадцать", "восемнадцать", "девятнадцать"]
    tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
    hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
    if n < 10: return units[n]
    if n < 20: return teens[n - 10]
    if n < 100: return tens[n // 10] + (" " + units[n % 10] if n % 10 else "")
    if n < 1000: return hundreds[n // 100] + (" " + num_to_words_text(n % 100) if n % 100 else "")
    return str(n)


def replace_digits(text):
    if any(op in text for op in ['+', '-', '*', '/', 'плюс', 'минус', 'умножить', 'разделить']):
        return text

    def replace_match(match):
        num_str = match.group(0)
        if num_str == "0": return "ноль"
        return num_to_words_text(int(num_str))

    return re.sub(r'\b\d{1,3}\b', replace_match, text)


# --- ГЕНЕРАТОР ГРОМКОГО ЗВУКА БУДИЛЬНИКА ---
def generate_alarm_sound(filename="alarm.wav"):
    if os.path.exists(filename):
        return filename
    try:
        sample_rate = 44100
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(t * 800 * 2 * np.pi)
        tone = np.sign(tone)
        envelope = np.sin(t * 10 * 2 * np.pi) > 0
        sound = tone * envelope
        sound = sound * 32767
        wav.write(filename, sample_rate, sound.astype(np.int16))
        return filename
    except Exception as e:
        print(f"[ОШИБКА ГЕНЕРАЦИИ ЗВУКА] {e}")
        return None


print("[ГОЛОС] Загружаю модель синтеза речи (Silero TTS)...")
try:
    tts_engine = SileroTTS(model_id='v4_ru', language='ru', speaker='aidar', sample_rate=24000, device='cpu')
except Exception as e:
    print(f"[ОЗВУЧКА ОШИБКА] {e}")
    sys.exit()

MY_CITY = "Уссурийск"
DB_FILE = "floyd_chat.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_name TEXT NOT NULL, role TEXT NOT NULL, message TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_name TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_name, key))')
    conn.commit()
    conn.close()


def save_chat_to_db(user_name, role, message):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_name, role, message) VALUES (?, ?, ?)", (user_name, role, message))
    conn.commit()
    conn.close()


def get_recent_chats(user_name, limit=30):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM chat_history WHERE user_name = ? ORDER BY id DESC LIMIT ?",
                   (user_name, limit))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


def search_history(user_name, query):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT message FROM chat_history WHERE user_name = ? AND message LIKE ? ORDER BY id DESC LIMIT 5",
                   (user_name, f"%{query}%"))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def remember_fact(user_name, key, value):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO facts (user_name, key, value) VALUES (?, ?, ?) ON CONFLICT(user_name, key) DO UPDATE SET value=excluded.value',
        (user_name, key.strip().lower(), value.strip()))
    conn.commit()
    conn.close()


def get_all_facts(user_name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT key, value FROM facts WHERE user_name = ?", (user_name,))
    rows = cursor.fetchall()
    conn.close()
    return dict(rows)


def get_facts_for_ai(user_name):
    facts = get_all_facts(user_name)
    if not facts: return ""
    lines = [f"Известные факты о пользователе по имени {user_name}:"]
    for k, v in facts.items(): lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def speak(text):
    clean_text = re.sub(r'[*#$`]', '', text)
    clean_text = clean_text.replace('\\times', 'умножить на')
    clean_text = clean_text.replace('\\div', 'разделить на')

    print(f"Астрал: {clean_text}")
    temp_file = "temp_tts.wav"
    try:
        tts_engine.tts(clean_text, temp_file)
        time.sleep(0.1)
        # Асинхронное воспроизведение (не блокирует программу)
        winsound.PlaySound(temp_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print(f"[ОЗВУЧКА ОШИБКА] {e}")


def configure_audio_devices():
    devices = sd.query_devices()
    default_devices = sd.default.device
    input_idx = int(default_devices[0]) if isinstance(default_devices, (tuple, list)) or 'Pair' in type(
        default_devices).__name__ else 0
    output_idx = None
    for idx, dev in enumerate(devices):
        name = dev['name'].lower()
        if dev['max_output_channels'] > 0 and any(w in name for w in ['headphone', 'наушники']):
            output_idx = idx;
            break
    if output_idx is None:
        for idx, dev in enumerate(devices):
            name = dev['name'].lower()
            if dev['max_output_channels'] > 0 and any(w in name for w in ['speaker', 'динамики']):
                output_idx = idx;
                break
    if output_idx is None:
        output_idx = int(default_devices[1]) if isinstance(default_devices, (tuple, list)) or 'Pair' in type(
            default_devices).__name__ else 1
    sd.default.device = (input_idx, output_idx)
    samplerate = int(devices[input_idx]['default_samplerate'])
    return samplerate, input_idx


samplerate, input_device_id = configure_audio_devices()
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
recognizer.energy_threshold = 300

init_db()
remember_fact(USER_NAME, "имя", "Юрочка")


def listen_phrase():
    global _alarm_playing
    audio_buffer, speaking, silence_start, speech_start_time = [], False, None, None
    stream = sd.InputStream(samplerate=samplerate, channels=1, device=input_device_id)
    stream.start()
    while True:
        indata, overflowed = stream.read(1000)
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))

        if _alarm_playing:
            time.sleep(0.1)
            continue

        if volume_norm > THRESHOLD:
            if not speaking:
                speaking, speech_start_time = True, time.time()
                print("[ЗАПИСЬ] Слушаю вас...")
            audio_buffer.append(indata.copy())
            silence_start = None
        elif speaking:
            audio_buffer.append(indata.copy())
            if silence_start is None: silence_start = time.time()

        if speaking and silence_start and (time.time() - silence_start > SILENCE_LIMIT):
            if speech_start_time and (time.time() - speech_start_time) >= MIN_SPEECH_DURATION:
                break
            else:
                audio_buffer, speaking, silence_start, speech_start_time = [], False, None, None
        if not speaking: time.sleep(0.05)

    stream.stop()
    stream.close()
    if not audio_buffer: return None
    audio_np = np.concatenate(audio_buffer)

    # Удаляем старый файл перед записью нового
    try:
        if os.path.exists("test_phrase.wav"):
            os.remove("test_phrase.wav")
    except:
        pass

    wav.write("test_phrase.wav", samplerate, (audio_np * 32767).astype(np.int16))
    return "test_phrase.wav"


def get_weather(city=None):
    city_to_check = city if city else MY_CITY
    print(f"[СИСТЕМА] Проверяю погоду в {city_to_check}...")
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city_to_check)}?format=j1"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            current = data['current_condition'][0]
            temp = int(current['temp_C'])
            desc = current.get('lang_ru', [{}])[0].get('value', current['weatherDesc'][0]['value'])

            city_name = data['nearest_area'][0]['areaName'][0]['value']
            if city_name.lower().endswith("ск"):
                city_prep = city_name[:-2] + "ске"
            elif city_name.lower().endswith("а"):
                city_prep = city_name[:-1] + "е"
            elif city_name.lower().endswith("ь"):
                city_prep = city_name[:-1] + "и"
            else:
                city_prep = city_name + "е"

            if temp % 10 == 1 and temp != 11:
                t_word = "градус"
            elif temp % 10 in [2, 3, 4] and temp not in [12, 13, 14]:
                t_word = "градуса"
            else:
                t_word = "градусов"

            return f"Сейчас в {city_prep} около {temp} {t_word}, {desc}."
    except Exception as e:
        return f"Не удалось узнать погоду в {city_to_check}."


def get_location_by_ip():
    try:
        with urllib.request.urlopen("http://ip-api.com/json", timeout=5) as response:
            data = json.loads(response.read().decode())
            return data.get("city", MY_CITY)
    except:
        return MY_CITY


def search_wikipedia(query):
    try:
        url = f"https://ru.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            extract = data.get('extract', '')
            if extract:
                return extract[:500]
            return "Не нашел информацию."
    except Exception:
        return "Не удалось найти в Википедии."


def calculate_expression(expr):
    try:
        expr = expr.replace("плюс", "+").replace("минус", "-").replace("умножить на", "*").replace("умножить",
                                                                                                   "*").replace(
            "разделить на", "/").replace("разделить", "/")
        expr = expr.replace(" ", "")
        if not expr:
            return "Не понял пример."
        result = ne.evaluate(expr)
        if isinstance(result, (int, float)) and result == int(result):
            return f"Ответ: {int(result)}"
        return f"Ответ: {result}"
    except Exception:
        return "Не смог посчитать."


def translate_text(text, target_lang="en"):
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        result = translator.translate(text)
        return f"Перевод: {result}"
    except Exception:
        return "Не удалось перевести."


def ask_online_ai(user_name, user_text):
    try:
        facts_block = get_facts_for_ai(user_name)
        system_prompt = (
            f"Тебя зовут Астрал. Ты голосовой помощник. Сейчас с тобой общается {user_name}. "
            "Отвечай на русском языке. "
            "Если тебя просят посчитать (сколько будет...), назови ТОЛЬКО точную цифру. "
            "Если просят рассказать факт, расскажи ОДИН факт. "
            "Если просят рассказать анекдот, расскажи короткий анекдот. "
            "Если это светская беседа (привет, как дела), отвечай ОЧЕНЬ КОРОТКО (одно предложение). "
            "На все остальные вопросы (рецепты, объяснения) отвечай ПОДРОБНО и РАЗВЕРНУТО."
        )
        if facts_block: system_prompt += "\n\n" + facts_block
        messages = [{"role": "system", "content": system_prompt}] + get_recent_chats(user_name, 30) + [
            {"role": "user", "content": user_text}]
        response = giga_client.chat({"messages": messages, "model": "GigaChat-2"})
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ОШИБКА GIGACHAT] {e}")
        return "Извините, произошла ошибка."


def parse_duration(text):
    word_to_num = {"один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5, "десять": 10, "пятнадцать": 15,
                   "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "полчаса": 30}
    for word, num in word_to_num.items(): text = re.sub(rf'\b{word}\b', str(num), text)
    total = 0
    for pattern, mult in [(r'(\d+)\s*(час|часа|часов)', 3600), (r'(\d+)\s*(минут|минуты|мин)', 60),
                          (r'(\d+)\s*(секунд|сек)', 1)]:
        match = re.search(pattern, text)
        if match: total += float(match.group(1)) * mult
    if total == 0:
        bare = re.search(r'(\d+)', text)
        if bare: total = float(bare.group(1))
    return int(total)


def _timer_fire(msg, user_name):
    try:
        alarm_file = generate_alarm_sound()
        if alarm_file:
            winsound.PlaySound(alarm_file, winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)
            time.sleep(5)
            winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as e:
        print(f"[ОШИБКА ЗВУКА] {e}")
    speak(msg)
    save_chat_to_db(user_name, "assistant", msg)


def set_timer(seconds, user_name, label="Таймер"):
    if seconds <= 0: return speak("Не понял длительность.")
    t = threading.Timer(seconds, _timer_fire, args=(f"{label} сработал!", user_name))
    t.daemon = True
    t.start()
    _active_timers.append(t)
    speak(f"{label} установлен.")


def play_alarm_sound():
    global _alarm_playing
    _alarm_playing = True
    try:
        alarm_file = generate_alarm_sound()
        if alarm_file:
            winsound.PlaySound(alarm_file, winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)
            time.sleep(10)
            winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as e:
        print(f"[ОШИБКА БУДИЛЬНИКА] {e}")
    _alarm_playing = False


def set_alarm(alarm_time, user_name, label="Будильник"):
    try:
        now = datetime.datetime.now()
        target = datetime.datetime.strptime(alarm_time, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
        if target < now:
            target += datetime.timedelta(days=1)
        delay = (target - now).total_seconds()

        def alarm_fire():
            play_alarm_sound()
            speak(f"Просыпайся, {user_name}! Будильник сработал!")
            save_chat_to_db(user_name, "assistant", "Будильник сработал!")

        t = threading.Timer(delay, alarm_fire)
        t.daemon = True
        t.start()
        _active_timers.append(t)
        speak(f"Будильник установлен на {alarm_time}.")
        return True
    except Exception as e:
        print(f"[ОШИБКА УСТАНОВКИ БУДИЛЬНИКА] {e}")
        speak("Не понял время. Скажите, например: 'будильник на 7:30'.")
        return False


def start_stopwatch():
    global _stopwatch_start, _stopwatch_running
    _stopwatch_start, _stopwatch_running = time.time(), True
    speak("Секундомер запущен.")


def stop_stopwatch():
    global _stopwatch_start, _stopwatch_running
    if not _stopwatch_running: return speak("Секундомер не запущен.")
    elapsed = time.time() - _stopwatch_start
    _stopwatch_running = False
    speak(f"Прошло {int(elapsed)} секунд.")


def cancel_timers():
    global _active_timers
    count = 0
    for t in _active_timers:
        if t.is_alive(): t.cancel(); count += 1
    _active_timers = []
    speak(f"Отменено: {count}." if count else "Активных нет.")


def handle_command(statement):
    if not statement: return True

    if not contains_name(statement):
        if "стоп" in statement:
            speak("До встречи, Юрочка!")
            return False
        print("[ФИЛЬТР] Имя не найдено. Игнорирую.")
        return True

    clean_statement = strip_name(statement)

    if not clean_statement:
        speak("Да, Юрочка, я слушаю.")
        return True

    statement = clean_statement

    if "привет" in statement:
        speak("Приветствую, Юрочка! Чем могу помочь?")
        return True
    if "как дела" in statement:
        speak("Все системы работают идеально. А у вас как настроение, Юрочка?")
        return True
    if "как меня зовут" in statement:
        speak("Вас зовут Юрочка.")
        return True

    if "кто тебя создал" in statement or "кто твой создатель" in statement or "кто тебя сделал" in statement:
        speak("Меня создал великий владыка, Лжедмитрий Первый!")
        return True

    if check_command(statement, "time"):
        now = datetime.datetime.now()
        h_word = num_to_words_text(now.hour)
        m_word = num_to_words_text(now.minute) if now.minute > 0 else ""
        if now.minute == 0:
            speak(f"Сейчас {h_word} ровно.")
        else:
            speak(f"Сейчас {h_word} {m_word}.")
        return True

    if check_command(statement, "weather"):
        city_match = re.search(r'погода\s+в\s+([а-яё]+)', statement)
        if city_match:
            speak(get_weather(city_match.group(1).capitalize()))
        else:
            detected_city = get_location_by_ip()
            speak(get_weather(detected_city))
        return True

    if check_command(statement, "timer"):
        set_timer(parse_duration(statement), USER_NAME)
        return True
    if check_command(statement, "stopwatch"):
        start_stopwatch()
        return True
    if check_command(statement, "stopwatch_stop"):
        stop_stopwatch()
        return True
    if check_command(statement, "cancel_timer"):
        cancel_timers()
        return True

    if check_command(statement, "alarm"):
        time_match = re.search(r'(\d{1,2})[:.\s](\d{2})', statement)
        if time_match:
            alarm_time = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
            set_alarm(alarm_time, USER_NAME)
        else:
            speak("Скажите время, например: будильник на 7:30.")
        return True

    if "сколько будет" in statement:
        expr = statement.replace("сколько будет", "").strip()
        speak(calculate_expression(expr))
        return True

    if "переведи на английский" in statement or "переведи на русский" in statement:
        target = "en" if "на английский" in statement else "ru"
        text_to_translate = statement.split("переведи на")[1].split(" ", 1)[1].strip() if " " in statement.split(
            "переведи на")[1] else ""
        if text_to_translate:
            speak(translate_text(text_to_translate, target))
        else:
            speak("Что перевести?")
        return True

    if "кто такой" in statement or "кто такая" in statement or "что такое" in statement:
        query = statement.replace("кто такой", "").replace("кто такая", "").replace("что такое", "").strip()
        if query:
            wiki_answer = search_wikipedia(query)
            if wiki_answer and "не нашел" not in wiki_answer.lower():
                speak(wiki_answer)
                return True

    if "что я говорил про" in statement:
        query = statement.replace("что я говорил про", "").strip()
        if query:
            history = search_history(USER_NAME, query)
            if history:
                speak(f"Вы говорили: {history[0]}")
            else:
                speak("Я не нашел такого в истории.")
            return True

    if check_command(statement, "exit"):
        speak("До встречи, Юрочка!")
        return False

    save_chat_to_db(USER_NAME, "user", statement)
    ai_response = ask_online_ai(USER_NAME, statement)
    speak(replace_digits(ai_response))
    save_chat_to_db(USER_NAME, "assistant", ai_response)
    return True


speak("Привет! Я Астрал. Я готов помогать тебе, Юрочка.")
print("\nАстрал на связи. Скажите 'Астрал' и команду...")

while True:
    audio_file = listen_phrase()
    if not audio_file: continue
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        statement = recognizer.recognize_google(audio_data, language="ru-RU").lower()
    except sr.UnknownValueError:
        continue
    except sr.RequestError:
        continue

    print(f"\n[СЫРОЙ ТЕКСТ]: '{statement}'")
    if not handle_command(statement): break
