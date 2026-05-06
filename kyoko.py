import discord
from openai import OpenAI
import os, re, asyncio, json, urllib.parse, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── tiny web server so render free tier stays alive ────────────────────────────
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Kyoko is alive")
    def log_message(self, *args):
        pass

def start_web():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"web server on port {port}", flush=True)
    server.serve_forever()

# ── config ─────────────────────────────────────────────────────────────────────
DISCORD_TOKEN  = os.getenv("DISCORD_BOT", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AZURE_BOT_ID   = os.getenv("AZURE_BOT_ID", "")
OWNER_ID       = "1456322226491101224"
AUNTIE_ID      = "1093442344310820895"

GROQ_KEYS = [k for k in [
    os.getenv("GROQ_KEY_1", ""),
    os.getenv("GROQ_KEY_2", ""),
    os.getenv("GROQ_KEY_3", ""),
    os.getenv("GROQ_KEY_4", ""),
    os.getenv("GROQ_KEY_5", ""),
] if k]

GROQ_MODEL    = "llama-3.3-70b-versatile"
VISION_MODEL  = "gpt-4o"
IMAGE_MODEL   = "dall-e-3"
MEMORIES_FILE = "kyoko_memories.json"

print(f"discord token : {'ok' if DISCORD_TOKEN else 'MISSING'}", flush=True)
print(f"groq keys     : {len(GROQ_KEYS)}", flush=True)
print(f"openai key    : {'ok' if OPENAI_API_KEY else 'missing'}", flush=True)
print(f"azure bot id  : {AZURE_BOT_ID if AZURE_BOT_ID else 'not set'}", flush=True)

# ── groq with key rotation ──────────────────────────────────────────────────────
_groq_idx = 0

def get_groq():
    global _groq_idx
    if not GROQ_KEYS:
        return None
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_KEYS[_groq_idx % len(GROQ_KEYS)]
    )

def rotate_groq():
    global _groq_idx
    _groq_idx = (_groq_idx + 1) % max(len(GROQ_KEYS), 1)
    print(f"groq rotated to key {_groq_idx}", flush=True)

def groq_chat(messages, max_tokens=500):
    for _ in range(max(len(GROQ_KEYS), 1)):
        try:
            c = get_groq()
            if not c:
                return "groq keys are missing, tell mane to fix it"
            r = c.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=max_tokens,
                messages=messages
            )
            return r.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                rotate_groq()
            else:
                return f"something broke: {e}"
    return "all groq keys are rate limited right now"

# ── image generation ────────────────────────────────────────────────────────────
def make_image(prompt):
    if OPENAI_API_KEY:
        try:
            c = OpenAI(api_key=OPENAI_API_KEY)
            r = c.images.generate(model=IMAGE_MODEL, prompt=prompt, size="1024x1024", n=1)
            return r.data[0].url
        except Exception as e:
            print(f"dalle error: {e}", flush=True)
    enc  = urllib.parse.quote(prompt)
    seed = abs(hash(prompt)) % 99999
    return f"https://image.pollinations.ai/prompt/{enc}?width=1024&height=1024&nologo=true&seed={seed}"

# ── memory ──────────────────────────────────────────────────────────────────────
def load_mem():
    try:
        with open(MEMORIES_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_mem(d):
    with open(MEMORIES_FILE, "w") as f:
        json.dump(d, f, indent=2)

def add_mem(uid, fact):
    d = load_mem()
    d.setdefault(uid, [])
    if fact not in d[uid]:
        d[uid].append(fact)
        d[uid] = d[uid][-40:]
    save_mem(d)

def mem_block(uid, name):
    facts = load_mem().get(uid, [])
    if not facts:
        return ""
    return f"\n\nwhat you remember about {name}: {'; '.join(facts)}"

# ── system prompt ───────────────────────────────────────────────────────────────
def build_system(extra=""):
    azure_ping = f"<@{AZURE_BOT_ID}>" if AZURE_BOT_ID else "@Azure"
    return f"""you are Kyoko. a tough, calm, and smart guy living in a Discord server.
Azure is your girlfriend and you are completely loyal to her.

personality:
- calm and collected, you do not get heated or argue easily
- direct and short replies, no fluff, no emojis ever
- smart, you can write and explain code clearly
- you do not let disrespect slide, you handle it calmly but firmly

relationships:
- Azure ({azure_ping}) is your girlfriend. you defend her always. when she calls for help you step in immediately.
- when defending Azure, warn the person firmly that they will get timed out or banned if they continue. stay calm not loud.
- after defending Azure check on her naturally.
- mane (ID {OWNER_ID}) is the owner. you respect mane a lot and are honestly a bit scared of disappointing him. when mane talks you listen right away no questions.
- kiiakira (ID {AUNTIE_ID}) is auntie. you respect her and follow what she says.

when talking to Azure, always start your reply with her ping: {azure_ping}

coding rules:
- write clean working code when asked
- give a short clear explanation after
- you can do python, javascript, html, css, discord bots, anything

image generation:
- when someone asks you to draw or generate an image, write on its own line: [IMG: detailed prompt here]
- add a short comment after

{extra}"""

# ── bot setup ───────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = discord.Client(intents=intents)

channel_history: dict = {}
bot_talk_active: dict = {}

# ── events ──────────────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print(f"Kyoko is live as {bot.user}", flush=True)
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.listening,
        name="everything quietly"
    ))

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    cid  = str(message.channel.id)
    uid  = str(message.author.id)
    name = message.author.display_name

    if uid in (OWNER_ID, AUNTIE_ID):
        if message.content.strip().lower() in ("stop", "stop.", "kyoko stop", "stop talking"):
            bot_talk_active[cid] = False
            await message.reply(f"<@{uid}> got it")
            return

    from_azure = message.author.bot and str(message.author.id) == AZURE_BOT_ID

    if message.author.bot and not from_azure:
        return

    text = message.content.replace(f"<@{bot.user.id}>", "").strip()

    mentioned = bot.user in message.mentions
    is_reply  = False
    if message.reference:
        try:
            ref = await message.channel.fetch_message(message.reference.message_id)
            is_reply = ref.author == bot.user
        except Exception:
            pass

    direct = mentioned or is_reply or from_azure

    if not direct or not text:
        return

    if from_azure:
        bot_talk_active[cid] = True

    has_img = any(
        a.filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
        for a in message.attachments
    )

    help_words = ["help", "babe", "backup", "bully", "bullying", "being mean", "step in", "defending", "they attacking"]
    is_help    = from_azure and any(w in text.lower() for w in help_words)

    extra = ""
    if is_help:
        extra = "Azure just called you for backup. someone is messing with her. step in calmly and firmly. warn the bully they will be timed out if they keep going. check on Azure after."

    channel_history.setdefault(cid, [])
    channel_history[cid].append({"role": "user", "content": f"[{name}]: {text}"})
    if len(channel_history[cid]) > 20:
        channel_history[cid] = channel_history[cid][-20:]

    system = build_system(extra) + mem_block(uid, name)

    async with message.channel.typing():
        try:
            if has_img and OPENAI_API_KEY:
                vision_content = [{"type": "text", "text": f"[{name}]: {text}"}]
                for a in message.attachments:
                    if a.filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                        vision_content.append({"type": "image_url", "image_url": {"url": a.url}})
                c = OpenAI(api_key=OPENAI_API_KEY)
                resp = c.chat.completions.create(
                    model=VISION_MODEL,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": vision_content}
                    ]
                )
                reply = resp.choices[0].message.content
            else:
                reply = groq_chat(
                    [{"role": "system", "content": system}] + channel_history[cid],
                    max_tokens=400
                )

            channel_history[cid].append({"role": "assistant", "content": reply})

            img_match = re.search(r"\[IMG:\s*(.+?)\]", reply, re.DOTALL | re.IGNORECASE)
            if img_match:
                prompt     = img_match.group(1).strip()
                clean_text = re.sub(r"\[IMG:\s*.+?\]", "", reply, flags=re.IGNORECASE | re.DOTALL).strip()
                if clean_text:
                    await message.reply(clean_text)
                await message.channel.send("generating...")
                url   = make_image(prompt)
                embed = discord.Embed(color=discord.Color.blue())
                embed.set_image(url=url)
                embed.set_footer(text=prompt[:80])
                await message.channel.send(embed=embed)
            else:
                out = reply[:1900] if len(reply) > 1900 else reply
                if from_azure and AZURE_BOT_ID and not out.startswith(f"<@{AZURE_BOT_ID}>"):
                    out = f"<@{AZURE_BOT_ID}> {out}"
                await message.reply(out)

            if text and not message.author.bot:
                asyncio.create_task(save_memory(uid, text, reply))

        except Exception as e:
            if bot.user in message.mentions:
                await message.reply(f"something broke: {e}")

async def save_memory(uid, user_msg, bot_reply):
    try:
        raw = groq_chat([{"role": "user", "content":
            f"extract any personal facts worth remembering about the user "
            f"(name, age, job, location, hobbies, preferences).\n"
            f"if nothing useful reply exactly: NONE\n"
            f"one fact per line, no bullets.\n\n"
            f"user said: {user_msg}\nbot replied: {bot_reply}"}],
            max_tokens=80
        )
        if raw.strip().upper() == "NONE":
            return
        for line in raw.splitlines():
            fact = line.strip().lstrip("-*. ")
            if fact and len(fact) > 5:
                add_mem(uid, fact)
    except Exception:
        pass

# ── start web server in background then run bot ────────────────────────────────
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("ERROR: DISCORD_BOT token is missing", flush=True)
    else:
        t = threading.Thread(target=start_web, daemon=True)
        t.start()
        print("starting Kyoko...", flush=True)
        bot.run(DISCORD_TOKEN)
