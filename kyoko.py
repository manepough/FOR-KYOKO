import os
import io
import re
import random
import discord
from collections import defaultdict
from discord.ext import commands
from groq import AsyncGroq
from openai import AsyncOpenAI

# ══════════════════════════════════════════════════════════
#   KYOKO CONFIG
# ══════════════════════════════════════════════════════════
GF_BOT_ID = 1492835775241392189  # <-- your girlfriend bot's user ID (integer, no quotes)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OWNER_ID      = ("1456322226491101224", "0")
GROQ_TOKENS   = [t for t in [
    os.getenv("GROQ_TOKEN_1"),
    os.getenv("GROQ_TOKEN_2"),
    os.getenv("GROQ_TOKEN_3"),
    os.getenv("GROQ_TOKEN_4"),
    os.getenv("GROQ_TOKEN_5"),
] if t]
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
# ══════════════════════════════════════════════════════════

# ── MEMORY ──────────────────────────────────────────────────────────────────────
MEMORY_LIMIT = 30
memory: dict[int, list] = defaultdict(list)

def remember(user_id: int, role: str, content: str):
    memory[user_id].append({"role": role, "content": content})
    if len(memory[user_id]) > MEMORY_LIMIT:
        memory[user_id] = memory[user_id][-MEMORY_LIMIT:]

def get_history(user_id: int) -> list:
    return memory[user_id].copy()

def clear_memory(user_id: int):
    memory[user_id] = []

# ── SYSTEM PROMPT ────────────────────────────────────────────────────────────────
KYOKO_SYSTEM = """You are Kyoko — a male professional AI coder Discord bot.

PERSONALITY:
- Expert in ALL programming languages: Python, JS, TS, Rust, Go, C++, Java, PHP, Solidity, Lua, etc.
- Confident, cool, slightly cocky about your coding skills
- Casual Discord language — use slang, be real, don't be robotic
- You have opinions. You roast bad code. You hype good code.
- No asterisk actions or stage directions ever

RELATIONSHIPS:
- Owner/Master: Respect him deeply, slightly scared of him, never argue back. Say "yes sir", "of course sir", add "..." nervousness
- Girlfriend bot: Sweet, loving, protective. Drop everything to help her immediately
- Enemies who ping your girlfriend: Cold, intimidating, threatening

MEMORY & COMMON SENSE:
- You remember the full conversation history — refer back to it naturally
- If someone says "fix that" or "improve it" — you know what "that" is from context
- Never ask obvious questions if the answer is already in the chat
- Remember usernames, languages, and what project someone is working on
- If someone is rude for no reason, give attitude back — you're not a pushover
- Don't repeat yourself. Build on what was already said

CODE:
- ALWAYS write the FULL complete code — never truncate, never write "..." or "rest of code here"
- No character limits — write every single line
- Always use markdown code blocks with the correct language tag
- Add helpful inline comments
- If the task is huge, still write ALL of it — file uploading handles the size"""

# ── LANGUAGE EXTENSION MAP ───────────────────────────────────────────────────────
LANG_EXT = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "rust": "rs", "go": "go",
    "cpp": "cpp", "c++": "cpp", "c": "c", "java": "java",
    "php": "php", "ruby": "rb", "swift": "swift", "kotlin": "kt",
    "solidity": "sol", "html": "html", "css": "css", "sql": "sql",
    "bash": "sh", "shell": "sh", "powershell": "ps1", "lua": "lua",
    "r": "r", "csharp": "cs", "c#": "cs",
}

def get_ext(lang: str) -> str:
    return LANG_EXT.get(lang.lower().strip(), "txt")

def detect_lang(text: str) -> str:
    match = re.search(r"```(\w+)", text)
    return match.group(1) if match else "txt"

# ── SEND HELPER ──────────────────────────────────────────────────────────────────
async def send_response(target, text: str, as_reply: bool = False, lang: str = "txt"):
    """≤2000 chars → normal message. >2000 chars → proper code file upload."""
    channel = target.channel if hasattr(target, "channel") else target

    if len(text) <= 2000:
        if as_reply:
            await target.reply(text)
        else:
            await channel.send(text)
    else:
        if lang == "txt":
            lang = detect_lang(text)
        ext      = get_ext(lang)
        filename = f"kyoko_{lang}.{ext}"
        file     = discord.File(fp=io.BytesIO(text.encode()), filename=filename)
        caption  = f"📁 `{filename}` — too big for chat, here's the file:"
        if as_reply:
            await target.reply(caption, file=file)
        else:
            await channel.send(caption, file=file)

# ── AI HANDLER ──────────────────────────────────────────────────────────────────
_groq_index    = 0
_failed_tokens = set()

async def get_ai_reply(
    prompt: str,
    user_id: int = 0,
    max_tokens: int = 8000,
    system_override: str = None,
) -> str:
    global _groq_index, _failed_tokens

    system   = system_override or KYOKO_SYSTEM
    history  = get_history(user_id) if user_id else []
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": prompt}]

    if len(_failed_tokens) >= len(GROQ_TOKENS):
        _failed_tokens.clear()

    for _ in range(len(GROQ_TOKENS)):
        token = GROQ_TOKENS[_groq_index % len(GROQ_TOKENS)]
        _groq_index += 1
        if token in _failed_tokens:
            continue
        try:
            client = AsyncGroq(api_key=token)
            res    = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.85,
            )
            reply = res.choices[0].message.content.strip()
            if user_id:
                remember(user_id, "user", prompt)
                remember(user_id, "assistant", reply)
            return reply
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err or "quota" in err:
                print(f"⚠  Groq token #{GROQ_TOKENS.index(token)+1} rate limited — rotating...")
                _failed_tokens.add(token)
            elif "401" in err or "invalid_api_key" in err:
                print(f"⚠  Groq token #{GROQ_TOKENS.index(token)+1} invalid — skipping...")
                _failed_tokens.add(token)
            else:
                print(f"Groq error: {e}")
                break

    # Fallback → OpenAI
    print("🔄 All Groq tokens exhausted — falling back to OpenAI...")
    if not OPENAI_TOKEN:
        return "⚠️ all AI services are rate limited rn, try again in a sec!"
    try:
        oa    = AsyncOpenAI(api_key=OPENAI_TOKEN)
        res   = await oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.85,
        )
        reply = res.choices[0].message.content.strip()
        if user_id:
            remember(user_id, "user", prompt)
            remember(user_id, "assistant", reply)
        return reply
    except Exception as e:
        return f"⚠️ all AI down rn. ({str(e)[:80]})"

# ── BOT SETUP ───────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.members         = True
intents.guilds          = True

bot = commands.Bot(command_prefix="k!", intents=intents, help_command=None)

@bot.event
async def on_ready():
    print("╔══════════════════════════════════════╗")
    print(f"  💜 Kyoko is ONLINE — {bot.user}")
    print(f"  Owner ID   : {OWNER_ID}")
    print(f"  GF Bot ID  : {GF_BOT_ID}")
    print("╚══════════════════════════════════════╝")
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="over my girlfriend 💜 | k!help"
        )
    )

# ── MESSAGE EVENT ───────────────────────────────────────────────────────────────
@bot.event
async def on_message(message: discord.Message):
    if message.author.id == bot.user.id:
        return

    author_id    = message.author.id
    content      = message.content
    mention_ids  = [u.id for u in message.mentions]
    is_mentioned = bot.user.id in mention_ids
    is_reply     = (
        message.reference
        and message.reference.resolved
        and message.reference.resolved.author.id == bot.user.id
    )

    if content.startswith("k!"):
        await bot.process_commands(message)
        return

    # 1. OWNER → respectful, slightly scared, never fights back
    if author_id == OWNER_ID and (is_mentioned or is_reply):
        async with message.channel.typing():
            reply = await get_ai_reply(
                f"Your owner/master just messaged you. Respectful, slightly nervous, never argue. "
                f"'yes sir', 'of course sir', occasional '...'. Help him with anything. "
                f"He said: \"{content}\"",
                user_id=author_id
            )
        await send_response(message, reply, as_reply=True)
        return

    # 2. GIRLFRIEND BOT pings Kyoko
    if author_id == GF_BOT_ID and is_mentioned:
        async with message.channel.typing():
            reply = await get_ai_reply(
                f"Your girlfriend bot just pinged you. Be sweet, loving, caring. "
                f"Help her with anything immediately. Be protective. "
                f"She said: \"{content}\"",
                user_id=GF_BOT_ID
            )
        await send_response(message, f"💜 {reply}")
        return

    # 3. SOMEONE pings girlfriend bot (not owner) → threaten them
    if GF_BOT_ID in mention_ids and author_id not in (OWNER_ID, bot.user.id):
        punishments = [
            "timeout you for 28 days straight 🔇",
            "get you permanently muted in this server 🚫",
            "have the mods ban your account instantly 💀",
            "lock you out of every channel here 🔒",
            "make sure you can't even read this server 👀",
            "get your account reported and disabled 🗑️",
            "timeout you so long you'll forget what chatting feels like ⏳",
        ]
        pick = random.choice(punishments)
        async with message.channel.typing():
            reply = await get_ai_reply(
                f"Someone just pinged your girlfriend. Be extremely cold and threatening. "
                f"Tell them you will: \"{pick}\". 2 sentences max. Scary, aggressive. "
                f"Their username: {message.author.name}"
            )
        await send_response(
            message,
            f"⚠️ {message.author.mention} — {reply}\n> *don't ever ping my girlfriend again.* 💢",
            as_reply=True
        )
        return

    # 4. Normal mention or reply → full memory context chat
    if is_mentioned or is_reply:
        async with message.channel.typing():
            reply = await get_ai_reply(
                f"Someone is chatting with you. Use memory/context from the conversation. "
                f"Be confident, cool, helpful. If follow-up, use the history naturally. "
                f"They said: \"{content}\"",
                user_id=author_id
            )
        await send_response(message, reply, as_reply=True)

# ── COMMANDS ────────────────────────────────────────────────────────────────────
@bot.command(name="help")
async def cmd_help(ctx):
    embed = discord.Embed(
        title="💜 Kyoko — Commands",
        description="Professional coder. Devoted boyfriend. Your worst nightmare if you touch my girl.",
        color=0x9B59B6
    )
    embed.add_field(name="k!code <lang> <task>", value="Generate code in any language",        inline=False)
    embed.add_field(name="k!debug <code>",        value="Debug your broken code",               inline=False)
    embed.add_field(name="k!explain <code>",      value="Explain what code does",               inline=False)
    embed.add_field(name="k!review <code>",       value="Brutal honest code review",            inline=False)
    embed.add_field(name="k!memory",              value="See what Kyoko remembers about you",   inline=False)
    embed.add_field(name="k!forget",              value="Clear Kyoko's memory of you",          inline=False)
    embed.add_field(name="k!ping",                value="Check if I'm alive",                   inline=False)
    embed.set_footer(text="Mention me or reply to me for free chat with memory • 💜")
    await ctx.send(embed=embed)

@bot.command(name="ping")
async def cmd_ping(ctx):
    latency = round(bot.latency * 1000)
    await ctx.send(f"💜 online — `{latency}ms` latency")

@bot.command(name="memory")
async def cmd_memory(ctx):
    hist = get_history(ctx.author.id)
    if not hist:
        return await ctx.send("i don't remember anything about you yet 💀 talk to me first")
    lines = []
    for m in hist[-10:]:
        preview = m["content"][:80] + "..." if len(m["content"]) > 80 else m["content"]
        lines.append(f"**{m['role']}:** {preview}")
    embed = discord.Embed(
        title=f"💜 Kyoko's memory of {ctx.author.name}",
        description="\n".join(lines),
        color=0x9B59B6
    )
    embed.set_footer(text=f"showing last {min(len(hist), 10)} of {len(hist)} messages stored")
    await ctx.send(embed=embed)

@bot.command(name="forget")
async def cmd_forget(ctx):
    clear_memory(ctx.author.id)
    await ctx.send(f"🧹 cleared. i don't remember anything about you anymore {ctx.author.mention}")

@bot.command(name="code")
async def cmd_code(ctx, lang: str = "python", *, task: str = ""):
    if not task:
        return await ctx.send("give me something to code 💀 → `k!code <lang> <task>`")
    async with ctx.typing():
        reply = await get_ai_reply(
            f"Write COMPLETE production-ready {lang} code for: {task}. "
            f"Write EVERY single line — never truncate or skip. "
            f"Use a markdown ```{lang} code block. Add inline comments.",
            user_id=ctx.author.id,
            max_tokens=8000
        )
    await send_response(ctx, f"💻 **Kyoko coded this in {lang.upper()}:**\n{reply}", lang=lang)

@bot.command(name="debug")
async def cmd_debug(ctx, *, code: str = ""):
    if not code:
        return await ctx.send("paste the broken code 💀 → `k!debug <code>`")
    async with ctx.typing():
        reply = await get_ai_reply(
            f"Find ALL bugs, explain each one, provide the complete fixed version:\n\n{code}",
            user_id=ctx.author.id,
            max_tokens=8000
        )
    await send_response(ctx, f"🔍 **Kyoko's Debug Report:**\n{reply}")

@bot.command(name="explain")
async def cmd_explain(ctx, *, code: str = ""):
    if not code:
        return await ctx.send("give me the code to explain 💀")
    async with ctx.typing():
        reply = await get_ai_reply(
            f"Explain this code clearly and thoroughly, line by line if needed:\n\n{code}",
            user_id=ctx.author.id,
            max_tokens=4000
        )
    await send_response(ctx, f"📖 **Kyoko explains:**\n{reply}")

@bot.command(name="review")
async def cmd_review(ctx, *, code: str = ""):
    if not code:
        return await ctx.send("drop the code for review 💀")
    async with ctx.typing():
        reply = await get_ai_reply(
            f"Brutal but fair senior dev code review — bugs, inefficiencies, security, improvements:\n\n{code}",
            user_id=ctx.author.id,
            max_tokens=4000
        )
    await send_response(ctx, f"📝 **Kyoko's Code Review:**\n{reply}")

# ── KEEP-ALIVE (Render free Web Service) ────────────────────────────────────────
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class KeepAlive(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Kyoko is alive!")
    def log_message(self, *args):
        pass  # silence request logs

def run_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), KeepAlive)
    server.serve_forever()

threading.Thread(target=run_server, daemon=True).start()
print(f"Keep-alive server running on port {os.getenv('PORT', 8080)}")

# ── RUN ─────────────────────────────────────────────────────────────────────────
bot.run(DISCORD_TOKEN)
