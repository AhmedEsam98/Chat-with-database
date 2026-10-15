"""
Streamlit Web UI for chatting with ANY database using AI (Talk2Tables).

Supports PostgreSQL, MySQL, SQLite, MSSQL, and custom database URIs.
Provides dynamic in-chat credential configuration, connection testing,
model switching, pure Semantic Vector Caching, collapsible SQL views,
and Azure Blob Storage backups.

Run with:
    streamlit run app.py

Or run via Docker:
    docker-compose up web
"""

import asyncio
import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import config
from db import (
    execute_sql,
    is_safe_question,
    test_db_connection,
    set_database_connection,
    get_active_dialect,
    get_active_connection_info,
    get_active_raw_url,
    resolve_masked_url,
    fetch_schema_info,
)
from llm import generate_sql, generate_answer, rewrite_question, explain_sql
from schema import get_schema_prompt, invalidate_cache
from storage import is_storage_configured, backup_schema_to_blob, CONTAINER_NAME
from cache import (
    get_semantic_cached_answer,
    set_semantic_cached_answer,
    clear_cache,
    is_redis_available,
)

st.set_page_config(page_title="Talk2Tables — Multi-Database AI Chat", page_icon="🗄️", layout="wide")


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------
async def run_pipeline(
    question: str,
    model: str = "model_1",
    temperature: float = config.DEFAULT_TEMPERATURE,
    max_sql_attempts: int = config.MAX_SQL_ATTEMPTS,
    enable_semantic_cache: bool = config.ENABLE_SEMANTIC_CACHE,
    semantic_threshold: float = config.SEMANTIC_SIMILARITY_THRESHOLD,
    enable_query_rewrite: bool = config.ENABLE_QUERY_REWRITE,
) -> tuple[str, str | None, str | None, bool, list | None, dict]:
    """Execute the text-to-SQL pipeline with semantic vector caching and SQL capture.

    Returns (answer, sql_query_or_None, sql_explanation_or_None, is_cached, rows_or_None, latency_stats).
    """
    start_time = time.perf_counter()
    db_latency_ms = 0.0

    # 0. Safety guardrail on user question
    is_safe, error_msg = is_safe_question(question)
    if not is_safe:
        duration = time.perf_counter() - start_time
        return error_msg, None, None, False, None, {"total_s": duration, "db_ms": 0.0, "source": "guardrail"}

    # 0.1 Fast-path for greetings & casual pleasantries
    q_clean = question.strip().lower().rstrip("!?.,")
    greetings = {
        "hi", "hello", "hey", "hola", "good morning", "good afternoon",
        "good evening", "how are you", "how are you doing", "what's up",
        "whats up", "greetings", "help",
    }
    if q_clean in greetings:
        duration = time.perf_counter() - start_time
        return (
            "Hello! I am your AI database assistant. Ask me questions about your connected tables, analytics, or summaries.",
            None,
            None,
            False,
            None,
            {"total_s": duration, "db_ms": 0.0, "llm_s": 0.0, "source": "greeting"},
        )

    # 1. Semantic Redis vector cache lookup
    if enable_semantic_cache:
        semantic_cached = await get_semantic_cached_answer(
            question, model=model, threshold=semantic_threshold
        )
        if semantic_cached:
            duration = time.perf_counter() - start_time
            return semantic_cached, None, None, True, None, {"total_s": duration, "db_ms": 0.0, "source": "cache"}

    total_llm_time = 0.0

    # 2. Rewrite the question for better SQL generation
    if enable_query_rewrite:
        t_rw0 = time.perf_counter()
        rewritten = await rewrite_question(question, model=model, temperature=temperature)
        total_llm_time += (time.perf_counter() - t_rw0)
    else:
        rewritten = question

    # 3. Text-to-SQL with self-correction
    last_sql = None
    history = []
    for _ in range(max_sql_attempts):
        t_llm1 = time.perf_counter()
        sql = await generate_sql(rewritten, history=history, model=model, temperature=temperature)
        llm_sql_s = time.perf_counter() - t_llm1
        total_llm_time += llm_sql_s
        last_sql = sql
        if sql.strip().upper() == "REJECTED":
            duration = time.perf_counter() - start_time
            return "Query rejected: only single SELECT statements are allowed.", None, None, False, None, {
                "total_s": duration,
                "db_ms": 0.0,
                "llm_s": total_llm_time,
                "source": "rejected",
            }

        t_db_start = time.perf_counter()
        rows, error = await execute_sql(sql)
        db_latency_ms = (time.perf_counter() - t_db_start) * 1000

        if error and "Query rejected" in error:
            duration = time.perf_counter() - start_time
            return error, sql, None, False, None, {
                "total_s": duration,
                "db_ms": db_latency_ms,
                "llm_s": total_llm_time,
                "source": "rejected",
            }
        if not error:
            t_llm2 = time.perf_counter()
            # Generate answer and query explanation concurrently
            answer, sql_explanation = await asyncio.gather(
                generate_answer(question, rows, model=model, temperature=temperature),
                explain_sql(sql, question=question, model=model, temperature=temperature),
            )
            total_llm_time += (time.perf_counter() - t_llm2)
            if enable_semantic_cache:
                await set_semantic_cached_answer(question, answer, model=model)
            duration = time.perf_counter() - start_time
            return answer, sql, sql_explanation, False, rows, {
                "total_s": duration,
                "db_ms": db_latency_ms,
                "llm_s": total_llm_time,
                "source": "llm_db",
            }
        history.append((sql, error))

    duration = time.perf_counter() - start_time
    return config.FAILURE_MESSAGE, last_sql, None, False, None, {
        "total_s": duration,
        "db_ms": db_latency_ms,
        "llm_s": total_llm_time,
        "source": "failed",
    }


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cached_questions" not in st.session_state:
    st.session_state.cached_questions = []
if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

# Dynamic configuration state defaults
if "model" not in st.session_state:
    st.session_state.model = "model_1"
if "temperature" not in st.session_state:
    st.session_state.temperature = config.DEFAULT_TEMPERATURE
if "max_sql_attempts" not in st.session_state:
    st.session_state.max_sql_attempts = config.MAX_SQL_ATTEMPTS
if "enable_semantic_cache" not in st.session_state:
    st.session_state.enable_semantic_cache = config.ENABLE_SEMANTIC_CACHE
if "semantic_threshold" not in st.session_state:
    st.session_state.semantic_threshold = config.SEMANTIC_SIMILARITY_THRESHOLD
if "enable_query_rewrite" not in st.session_state:
    st.session_state.enable_query_rewrite = config.ENABLE_QUERY_REWRITE
if "db_uri_input" not in st.session_state:
    st.session_state.db_uri_input = get_active_raw_url()
if "db_connected" not in st.session_state:
    st.session_state.db_connected = None
if "db_status_msg" not in st.session_state:
    st.session_state.db_status_msg = None
if "schema_cols" not in st.session_state:
    st.session_state.schema_cols = None


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Return an event loop for the current execution thread."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ask(question: str):
    st.session_state.messages.append({"role": "user", "content": question})

    # Immediate safety guardrail
    is_safe, error_msg = is_safe_question(question)
    if not is_safe:
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "sql": None,
            "sql_explanation": None,
            "data": None,
            "latency": {"total_s": 0.0, "db_ms": 0.0, "source": "guardrail"},
        })
        return

    if question not in st.session_state.cached_questions:
        st.session_state.cached_questions.insert(0, question)
        st.session_state.cached_questions = st.session_state.cached_questions[:10]

    try:
        loop = _get_or_create_event_loop()
        answer, sql, sql_explanation, is_cached, rows, latency_stats = loop.run_until_complete(
            run_pipeline(
                question=question,
                model=st.session_state.model,
                temperature=st.session_state.temperature,
                max_sql_attempts=st.session_state.max_sql_attempts,
                enable_semantic_cache=st.session_state.enable_semantic_cache,
                semantic_threshold=st.session_state.semantic_threshold,
                enable_query_rewrite=st.session_state.enable_query_rewrite,
            )
        )
        st.session_state.last_latency = latency_stats
        if is_cached:
            answer = f"{answer}\n\n*(served from semantic cache 🧠)*"
    except Exception as e:
        answer = f"Error while processing your question: {e}"
        sql = None
        sql_explanation = None
        rows = None
        latency_stats = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sql": sql,
        "sql_explanation": sql_explanation,
        "data": rows,
        "latency": latency_stats,
    })


# ---------------------------------------------------------------------------
# Sidebar: Settings & Multi-Database Credentials
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- SECTION 1: DATABASE CONNECTION (URI) ---
    with st.expander("🗄️ Database Connection (URI)", expanded=True):
        loop = _get_or_create_event_loop()

        def _on_use_demo_db():
            demo_path = os.path.join("data", "sample_ecommerce.db")
            demo_url, _ = config.normalize_db_url(f"sqlite+aiosqlite:///{demo_path}")
            set_database_connection(demo_url, "SQLite Demo (sample_ecommerce.db)")
            st.session_state.db_uri_input = demo_url
            st.session_state.db_connected = True
            st.session_state.db_status_msg = "Connected"
            st.session_state.schema_cols = None
            st.session_state.messages = []
            st.session_state.flash_msg = ("success", "Connected to sample SQLite database!")

        def _on_connect_db():
            curr_uri = st.session_state.get("db_uri_input", "").strip()
            resolved = resolve_masked_url(curr_uri)
            target, dialect = config.normalize_db_url(resolved)
            t_label = f"{dialect.upper()} Database"

            ev_loop = _get_or_create_event_loop()
            t0 = time.perf_counter()
            ok, msg = ev_loop.run_until_complete(test_db_connection(target))
            lat_ms = (time.perf_counter() - t0) * 1000

            if ok:
                set_database_connection(target, t_label)
                st.session_state.db_uri_input = target
                st.session_state.db_connected = True
                st.session_state.db_status_msg = "Connected"
                st.session_state.schema_cols = None
                st.session_state.messages = []
                st.session_state.flash_msg = ("success", f"Connected to {t_label}! ⏱️ `{lat_ms:.1f}ms`")
            else:
                st.session_state.db_connected = False
                st.session_state.db_status_msg = msg
                st.session_state.flash_msg = ("error", f"Could not connect: {msg}")

        def _on_reset_to_default():
            config.reload_env()
            def_url = config.get_default_db_url()
            norm, dialect = config.normalize_db_url(def_url)
            lbl = f"{dialect.upper()} (Default .env)"

            ev_loop = _get_or_create_event_loop()
            ok, msg = ev_loop.run_until_complete(test_db_connection(norm))
            set_database_connection(norm, lbl)
            st.session_state.db_uri_input = norm
            st.session_state.db_connected = ok
            st.session_state.db_status_msg = msg if not ok else "Connected"
            st.session_state.schema_cols = None
            st.session_state.messages = []

            if ok:
                st.session_state.flash_msg = ("success", f"Reset and connected to {lbl}!")
            else:
                st.session_state.flash_msg = ("error", f"Reset to default (.env), but connection test failed: {msg}")

        # Check or cache connection status to avoid blocking every Streamlit rerun
        if st.session_state.get("db_connected") is None:
            try:
                db_connected, db_status_msg = loop.run_until_complete(test_db_connection())
                st.session_state.db_connected = db_connected
                st.session_state.db_status_msg = db_status_msg
            except Exception as e:
                st.session_state.db_connected = False
                st.session_state.db_status_msg = str(e)

        active_info = get_active_connection_info()

        if st.session_state.db_connected:
            st.success(f"🟢 **{active_info['label']}** (Connected)")
        else:
            st.error(f"🔴 **Disconnected**: {st.session_state.db_status_msg}")

        # Display flash notifications from callbacks
        if "flash_msg" in st.session_state and st.session_state.flash_msg:
            fl_level, fl_text = st.session_state.pop("flash_msg")
            if fl_level == "success":
                st.success(fl_text)
            else:
                st.error(fl_text)

        demo_db_path = os.path.join("data", "sample_ecommerce.db")
        if os.path.exists(demo_db_path):
            st.button("📦 Use Built-in Demo E-Commerce SQLite DB", on_click=_on_use_demo_db, use_container_width=True)

        # Bind with key="db_uri_input" so reset and manual edits keep state in sync
        raw_uri = st.text_area(
            "Database Connection URI",
            key="db_uri_input",
            help="Supports PostgreSQL (postgresql+asyncpg://), MySQL (mysql+aiomysql://), SQLite (sqlite+aiosqlite:///), and MSSQL.",
            placeholder="postgresql+asyncpg://user:password@localhost:5432/dbname",
        )

        st.caption(
            "**Examples:**\n"
            "- PostgreSQL: `postgresql+asyncpg://user:pass@host:5432/dbname`\n"
            "- MySQL: `mysql+aiomysql://user:pass@host:3306/dbname`\n"
            "- SQLite: `sqlite+aiosqlite:///data/sample_ecommerce.db`\n"
            "- MSSQL: `mssql+aioodbc://user:pass@host:1433/dbname`"
        )

        resolved_uri = resolve_masked_url(raw_uri)
        target_url, target_dialect = config.normalize_db_url(resolved_uri)

        # Action Buttons
        col_test, col_connect = st.columns(2)
        with col_test:
            if st.button("🔌 Test Connection", use_container_width=True):
                t_test0 = time.perf_counter()
                test_ok, test_msg = loop.run_until_complete(test_db_connection(target_url))
                test_latency_ms = (time.perf_counter() - t_test0) * 1000
                st.session_state.db_connected = test_ok
                st.session_state.db_status_msg = test_msg if not test_ok else "Connected"
                if test_ok:
                    st.success(f"✅ Connected! ⏱️ `{test_latency_ms:.1f}ms`")
                else:
                    st.error(f"❌ Failed: {test_msg} (⏱️ `{test_latency_ms:.1f}ms`)")

        with col_connect:
            st.button("💾 Connect DB", on_click=_on_connect_db, use_container_width=True, type="primary")

        st.button("🔄 Reset to Default (.env)", on_click=_on_reset_to_default, use_container_width=True)

    # --- SECTION 2: INTROSPECTED TABLES & SCHEMA ---
    with st.expander("📋 Introspected Tables & Schema", expanded=False):
        try:
            if st.session_state.get("schema_cols") is None and st.session_state.get("db_connected"):
                st.session_state.schema_cols = loop.run_until_complete(fetch_schema_info())
            schema_cols = st.session_state.get("schema_cols")
            if schema_cols:
                from collections import defaultdict
                tables_map = defaultdict(list)
                for c in schema_cols:
                    tables_map[c["table_name"]].append(c["column_name"])
                st.caption(f"Dialect: **{get_active_dialect()}** | Tables: **{len(tables_map)}**")
                for t_name, cols in sorted(tables_map.items()):
                    st.markdown(f"**`{t_name}`** ({', '.join(cols)})")
            elif not st.session_state.get("db_connected"):
                st.caption("Database disconnected. Connect to view tables.")
            else:
                st.caption("No user tables detected.")
        except Exception as e:
            st.caption(f"Error fetching schema: {e}")

    # --- SECTION 3: MODEL & GENERATION ---
    with st.expander("🤖 Model & Generation", expanded=False):
        model_options = list(config.MODELS.keys())
        selected_index = (
            model_options.index(st.session_state.model)
            if st.session_state.model in model_options
            else 0
        )
        st.session_state.model = st.selectbox(
            "Model",
            options=model_options,
            format_func=lambda k: config.MODELS[k],
            index=selected_index,
            help="Switch which Azure OpenAI deployment answers questions.",
        )

        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.05,
            help="0.0 = deterministic (best for SQL). Higher = more diverse phrasing.",
        )

        st.session_state.max_sql_attempts = st.slider(
            "Max SQL Retries",
            min_value=1,
            max_value=6,
            value=int(st.session_state.max_sql_attempts),
            step=1,
            help="Number of self-correction attempts if the generated query produces an error.",
        )

        st.session_state.enable_query_rewrite = st.toggle(
            "Query Preprocessing",
            value=bool(st.session_state.enable_query_rewrite),
            help="Normalizes abbreviations, typos, and maps terms to sample values.",
        )

    # --- SECTION 4: SEMANTIC CACHING ---
    with st.expander("⚡ Semantic Cache", expanded=False):
        st.session_state.enable_semantic_cache = st.toggle(
            "Enable Semantic Cache",
            value=bool(st.session_state.enable_semantic_cache),
            help="Return answers to semantically equivalent questions using embeddings.",
        )

        st.session_state.semantic_threshold = st.slider(
            "Similarity Cutoff",
            min_value=0.70,
            max_value=0.99,
            value=float(st.session_state.semantic_threshold),
            step=0.01,
            disabled=not st.session_state.enable_semantic_cache,
            help="Cosine similarity cutoff (0.90 default).",
        )

        if st.button("🧹 Clear Semantic Cache", use_container_width=True):
            loop = _get_or_create_event_loop()
            cleared = loop.run_until_complete(clear_cache())
            if cleared:
                st.success("Semantic cache cleared!")
            else:
                st.warning("Could not clear cache (Redis may be offline).")

    # --- SECTION 5: SERVICE DIAGNOSTICS ---
    with st.expander("🔌 Service Diagnostics", expanded=False):
        loop = _get_or_create_event_loop()
        t_diag_db = time.perf_counter()
        try:
            db_ok, db_msg = loop.run_until_complete(test_db_connection())
            db_diag_ms = (time.perf_counter() - t_diag_db) * 1000
        except Exception:
            db_ok, db_msg = False, "Error checking DB"
            db_diag_ms = 0.0

        t_diag_red = time.perf_counter()
        try:
            redis_ok = loop.run_until_complete(is_redis_available())
            red_diag_ms = (time.perf_counter() - t_diag_red) * 1000
        except Exception:
            redis_ok = False
            red_diag_ms = 0.0

        if db_ok:
            st.markdown(f"🟢 **Database ({get_active_dialect().upper()})**: Connected (`{db_diag_ms:.1f}ms`)")
        else:
            st.markdown(f"🔴 **Database**: {db_msg}")

        if redis_ok:
            st.markdown(f"🟢 **Redis**: Connected (`{red_diag_ms:.1f}ms`)")
        else:
            st.markdown("🟡 **Redis**: Offline (Fail-open)")

        st.markdown(f"🤖 **Azure Deployment**: `{config.AZURE_OPENAI_DEPLOYMENT_NAME}`")
        if is_storage_configured():
            st.markdown(f"☁️ **Storage Container**: `{CONTAINER_NAME}`")
        else:
            st.markdown("☁️ **Storage**: Not configured")

    # --- SECTION 6: AZURE BLOB STORAGE ---
    with st.expander("☁️ Azure Blob Storage", expanded=False):
        if is_storage_configured():
            st.success(f"Container: `{CONTAINER_NAME}`")
            if st.button("Backup Schema to Blob", use_container_width=True):
                with st.spinner("Introspecting schema and uploading to Azure Blob..."):
                    try:
                        loop = _get_or_create_event_loop()
                        schema_p = loop.run_until_complete(get_schema_prompt())
                        url = loop.run_until_complete(backup_schema_to_blob(schema_p))
                        if url:
                            st.success("Schema backed up to Azure Blob!")
                            st.caption(f"`{url}`")
                        else:
                            st.error("Failed to upload schema backup.")
                    except Exception as e:
                        st.error(f"Failed to upload schema backup: {e}")
        else:
            st.caption("Not configured. Set `AZURE_STORAGE_CONNECTION_STRING` in `.env`.")

    # --- SECTION 7: RECENT QUESTIONS ---
    with st.expander("🗂️ Recent Questions", expanded=False):
        if not st.session_state.cached_questions:
            st.caption("No questions yet.")
        else:
            for q in st.session_state.cached_questions:
                if st.button(q, key=f"cached_{q}", use_container_width=True):
                    ask(q)
                    st.rerun()

    st.divider()

    # Reset & Clear controls
    col_reset, col_clear = st.columns(2)
    with col_reset:
        if st.button("🔄 Reset", help="Reset all configs to defaults", use_container_width=True):
            st.session_state.model = "model_1"
            st.session_state.temperature = config.DEFAULT_TEMPERATURE
            st.session_state.max_sql_attempts = config.MAX_SQL_ATTEMPTS
            st.session_state.enable_semantic_cache = config.ENABLE_SEMANTIC_CACHE
            st.session_state.semantic_threshold = config.SEMANTIC_SIMILARITY_THRESHOLD
            st.session_state.enable_query_rewrite = config.ENABLE_QUERY_REWRITE
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear", help="Clear conversation chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
current_info = get_active_connection_info()
st.title("🗄️ Talk2Tables")

# Display current active configuration banner
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.caption(f"🔌 **Database**: `{current_info['label']}`")
with col_m2:
    st.caption(f"🧠 **Model**: `{config.MODELS.get(st.session_state.model, st.session_state.model)}`")
with col_m3:
    cache_str = f"Semantic ({st.session_state.semantic_threshold:.2f})" if st.session_state.enable_semantic_cache else "Disabled"
    st.caption(f"⚡ **Cache**: `{cache_str}`")
with col_m4:
    last_lat = st.session_state.get("last_latency")
    if last_lat:
        tot_s = last_lat.get("total_s", 0)
        db_ms = last_lat.get("db_ms", 0)
        src = last_lat.get("source", "")
        if src == "cache":
            lat_txt = f"{tot_s * 1000:.0f}ms (⚡ Cache)"
        elif src == "guardrail":
            lat_txt = f"{tot_s * 1000:.0f}ms (🛡️ Guardrail)"
        else:
            tot_str = f"{tot_s:.2f}s" if tot_s >= 1.0 else f"{tot_s * 1000:.0f}ms"
            lat_txt = f"{tot_str} (DB: {db_ms:.1f}ms)"
        st.caption(f"⏱️ **Last Latency**: `{lat_txt}`")
    else:
        st.caption("⏱️ **Last Latency**: `—`")

st.divider()

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        # Per-message latency caption for assistant responses
        if msg["role"] == "assistant" and msg.get("latency"):
            lat = msg["latency"]
            tot_s = lat.get("total_s", 0.0)
            db_ms = lat.get("db_ms", 0.0)
            llm_s = lat.get("llm_s", 0.0)
            src = lat.get("source", "llm_db")

            tot_formatted = f"{tot_s:.2f}s" if tot_s >= 1.0 else f"{tot_s * 1000:.0f}ms"
            if src == "cache":
                st.caption(f"⏱️ **Latency**: `{tot_formatted}` • ⚡ *Served from semantic cache*")
            elif src == "guardrail":
                st.caption(f"⏱️ **Latency**: `{tot_formatted}` • 🛡️ *Blocked by guardrail*")
            elif src == "rejected":
                st.caption(f"⏱️ **Latency**: `{tot_formatted}` • ⚠️ *Query rejected by safety validation*")
            else:
                llm_formatted = f"{llm_s:.2f}s" if llm_s >= 1.0 else f"{llm_s * 1000:.0f}ms"
                st.caption(f"⏱️ **Latency**: `{tot_formatted}` (🤖 LLM: `{llm_formatted}` • 🗄️ DB: `{db_ms:.1f}ms`)")

        # Show all retrieved data in an interactive table if available
        if msg["role"] == "assistant" and msg.get("data"):
            total_records = len(msg["data"])
            with st.expander(f"📊 View all {total_records} retrieved records"):
                st.dataframe(msg["data"], use_container_width=True)

        # Show SQL query & explanation in a collapsible expander for assistant messages
        if msg["role"] == "assistant" and msg.get("sql"):
            with st.expander("🔍 View generated SQL, execution metrics & explanation"):
                # Detailed latency metric cards
                if msg.get("latency"):
                    lat = msg["latency"]
                    tot_s = lat.get("total_s", 0.0)
                    db_ms = lat.get("db_ms", 0.0)
                    llm_s = lat.get("llm_s", 0.0)
                    tot_val = f"{tot_s:.2f}s" if tot_s >= 1.0 else f"{tot_s * 1000:.0f}ms"
                    llm_val = f"{llm_s:.2f}s" if llm_s >= 1.0 else f"{llm_s * 1000:.0f}ms"

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Latency", tot_val)
                    with m2:
                        st.metric("LLM Inference", llm_val)
                    with m3:
                        st.metric("Database Query", f"{db_ms:.1f}ms")
                    with m4:
                        st.metric("Records Returned", len(msg["data"]) if msg.get("data") else 0)
                    st.divider()

                st.markdown("##### 💻 Generated SQL")
                st.code(msg["sql"], language="sql")
                st.divider()
                st.markdown("##### 📝 Query Explanation")
                explanation = msg.get("sql_explanation")
                if explanation:
                    st.markdown(explanation)


user_input = st.chat_input("Ask a question about the connected database...")
if user_input:
    ask(user_input)
    st.rerun()
