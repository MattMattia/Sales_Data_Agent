"""
i18n.py - User-facing language support (English and Latin American Spanish).

Kept free of Streamlit imports so translations and language rules can be
unit-tested. Internal agent artifacts (plans, executor decisions) stay in
English; only text shown to the user follows the selected language.
"""

from typing import Dict, Optional

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "es": "Español (Latinoamérica)",
}

# Appended to prompts that produce user-facing text (synthesis, chart captions).
LANGUAGE_INSTRUCTIONS: Dict[str, str] = {
    "en": "Write your answer in English.",
    "es": (
        "Responde en español latinoamericano (es-419), con un tono profesional y neutro. "
        "Conserva los nombres de empresas, cifras y citas tal como aparecen."
    ),
}

SAMPLE_QUERIES: Dict[str, tuple] = {
    "en": (
        "What are our top 3 customer deals? Chart the deal value for each.",
        "What are our top 3 customer deals and what are the main themes in meeting notes?",
        "Research recent regulatory updates affecting financial services and healthcare deals.",
    ),
    "es": (
        "¿Cuáles son nuestros 3 principales negocios con clientes? Haz un gráfico del valor de cada uno.",
        "¿Cuáles son nuestros 3 principales negocios y cuáles son los temas centrales de las notas de reunión?",
        "Investiga las actualizaciones regulatorias recientes que afectan los negocios de servicios financieros y salud.",
    ),
}

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "language_label": "🌐 Language / Idioma",
        "page_subtitle": "Autonomous Multi-Agent Intelligence System for Enterprise Sales Analytics",
        "system_overview": "🤖 System Overview",
        "system_description": (
            "**Sales Data Agent** is an autonomous multi-agent analytics pipeline built with:\n"
            "- **LangGraph** (Dynamic Plan & Execute pattern)\n"
            "- **Snowflake Cortex** (Analyst & Search)\n"
            "- **Tavily Search** (Public web benchmarks)\n"
            "- **Python REPL** (Data visualization)\n"
            "- **TruLens** (RAG Triad + GPA Alignment)"
        ),
        "agents_header": "👥 Active Specialist Agents",
        "agents_description": (
            "- 🧠 **planner**: Decomposes queries into structured sub-goals.\n"
            "- 🧭 **executor**: Routes and evaluates sub-agent progression.\n"
            "- ❄️ **cortex_researcher**: Queries CRM metrics & meeting notes.\n"
            "- 🌐 **web_researcher**: Conducts real-time external research.\n"
            "- 📈 **chart_generator**: Synthesizes and plots metrics.\n"
            "- 📝 **chart_summarizer**: Generates standalone visual captions.\n"
            "- 🎯 **synthesizer**: Compiles final executive briefings."
        ),
        "run_config": "🎛️ Run Configuration",
        "enabled_agents": "Enabled agents",
        "enabled_agents_help": "The planner only schedules enabled agents. The synthesizer always remains available to conclude.",
        "run_evaluation": "Run TruLens Evaluation Scorecard",
        "sample_questions": "💡 Sample Questions",
        "clear_conversation": "🗑️ Clear Conversation",
        "connection_status": "⚙️ Connection Status",
        "connected": "🟢 Connected",
        "fallback": "🟡 Fallback / Mock",
        "welcome": (
            "Welcome! I am the **Sales Data Agent**. You can ask me to analyze CRM enterprise deals, "
            "retrieve customer meeting notes, run real-time market research, or create visual charts."
        ),
        "chat_placeholder": "Ask a question about sales deals, meeting transcripts, or market research...",
        "spinner": "Agents researching and analyzing...",
        "no_final_answer": "The multi-agent workflow completed, but no final synthesis was returned.",
        "execution_error": "⚠️ **Execution Error**: An error occurred while running the multi-agent graph:",
        "chart_caption": "Generated Visualization",
        "view_plan": "🔍 View Execution Plan",
        "view_multi_agent_plan": "🔍 View Multi-Agent Execution Plan",
        "scorecard_title": "📊 TruLens Evaluation Scorecard (RAG Triad + GPA)",
        "chart_summary_label": "Chart summary",
        "graph_unavailable": "Error: Graph engine could not be loaded: {error}",
    },
    "es": {
        "language_label": "🌐 Language / Idioma",
        "page_subtitle": "Sistema autónomo multiagente de inteligencia para analítica de ventas empresariales",
        "system_overview": "🤖 Descripción del sistema",
        "system_description": (
            "**Sales Data Agent** es un pipeline analítico autónomo multiagente construido con:\n"
            "- **LangGraph** (patrón dinámico Plan & Execute)\n"
            "- **Snowflake Cortex** (Analyst y Search)\n"
            "- **Tavily Search** (referencias públicas de la web)\n"
            "- **Python REPL** (visualización de datos)\n"
            "- **TruLens** (RAG Triad + alineación GPA)"
        ),
        "agents_header": "👥 Agentes especialistas",
        "agents_description": (
            "- 🧠 **planner**: descompone la consulta en submetas estructuradas.\n"
            "- 🧭 **executor**: enruta y evalúa el avance de los subagentes.\n"
            "- ❄️ **cortex_researcher**: consulta métricas del CRM y notas de reuniones.\n"
            "- 🌐 **web_researcher**: investiga fuentes externas en tiempo real.\n"
            "- 📈 **chart_generator**: sintetiza y grafica métricas.\n"
            "- 📝 **chart_summarizer**: redacta descripciones de los gráficos.\n"
            "- 🎯 **synthesizer**: compila el informe ejecutivo final."
        ),
        "run_config": "🎛️ Configuración de la ejecución",
        "enabled_agents": "Agentes habilitados",
        "enabled_agents_help": "El planner solo programa agentes habilitados. El synthesizer siempre está disponible para concluir.",
        "run_evaluation": "Ejecutar scorecard de evaluación de TruLens",
        "sample_questions": "💡 Preguntas de ejemplo",
        "clear_conversation": "🗑️ Borrar conversación",
        "connection_status": "⚙️ Estado de conexiones",
        "connected": "🟢 Conectado",
        "fallback": "🟡 Respaldo / Mock",
        "welcome": (
            "¡Hola! Soy **Sales Data Agent**. Puedo analizar los negocios empresariales del CRM, "
            "consultar notas de reuniones con clientes, investigar el mercado en tiempo real o crear gráficos."
        ),
        "chat_placeholder": "Haz una pregunta sobre negocios, transcripciones de reuniones o investigación de mercado...",
        "spinner": "Los agentes están investigando y analizando...",
        "no_final_answer": "El flujo multiagente terminó, pero no devolvió una síntesis final.",
        "execution_error": "⚠️ **Error de ejecución**: ocurrió un error al ejecutar el grafo multiagente:",
        "chart_caption": "Visualización generada",
        "view_plan": "🔍 Ver plan de ejecución",
        "view_multi_agent_plan": "🔍 Ver plan de ejecución multiagente",
        "scorecard_title": "📊 Scorecard de evaluación de TruLens (RAG Triad + GPA)",
        "chart_summary_label": "Resumen del gráfico",
        "graph_unavailable": "Error: no se pudo cargar el motor del grafo: {error}",
    },
}


def normalize_language(language: Optional[str]) -> str:
    """Maps any locale tag ('es', 'es-AR', 'es_419', 'en-US', None) to a supported code."""
    code = (language or "").replace("_", "-").split("-")[0].lower()
    return code if code in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def t(key: str, language: Optional[str] = DEFAULT_LANGUAGE, **values: str) -> str:
    """Returns the UI string for `key` in `language`, falling back to English."""
    lang = normalize_language(language)
    text = TRANSLATIONS[lang].get(key) or TRANSLATIONS[DEFAULT_LANGUAGE][key]
    return text.format(**values) if values else text


def language_instruction(language: Optional[str]) -> str:
    """Prompt line asking the model to answer in the user's language."""
    return LANGUAGE_INSTRUCTIONS[normalize_language(language)]


def sample_queries(language: Optional[str]) -> tuple:
    return SAMPLE_QUERIES[normalize_language(language)]
