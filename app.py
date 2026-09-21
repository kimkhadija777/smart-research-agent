import os
import streamlit as st
import litellm
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# Prevent parameter errors
litellm.drop_params = True
litellm.set_verbose = False

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

# --- FAST & RELIABLE SEARCH TOOL ---
@tool("DuckDuckGo Internet Search")
def web_search_tool(query: str) -> str:
    """Useful for searching the web for current facts, news, and technical topics."""
    try:
        results = []
        with DDGS() as ddgs:
            # Fetch search results cleanly
            raw_results = list(ddgs.text(query, max_results=5))
            for item in raw_results:
                title = item.get("title", "")
                snippet = item.get("body", "")
                link = item.get("href", "")
                results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")
        
        if results:
            return "\n\n---\n\n".join(results)
        
        return "Search completed, but no direct web results were returned. Rely on pre-existing analytical knowledge to complete the report."
    except Exception as e:
        return f"Search notice: {str(e)}. Proceed using direct analytical knowledge."


# --- SIDEBAR & API KEY ---
st.sidebar.title("⚙️ Agent Settings")

secret_api_key = st.secrets.get("GROQ_API_KEY", "")

if secret_api_key:
    groq_api_key = secret_api_key
    st.sidebar.success("✅ Groq API Key loaded!")
else:
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

model_choice = st.sidebar.selectbox(
    "Select Model",
    options=["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Get key at [console.groq.com](https://console.groq.com)")


# --- MAIN INTERFACE ---
st.title("🤖 AI Research Agent")
st.subheader("Autonomous Research & Report Generation Powered by CrewAI & Groq")

research_topic = st.text_input(
    "Enter Research Topic:",
    placeholder="e.g., Introduction to Agentic AI and Multi-Agent Systems"
)

generate_btn = st.button("🚀 Start Research", type="primary", use_container_width=True)

# --- CREWAI EXECUTION LOGIC ---
if generate_btn:
    if not groq_api_key.strip():
        st.error("Please enter a valid Groq API Key.")
    elif not research_topic.strip():
        st.warning("Please enter a research topic.")
    else:
        os.environ["GROQ_API_KEY"] = groq_api_key

        with st.status("🔍 Generating Research Report...", expanded=True) as status:
            try:
                st.write("Initializing Groq Model...")
                llm = LLM(
                    model=f"openai/{model_choice}",
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.3
                )

                st.write("Configuring Agent limits...")
                research_agent = Agent(
                    role="Senior Academic Researcher",
                    goal=f"Provide a structured research analysis on: '{research_topic}'",
                    backstory="You are an expert researcher. You perform quick web lookups and synthesize technical insights efficiently.",
                    tools=[web_search_tool],
                    llm=llm,
                    verbose=True,
                    allow_delegation=False,
                    max_iter=3,          # Limits total tool usage calls to avoid long loops
                    max_execution_time=120  # Timeout protection (2 minutes max)
                )

                st.write("Formulating Task...")
                research_task = Task(
                    description=(
                        f"1. Perform a web search for: '{research_topic}'.\n"
                        "2. Draft a clear report containing: Overview, Core Principles, Key Use Cases, and Future Outlook.\n"
                        "3. Keep the output well-structured in Markdown format."
                    ),
                    expected_output="A clean, structured Markdown report with clear section headings.",
                    agent=research_agent
                )

                st.write("Executing CrewAI Process...")
                crew = Crew(
                    agents=[research_agent],
                    tasks=[research_task],
                    verbose=True
                )

                result = crew.kickoff()
                status.update(label="✅ Research Complete!", state="complete", expanded=False)

                st.markdown("### 📄 Generated Research Report")
                st.markdown("---")
                
                report_text = str(result)
                st.markdown(report_text)

                st.download_button(
                    label="📥 Download Report (.md)",
                    data=report_text,
                    file_name=f"{research_topic.lower().replace(' ', '_')}_report.md",
                    mime="text/markdown"
                )

            except Exception as e:
                status.update(label="❌ Error Occurred", state="error", expanded=True)
                st.error(f"Execution failed: {str(e)}")
