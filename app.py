import os
import streamlit as st
import litellm
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# --- GLOBAL LITELLM OVERRIDES FOR GROQ COMPATIBILITY ---
# Instruct LiteLLM to drop any unmapped parameters globally
litellm.drop_params = True
litellm.set_verbose = False

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

# --- CUSTOM SEARCH TOOL DEFINITION ---
@tool("DuckDuckGo Internet Search")
def web_search_tool(query: str) -> str:
    """Useful for searching the web for current events, facts, and up-to-date research topics."""
    try:
        ddgs = DDGS()
        results = list(ddgs.text(keywords=query, max_results=5))
        if not results:
            return "No relevant search results found."
        
        formatted_results = []
        for item in results:
            title = item.get("title", "No Title")
            snippet = item.get("body", "No Snippet")
            link = item.get("href", "")
            formatted_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}\n")
        
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error executing internet search: {str(e)}"


# --- API KEY MANAGEMENT ---
st.sidebar.title("⚙️ Agent Settings")

secret_api_key = st.secrets.get("GROQ_API_KEY", "")

if secret_api_key:
    groq_api_key = secret_api_key
    st.sidebar.success("✅ Groq API Key loaded from Streamlit Secrets!")
else:
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key (starts with 'gsk_')"
    )

model_choice = st.sidebar.selectbox(
    "Select Model",
    options=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    help="Select the Groq-hosted model to power your agent."
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Obtain your free API key at [console.groq.com](https://console.groq.com).")


# --- MAIN INTERFACE ---
st.title("🤖 AI Research Agent")
st.subheader("Autonomous Research & Report Generation Powered by CrewAI & Groq")

research_topic = st.text_input(
    "Enter Research Topic:",
    placeholder="e.g., Impact of Quantum Computing on Cybersecurity in 2026"
)

generate_btn = st.button("🚀 Start Research", type="primary", use_container_width=True)

# --- CREWAI EXECUTION LOGIC ---
if generate_btn:
    if not groq_api_key.strip():
        st.error("Please enter a valid Groq API Key in the sidebar or save it in Streamlit Secrets.")
    elif not research_topic.strip():
        st.warning("Please provide a topic for the research agent.")
    else:
        # Set environment variables expected by LiteLLM / Groq API
        os.environ["GROQ_API_KEY"] = groq_api_key

        with st.status("🔍 Research Agent at Work...", expanded=True) as status:
            try:
                st.write("Initializing Groq Endpoint...")
                
                # Configure native CrewAI LLM targeting Groq's OpenAI-compatible base URL directly
                # Passing base_url avoids LiteLLM adding provider-specific caching headers
                llm = LLM(
                    model=f"openai/{model_choice}",
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.3
                )

                st.write("Configuring Research Agent...")
                research_agent = Agent(
                    role="Senior Academic & Technical Researcher",
                    goal=f"Conduct thorough and accurate web research on the given topic: '{research_topic}'",
                    backstory=(
                        "You are an expert analytical researcher skilled at finding concise, accurate, "
                        "and current information across the web using internet search tools. You specialize in "
                        "synthesizing research into clean, structured reports."
                    ),
                    tools=[web_search_tool],
                    llm=llm,
                    verbose=True,
                    allow_delegation=False,
                    cache=False
                )

                st.write("Formulating research tasks...")
                research_task = Task(
                    description=(
                        f"1. Search the web for relevant details about: '{research_topic}'.\n"
                        "2. Analyze the retrieved search results for credibility and accuracy.\n"
                        "3. Synthesize the findings into a clear, comprehensive Markdown research report.\n"
                        "4. Ensure your output includes an Overview, Key Findings, Technical Implications, and Future Outlook."
                    ),
                    expected_output=(
                        "A fully formatted Markdown report with section headings, actionable insights, "
                        "and bulleted technical takeaways."
                    ),
                    agent=research_agent
                )

                st.write("Running CrewAI Orchestration...")
                crew = Crew(
                    agents=[research_agent],
                    tasks=[research_task],
                    verbose=True,
                    memory=False
                )

                result = crew.kickoff()
                status.update(label="✅ Research Complete!", state="complete", expanded=False)

                # --- DISPLAY OUTPUT ---
                st.markdown("### 📄 Generated Research Report")
                st.markdown("---")
                
                report_text = str(result)
                st.markdown(report_text)

                st.download_button(
                    label="📥 Download Report as Markdown",
                    data=report_text,
                    file_name=f"{research_topic.lower().replace(' ', '_')}_report.md",
                    mime="text/markdown"
                )

            except Exception as e:
                status.update(label="❌ Error Occurred", state="error", expanded=True)
                st.error(f"Execution failed: {str(e)}")
                
