import streamlit as st
import json
import anthropic
import openai

# Custom CSS for better styling
st.set_page_config(
    page_title="AI Prompt Debugger",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin-top: 2rem;
    }
    .stButton button:hover {
        background-color: #FF2B2B;
    }
    .stTextArea textarea {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .stTextArea textarea:focus {
        border-color: #FF4B4B;
        box-shadow: 0 0 0 1px #FF4B4B;
    }
    .sidebar .element-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    h3 {
        color: #34495e;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .stMarkdown {
        padding: 0.5rem 0;
    }
    .help-text {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

class PromptDebugger:
    def __init__(self):
        # Initialize API clients with keys from st.secrets
        self.anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        # Mapping of model providers to models
        self.model_providers = {
            "Anthropic": {
                "claude-3-opus-20240229": "Claude 3 Opus",
                "claude-3-sonnet-20240229": "Claude 3 Sonnet",
                "claude-3-haiku-20240307": "Claude 3 Haiku",
                "claude-3-5-haiku-latest": "Claude 3.5 Haiku",
                "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet"
            },
            "OpenAI": {
                "gpt-4-0125-preview": "GPT-4 Turbo",
                "gpt-4": "GPT-4",
                "gpt-3.5-turbo": "GPT-3.5 Turbo",
                "gpt-4o": "GPT-4o"
            }
        }
    
    def analyze_prompt(
        self,
        bot_type: str,
        system_prompt: str,
        conversation_history: list,
        defective_user_message: str,
        defective_agent_response: str,
        defective_description: str,
        agent_interpretation: str,
        expected_behavior: str,
        behavioral_guidelines: str,
        provider: str,
        model: str
    ) -> dict:
        # Build the analysis prompt incorporating all provided inputs.
        analysis_prompt = f"""
You are an expert in AI system prompt debugging and analysis.

The following inputs are provided from a prompt debugging session for an AI conversational agent:

Bot Type: {bot_type}

1. System Prompt:
{system_prompt}

2. Conversational History:
{json.dumps(conversation_history, indent=2)}

3. Defective Interaction:
   - User Message: {defective_user_message}
   - Agent Response: {defective_agent_response}

4. User's Description of the Defective Agent Response:
{defective_description}

5. Agent's Interpretation of the Prompt:
{agent_interpretation}

6. Expected Behavior from the Bot:
{expected_behavior}

7. Behavioral Guidelines:
{behavioral_guidelines}

Analyze the above information to determine:

A. The location of the error source: Identify if the error originates in the System Prompt, the Behavioral Guidelines, or both. Highlight the problematic sections from the respective inputs.

B. Suggest changes to improve the prompt: Provide modifications for the System Prompt and/or the Behavioral Guidelines if applicable.

C. Explain how the agent's reasoning or interpretation changes compared to its previous interpretation.

Provide your output in a JSON format with the following keys:
{{
    "error_source_analysis": {{
        "system_prompt_error": "Detailed explanation of errors in the system prompt (if any)",
        "behavioral_guidelines_error": "Detailed explanation of errors in the behavioral guidelines (if any)"
    }},
    "prompt_suggestions": {{
        "system_prompt_modifications": "Suggested changes for the system prompt (if applicable)",
        "behavioral_guidelines_modifications": "Suggested changes for the behavioral guidelines (if applicable)"
    }},
    "agent_interpretation_change": "Explanation of how the agent's reasoning changes relative to its previous interpretation"
}}
        """
        try:
            if provider == "Anthropic":
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                analysis = json.loads(response.content[0].text)
            else:  # OpenAI
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert in AI prompt debugging and analysis."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.2
                )
                analysis = json.loads(response.choices[0].message.content)
            
            return analysis
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            return {}

def main():
    st.title("🔍 AI Prompt Debugger")
    
    debugger = PromptDebugger()
    
    # Create two columns for the main layout
    col1, col2 = st.columns([1, 3])
    
    # Sidebar configuration in the first column
    with col1:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        
        bot_type = st.selectbox(
            "🤖 Select Bot Type",
            ["Text Bot", "Voice Bot"],
            help="Choose the type of bot you're debugging"
        )
        
        provider = st.selectbox(
            "🏢 Model Provider",
            options=list(debugger.model_providers.keys()),
            help="Select the AI model provider"
        )
        
        model = st.selectbox(
            "🧠 Model",
            options=list(debugger.model_providers[provider].keys()),
            format_func=lambda x: debugger.model_providers[provider][x],
            help="Choose the specific model version"
        )
    
    # Main content in the second column
    with col2:
        st.markdown("### 📝 Input Fields")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["System", "Interaction", "Behaviour"])
        
        with tab1:
            st.subheader("System Prompt")
            system_prompt = st.text_area(
                "Enter the System Prompt",
                height=150,
                help="The initial instructions given to the agent",
                placeholder="Enter the system prompt that defines the agent's behavior..."
            )
            
            st.subheader("Behavioral Guidelines")
            behavioral_guidelines = st.text_area(
                "Enter Behavioral Guidelines",
                height=150,
                help="The foundational guidelines given to the agent",
                placeholder="Enter any specific behavioral guidelines or constraints..."
            )
        
        with tab2:
            st.subheader("Conversation History")
            num_exchanges = st.number_input(
                "Number of exchanges",
                min_value=1,
                value=1,
                help="How many back-and-forth messages to include"
            )
            
            conversation_history = []
            for i in range(num_exchanges):
                st.markdown(f"**Exchange {i+1}**")
                col_user, col_agent = st.columns(2)
                
                with col_user:
                    user_msg = st.text_area(
                        "User Message",
                        key=f"user_{i}",
                        placeholder="What the user said..."
                    )
                
                with col_agent:
                    agent_msg = st.text_area(
                        "Agent Response",
                        key=f"agent_{i}",
                        placeholder="How the agent responded..."
                    )
                
                if user_msg:
                    conversation_history.append({"role": "user", "content": user_msg})
                if agent_msg:
                    conversation_history.append({"role": "assistant", "content": agent_msg})
        
        with tab3:
            st.subheader("Defective Interaction")
            defective_user_message = st.text_area(
                "User Message",
                help="The user message that led to the defective response",
                placeholder="Enter the user message that caused issues..."
            )
            
            defective_agent_response = st.text_area(
                "Defective Agent Response",
                help="The problematic response from the agent",
                placeholder="Enter the agent's problematic response..."
            )
            
            st.subheader("Agent Logs")
            defective_description = st.text_area(
                "Description of defective behaviour",
                help="Explain what went wrong with the agent's response",
                placeholder="Describe why the response was problematic..."
            )
            
            agent_interpretation = st.text_area(
                "Agent Reasoning",
                help="How did the agent understand its instructions",
                placeholder="Explain how the agent interpreted the prompt..."
            )
            
            expected_behavior = st.text_area(
                "Expected Behavior",
                help="What should the agent have done",
                placeholder="Describe the desired behavior..."
            )
        
        # Analysis button with custom styling
        if st.button("🔍 Analyze Prompt", help="Click to analyze the prompt and get suggestions"):
            if not all([
                system_prompt, defective_user_message, defective_agent_response,
                defective_description, agent_interpretation, expected_behavior,
                behavioral_guidelines
            ]):
                st.warning("⚠️ Please fill in all required fields before analyzing.")
            else:
                with st.spinner("🔄 Analyzing prompt and generating suggestions..."):
                    analysis = debugger.analyze_prompt(
                        bot_type,
                        system_prompt,
                        conversation_history,
                        defective_user_message,
                        defective_agent_response,
                        defective_description,
                        agent_interpretation,
                        expected_behavior,
                        behavioral_guidelines,
                        provider,
                        model
                    )
                
                if analysis:
                    st.success("✅ Analysis Complete!")
                    
                    # Display results in an organized way
                    st.markdown("### 📊 Analysis Results")
                    
                    # Create three columns for the results
                    col_error, col_suggest, col_interpret = st.columns(3)
                    
                    with col_error:
                        st.markdown("#### 🔍 Error Sources")
                        st.markdown("**System Prompt:**")
                        st.info(analysis.get("error_source_analysis", {}).get("system_prompt_error", "No issues found"))
                        st.markdown("**Guidelines:**")
                        st.info(analysis.get("error_source_analysis", {}).get("behavioral_guidelines_error", "No issues found"))
                    
                    with col_suggest:
                        st.markdown("#### 💡 Suggestions")
                        st.markdown("**System Prompt:**")
                        st.success(analysis.get("prompt_suggestions", {}).get("system_prompt_modifications", "No modifications needed"))
                        st.markdown("**Guidelines:**")
                        st.success(analysis.get("prompt_suggestions", {}).get("behavioral_guidelines_modifications", "No modifications needed"))
                    
                    with col_interpret:
                        st.markdown("#### 🤔 Interpretation Changes")
                        st.warning(analysis.get("agent_interpretation_change", "No changes in interpretation"))

if __name__ == "__main__":
    main()
