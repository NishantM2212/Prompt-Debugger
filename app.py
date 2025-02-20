import streamlit as st
import json
import anthropic
import openai
from datetime import datetime

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
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #4CAF50;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.2);
        color: white;
        font-weight: 600;
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
        try:
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

Analyze the above information and provide your analysis in the following JSON format:

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

Important: Ensure your response is in valid JSON format with proper escaping of special characters.
"""
            if provider == "Anthropic":
                # Modify the prompt to explicitly request JSON
                anthropic_prompt = f"""Analyze the following prompt debugging session and provide your analysis in valid JSON format.
                
{analysis_prompt}

IMPORTANT: Your response must be a valid JSON object with the exact structure shown above. Do not include any text outside of the JSON object.
Do not include any explanatory text or markdown formatting. Respond only with the JSON object.
"""
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{
                        "role": "user",
                        "content": anthropic_prompt
                    }],
                    system="You are an expert in AI prompt debugging and analysis. Always respond with valid JSON format.",
                    temperature=0.2
                )
                try:
                    response_content = response.content[0].text
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    return {
                        "error_source_analysis": f"An error occurred during analysis: {str(e)}",
                        "prompt_suggestions": ["The system encountered an error. Please try again.",
                                            "If the error persists, try simplifying your prompt.",
                                            "Check your API keys and connection."],
                        "agent_interpretation_change": "Analysis interrupted due to an error."
                    }
            else:  # OpenAI
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in AI prompt debugging and analysis. Provide your analysis in valid JSON format."
                        },
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.2,
                    response_format={ "type": "json_object" }  # Ensure JSON response
                )
                try:
                    response_content = response.choices[0].message.content
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    return {
                        "error_source_analysis": f"An error occurred during analysis: {str(e)}",
                        "prompt_suggestions": ["The system encountered an error. Please try again.",
                                            "If the error persists, try simplifying your prompt.",
                                            "Check your API keys and connection."],
                        "agent_interpretation_change": "Analysis interrupted due to an error."
                    }
            
            # Try to parse JSON response
            try:
                analysis_result = json.loads(response_content)
            except json.JSONDecodeError as e:
                st.error(f"Error parsing {provider} response: Invalid JSON format")
                print(f"Raw response content: {response_content}")
                print(f"JSON parse error: {str(e)}")
                
                # Return a fallback response
                return {
                    "error_source_analysis": "Unable to parse API response into JSON format. Please try again.",
                    "prompt_suggestions": ["Consider rephrasing your prompt to be more specific.",
                                        "Try breaking down your request into smaller parts.",
                                        "Add more context to help the AI understand your intent."],
                    "agent_interpretation_change": "Unable to analyze agent interpretation due to parsing error."
                }

            return analysis_result

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return {
                "error_source_analysis": f"An error occurred during analysis: {str(e)}",
                "prompt_suggestions": ["The system encountered an error. Please try again.",
                                    "If the error persists, try simplifying your prompt.",
                                    "Check your API keys and connection."],
                "agent_interpretation_change": "Analysis interrupted due to an error."
            }

def main():
    st.title("🔍 AI Prompt Debugger")
    
    # Initialize session state for analysis history
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
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

        # Add History Section in Sidebar
        st.markdown("---")
        st.markdown("### 📚 Analysis History")
        
        if len(st.session_state.analysis_history) > 0:
            for idx, history_item in enumerate(st.session_state.analysis_history):
                with st.expander(f"Analysis #{idx + 1} - {history_item['timestamp']}", expanded=False):
                    st.markdown("**Bot Type:** " + history_item['bot_type'])
                    st.markdown("**Provider:** " + history_item['provider'])
                    st.markdown("**Model:** " + history_item['model'])
                    
                    if st.button("Load Inputs", key=f"load_{idx}"):
                        # Store the selected history item in session state
                        st.session_state.selected_history = history_item
                        st.rerun()
        else:
            st.info("No analysis history available")
            
        # Add clear history button
        if len(st.session_state.analysis_history) > 0:
            if st.button("🗑️ Clear History"):
                st.session_state.analysis_history = []
                st.rerun()
    
    # Main content in the second column
    with col2:
        st.markdown("### 📝 Input Fields")
        
        # Load historical inputs if selected
        if "selected_history" in st.session_state:
            history_item = st.session_state.selected_history
            # Clear the selection after loading
            del st.session_state.selected_history
            
            # Pre-fill the form fields
            bot_type = history_item['bot_type']
            provider = history_item['provider']
            model = history_item['model']
            system_prompt = history_item['system_prompt']
            behavioral_guidelines = history_item['behavioral_guidelines']
            conversation_history = history_item['conversation_history']
            defective_user_message = history_item['defective_user_message']
            defective_agent_response = history_item['defective_agent_response']
            defective_description = history_item['defective_description']
            agent_interpretation = history_item['agent_interpretation']
            expected_behavior = history_item['expected_behavior']
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["System", "Interaction", "Behaviour"])
        
        with tab1:
            st.subheader("System Prompt")
            system_prompt = st.text_area(
                "Enter the System Prompt",
                value=system_prompt if "selected_history" in st.session_state else "",
                height=150,
                help="The initial instructions given to the agent",
                placeholder="Enter the system prompt that defines the agent's behavior..."
            )
            
            st.subheader("Behavioral Guidelines")
            behavioral_guidelines = st.text_area(
                "Enter Behavioral Guidelines",
                value=behavioral_guidelines if "selected_history" in st.session_state else "",
                height=150,
                help="The foundational guidelines given to the agent",
                placeholder="Enter any specific behavioral guidelines or constraints..."
            )
        
        with tab2:
            st.subheader("Conversation History")
            # If loading from history, set the number of exchanges based on the history
            initial_exchanges = len(conversation_history) // 2 if "selected_history" in st.session_state else 1
            num_exchanges = st.number_input(
                "Number of exchanges",
                min_value=1,
                value=initial_exchanges,
                help="How many back-and-forth messages to include"
            )
            
            conversation_history = []
            for i in range(num_exchanges):
                st.markdown(f"**Exchange {i+1}**")
                col_user, col_agent = st.columns(2)
                
                # Get historical messages if available
                historical_user_msg = ""
                historical_agent_msg = ""
                if "selected_history" in st.session_state and i * 2 < len(history_item['conversation_history']):
                    historical_user_msg = history_item['conversation_history'][i * 2]['content']
                    if i * 2 + 1 < len(history_item['conversation_history']):
                        historical_agent_msg = history_item['conversation_history'][i * 2 + 1]['content']
                
                with col_user:
                    user_msg = st.text_area(
                        "User Message",
                        value=historical_user_msg,
                        key=f"user_{i}",
                        placeholder="What the user said..."
                    )
                
                with col_agent:
                    agent_msg = st.text_area(
                        "Agent Response",
                        value=historical_agent_msg,
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
                value=defective_user_message if "selected_history" in st.session_state else "",
                help="The user message that led to the defective response",
                placeholder="Enter the user message that caused issues..."
            )
            
            defective_agent_response = st.text_area(
                "Defective Agent Response",
                value=defective_agent_response if "selected_history" in st.session_state else "",
                help="The problematic response from the agent",
                placeholder="Enter the agent's problematic response..."
            )
            
            st.subheader("Agent Logs")
            defective_description = st.text_area(
                "Description of defective behaviour",
                value=defective_description if "selected_history" in st.session_state else "",
                help="Explain what went wrong with the agent's response",
                placeholder="Describe why the response was problematic..."
            )
            
            agent_interpretation = st.text_area(
                "Agent Reasoning",
                value=agent_interpretation if "selected_history" in st.session_state else "",
                help="How did the agent understand its instructions",
                placeholder="Explain how the agent interpreted the prompt..."
            )
            
            expected_behavior = st.text_area(
                "Expected Behavior",
                value=expected_behavior if "selected_history" in st.session_state else "",
                help="What should the agent have done",
                placeholder="Describe the desired behavior..."
            )
            
            # Analysis button moved to tab3
            if st.button("🔍 Analyze Prompt", help="Click to analyze the prompt and get suggestions"):
                if not all([
                    system_prompt, defective_user_message, defective_agent_response,
                    defective_description, agent_interpretation, expected_behavior,
                    behavioral_guidelines
                ]):
                    st.warning("⚠️ Please fill in all required fields before analyzing.")
                else:
                    # Save current inputs to history before analysis
                    current_inputs = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "bot_type": bot_type,
                        "provider": provider,
                        "model": model,
                        "system_prompt": system_prompt,
                        "behavioral_guidelines": behavioral_guidelines,
                        "conversation_history": conversation_history,
                        "defective_user_message": defective_user_message,
                        "defective_agent_response": defective_agent_response,
                        "defective_description": defective_description,
                        "agent_interpretation": agent_interpretation,
                        "expected_behavior": expected_behavior
                    }
                    
                    # Add to history if not already present
                    if current_inputs not in st.session_state.analysis_history:
                        st.session_state.analysis_history.append(current_inputs)
                    
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
                        st.markdown("""
                        <div style='text-align: center; margin-bottom: 2rem;'>
                            <h2 style='color: #FF4B4B;'>📊 Analysis Results</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # First section: Error Sources (horizontal layout)
                        st.markdown("#### 🔍 Error Source Analysis")
                        col_sys, col_guide = st.columns(2)
                        
                        with col_sys:
                            st.markdown("""
                            <div style='padding: 1rem; border-radius: 10px; height: 80%;'>
                                <h5 style='color: #f8f9fa;'>System Prompt</h5>
                                <div style='padding-left: 1rem;'>
                            """, unsafe_allow_html=True)
                            st.info(analysis.get("error_source_analysis", {}).get("system_prompt_error", "No issues found"))
                            st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        with col_guide:
                            st.markdown("""
                            <div style='padding: 1rem; border-radius: 10px; height: 80%;'>
                            <h5 style='color: #f8f9fa;'>Behavioral Guidelines</h5>
                                <div style='border-left: 4px solid #FF4B4B; padding-left: 1rem;'>
                            """, unsafe_allow_html=True)
                            st.info(analysis.get("error_source_analysis", {}).get("behavioral_guidelines_error", "No issues found"))
                            st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        # Add spacing
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Second section: Suggestions (vertical)
                        st.markdown("""
                        <div style='padding: 1.5rem; padding-left: 0; border-radius: 10px;'>
                            <h4 style='color: #f8f9fa; margin-bottom: 1rem;'>💡 Improvement Suggestions</h4>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**System Prompt Modifications:**")
                        st.success(analysis.get("prompt_suggestions", {}).get("system_prompt_modifications", "No modifications needed"))
                        
                        st.markdown("*Behavioral Guidelines Modifications:*", unsafe_allow_html=True)
                        st.success(analysis.get("prompt_suggestions", {}).get("behavioral_guidelines_modifications", "No modifications needed"))
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Add spacing
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Third section: Interpretation Changes (vertical)
                        st.markdown("""
                        <div style='padding: 1.5rem; padding-left: 0; border-radius: 10px; '>
                            <h4 style='color: #f8f9fa;'>🤔 Interpretation Analysis</h4>                
                        """, unsafe_allow_html=True)
                        st.warning(analysis.get("agent_interpretation_change", "No changes in interpretation"))
                        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
