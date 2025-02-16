import streamlit as st
import json
import anthropic
import openai

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
    st.title("Prompt Debugger using")
    
    debugger = PromptDebugger()
    
    # Sidebar configuration: Bot type and model selection.
    st.sidebar.header("Configuration")
    bot_type = st.sidebar.selectbox("Select Bot Type", ["Text Bot", "Voice Bot"])
    
    provider = st.sidebar.selectbox(
        "Select Model Provider",
        options=list(debugger.model_providers.keys())
    )
    
    model = st.sidebar.selectbox(
        "Select Model",
        options=list(debugger.model_providers[provider].keys()),
        format_func=lambda x: debugger.model_providers[provider][x]
    )
    
    st.header("Input Fields for Prompt Debugging")
    
    # 1. System Prompt
    st.subheader("1. System Prompt")
    system_prompt = st.text_area("Enter the System Prompt given to the agent", height=150)
    
    # 2. Conversational History
    st.subheader("2. Conversational History")
    num_exchanges = st.number_input("Number of conversation exchanges", min_value=1, value=1)
    conversation_history = []
    for i in range(num_exchanges):
        st.markdown(f"**Exchange {i+1}**")
        user_msg = st.text_area(f"User Message {i+1}", key=f"user_{i}")
        agent_msg = st.text_area(f"Agent Response {i+1}", key=f"agent_{i}")
        if user_msg:
            conversation_history.append({"role": "user", "content": user_msg})
        if agent_msg:
            conversation_history.append({"role": "assistant", "content": agent_msg})
    
    # 3. Defective Interaction (User and Agent responses)
    st.subheader("3. Defective Interaction")
    defective_user_message = st.text_area("Defective User Message", help="The user message in the interaction that produced a defective response")
    defective_agent_response = st.text_area("Defective Agent Response", help="The agent response that is defective")
    
    # 4. Description of the Defective Response
    st.subheader("4. Description of the Defective Response")
    defective_description = st.text_area("Describe what is wrong with the agent's response", help="Explain why the agent’s response is considered defective")
    
    # 5. Agent's Interpretation
    st.subheader("5. Agent's Interpretation of the Prompt")
    agent_interpretation = st.text_area("What did the agent interpret from the prompt?", help="Include the agent’s understanding of its instruction")
    
    # 6. Expected Behavior
    st.subheader("6. Expected Behavior")
    expected_behavior = st.text_area("Describe the expected behavior of the bot", help="Explain how you expected the agent to behave")
    
    # 7. Behavioral Guidelines
    st.subheader("7. Behavioral Guidelines")
    behavioral_guidelines = st.text_area("Enter the Behavioral Guidelines provided to the agent", height=150, help="These are the foundational guidelines given to the agent during its build")
    
    if st.button("Analyze"):
        # Validate required fields
        if not all([
            system_prompt, defective_user_message, defective_agent_response,
            defective_description, agent_interpretation, expected_behavior, behavioral_guidelines
        ]):
            st.warning("Please fill in all required fields before analyzing.")
        else:
            with st.spinner("Analyzing prompt and generating suggestions..."):
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
                st.header("Analysis Results")
                
                # Output field 1: Location of Error Source
                st.subheader("1. Location of Error Source")
                system_prompt_error = analysis.get("error_source_analysis", {}).get("system_prompt_error", "No issues found in System Prompt.")
                behavioral_guidelines_error = analysis.get("error_source_analysis", {}).get("behavioral_guidelines_error", "No issues found in Behavioral Guidelines.")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**System Prompt Error:**")
                    st.text_area("", value=system_prompt_error, height=120)
                with col2:
                    st.markdown("**Behavioral Guidelines Error:**")
                    st.text_area("", value=behavioral_guidelines_error, height=120)
                
                # Output field 2: Prompt Suggestions
                st.subheader("2. Prompt Suggestions")
                system_prompt_modifications = analysis.get("prompt_suggestions", {}).get("system_prompt_modifications", "No modifications suggested for System Prompt.")
                behavioral_guidelines_modifications = analysis.get("prompt_suggestions", {}).get("behavioral_guidelines_modifications", "No modifications suggested for Behavioral Guidelines.")
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**System Prompt Modifications:**")
                    st.text_area("", value=system_prompt_modifications, height=120)
                with col4:
                    st.markdown("**Behavioral Guidelines Modifications:**")
                    st.text_area("", value=behavioral_guidelines_modifications, height=120)
                
                # Output field 3: Agent Interpretation Change
                st.subheader("3. Agent Interpretation Change")
                interpretation_change = analysis.get("agent_interpretation_change", "No changes in agent interpretation provided.")
                st.text_area("", value=interpretation_change, height=120)

if __name__ == "__main__":
    main()
