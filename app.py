import streamlit as st
import requests
import json
import pandas as pd
import base64
import io
from datetime import datetime
from pydub import AudioSegment

# Set page title and configuration
st.set_page_config(
    page_title="Multi-Model AI Chat",
    layout="wide"
)

# Cloudflare API configuration
CF_ACCOUNT_ID = "4bea8a3acef8b92fb91b25cbdd8d00e7"
CF_API_TOKEN = "O-uAfB4WWGuKILGrnTBxWauSGjXrN7j0rPBXo-XV"

# Available models with metadata
MODELS = {
    "Llama 3 (8B) - Chat": {
        "id": "@cf/meta/llama-3-8b-instruct",
        "type": "chat",
        "description": "Llama 3 8B Instruct model for chat-based interactions."
    },
    "M2M100 (1.2B) - Translation": {
        "id": "@cf/meta/m2m100-1.2b",
        "type": "translation",
        "description": "Multilingual translation model supporting 100+ languages."
    },
    "Whisper (Large v3) - Speech Recognition": {
        "id": "@cf/openai/whisper-large-v3-turbo",
        "type": "speech",
        "description": "Automatic speech recognition and translation model."
    }
}

# Language options for translation
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr",
    "Korean": "ko"
}

# Default system messages for different model types
DEFAULT_SYSTEM_MESSAGES = {
    "chat": "You are a friendly assistant that helps write stories",
    "translation": "You are a helpful translator.",
    "speech": "You are a helpful speech recognition assistant."
}

# Function to call Cloudflare AI API
def call_cloudflare_ai(model_id, input_data):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model_id}"
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, headers=headers, json=input_data)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "errors": [{"message": str(e)}]}

# Function to extract response from Cloudflare API result
def extract_response(result, model_type):
    if not result.get("success", False):
        error_msg = result.get("errors", [{"message": "Unknown error"}])[0].get("message", "Unknown error")
        return f"❌ **Error:** {error_msg}"
    
    if model_type == "translation":
        if "result" in result and "translated_text" in result["result"]:
            return result["result"]["translated_text"]
    elif model_type == "chat":
        if "result" in result:
            # Different models might have different response structures
            if isinstance(result["result"], dict):
                # For structured responses like llama-3-8b-instruct
                return result["result"].get("response", "No response found in result")
            elif isinstance(result["result"], str):
                # For models that return a direct string
                return result["result"]
    elif model_type == "speech":
        if "result" in result:
            text = result["result"].get("text", "")
            
            # Add additional details if available
            details = {}
            if "transcription_info" in result["result"]:
                info = result["result"]["transcription_info"]
                details["language"] = info.get("language", "")
                details["confidence"] = info.get("language_probability", 0)
                details["duration"] = info.get("duration", 0)
            
            if details:
                detail_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
                return f"{text}\n\n*Details: {detail_str}*"
            return text
    
    # Fallback for unexpected response structure
    return f"⚠️ **Unexpected response format:** ```json\n{json.dumps(result, indent=2)}\n```"

# Function to export chat history to CSV
def export_chat_to_csv(chat_history):
    if not chat_history:
        return None, None
    
    # Create a dataframe from the chat history
    df = pd.DataFrame(chat_history)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = df.to_csv(index=False).encode('utf-8')
    
    return csv, timestamp

# Initialize session state variables
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "system_messages" not in st.session_state:
    st.session_state.system_messages = {}
    
    # Initialize default system messages for each model
    for model_name, model_info in MODELS.items():
        model_type = model_info["type"]
        st.session_state.system_messages[model_name] = DEFAULT_SYSTEM_MESSAGES.get(model_type, "You are a helpful assistant.")

# Title and description
st.title("🤖 Multi-Model AI Chat")
st.markdown("Chat with different AI models deployed on Cloudflare")

# Helper function to convert audio file to base64
def audio_to_base64(audio_file):
    try:
        # Read uploaded file
        audio_bytes = audio_file.getvalue()
        
        # Convert to base64
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        return base64_audio
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None

# Sidebar for model selection and configuration
with st.sidebar:
    st.header("Model Selection")
    selected_model_name = st.selectbox(
        "Choose a model:", 
        list(MODELS.keys())
    )
    
    selected_model = MODELS[selected_model_name]
    model_id = selected_model["id"]
    model_type = selected_model["type"]
    
    # Show model description
    st.markdown(f"**Description:** {selected_model['description']}")
    
    # Initialize chat history for the selected model if it doesn't exist
    if model_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[model_id] = []
    
    # Configuration options based on model type
    st.header("Configuration")
    
    if model_type == "translation":
        st.subheader("Translation Settings")
        source_lang = st.selectbox("Source Language:", list(LANGUAGES.keys()), index=0)
        target_lang = st.selectbox("Target Language:", list(LANGUAGES.keys()), index=1)
        
        # Convert language names to codes
        source_lang_code = LANGUAGES[source_lang]
        target_lang_code = LANGUAGES[target_lang]
    elif model_type == "chat":
        # System message for chat models
        st.subheader("System Message")
        system_message = st.text_area(
            "Customize the system message:",
            value=st.session_state.system_messages.get(selected_model_name, DEFAULT_SYSTEM_MESSAGES["chat"]),
            height=100
        )
        st.session_state.system_messages[selected_model_name] = system_message
    elif model_type == "speech":
        st.subheader("Speech Recognition Settings")
        speech_task = st.radio(
            "Task:",
            ["Transcribe", "Translate"],
            index=0
        ).lower()
        
        speech_lang = st.selectbox("Language:", list(LANGUAGES.keys()), index=0)
        speech_lang_code = LANGUAGES[speech_lang]
        
        use_vad = st.checkbox("Use Voice Activity Detection", value=False)
        vad_setting = "true" if use_vad else "false"
        
        initial_prompt = st.text_area(
            "Initial prompt (optional):",
            placeholder="Provide context to help the model understand the audio content",
            height=80
        )
    
    # Chat history management
    st.header("Chat Management")
    if st.button("Clear Current Chat History"):
        st.session_state.chat_histories[model_id] = []
        st.experimental_rerun()
    
    # Export chat history
    if st.session_state.chat_histories[model_id]:
        chat_csv, timestamp = export_chat_to_csv(st.session_state.chat_histories[model_id])
        if chat_csv:
            st.download_button(
                label="Download Chat History",
                data=chat_csv,
                file_name=f"chat_history_{selected_model_name.replace(' ', '_')}_{timestamp}.csv",
                mime="text/csv"
            )

# Current model's chat history
current_chat_history = st.session_state.chat_histories[model_id]

# Main chat interface
st.header(f"Chat with {selected_model_name}")

# Display model info
st.markdown(f"**Model ID:** `{model_id}`")
st.markdown(f"**Model Type:** {model_type.capitalize()}")

# Display current chat history
for message in current_chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input based on model type
if model_type == "speech":
    # Audio input for speech models
    st.subheader("Upload Audio")
    uploaded_audio = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])
    process_button = st.button("Process Audio")
    user_input = None  # Will be set later if audio is processed
    
    if uploaded_audio is not None and process_button:
        with st.spinner("Processing audio..."):
            # Convert audio to base64
            base64_audio = audio_to_base64(uploaded_audio)
            if base64_audio:
                user_input = f"Processed audio file: {uploaded_audio.name}"
else:
    # Text input for other models
    user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    current_chat_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process with selected model
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("Thinking...")
        
        try:
            if model_type == "translation":
                # Handle translation model
                input_data = {
                    "text": user_input,
                    "source_lang": source_lang_code,
                    "target_lang": target_lang_code
                }
                
                result = call_cloudflare_ai(model_id, input_data)
                translated_text = extract_response(result, "translation")
                
                assistant_response = f"**Translated from {source_lang} to {target_lang}:**\n\n{translated_text}"
            
            elif model_type == "chat":
                # Handle chat model
                # Prepare messages in the format expected by the model
                messages = [
                    {"role": "system", "content": st.session_state.system_messages[selected_model_name]}
                ]
                
                # Add previous conversation context (limited to prevent token limit issues)
                message_history = []
                for msg in current_chat_history[-10:]:
                    if msg["role"] in ["user", "assistant"]:
                        message_history.append(msg)
                
                messages.extend(message_history)
                
                input_data = {"messages": messages}
                result = call_cloudflare_ai(model_id, input_data)
                assistant_response = extract_response(result, "chat")
            
            elif model_type == "speech":
                # Handle speech recognition model
                if base64_audio:
                    input_data = {
                        "audio": base64_audio,
                        "task": speech_task,
                        "language": speech_lang_code,
                        "vad_filter": vad_setting
                    }
                    
                    # Add optional parameters if provided
                    if initial_prompt:
                        input_data["initial_prompt"] = initial_prompt
                    
                    result = call_cloudflare_ai(model_id, input_data)
                    transcription = extract_response(result, "speech")
                    
                    if speech_task == "transcribe":
                        assistant_response = f"**Transcription ({speech_lang}):**\n\n{transcription}"
                    else:
                        assistant_response = f"**Translation to English:**\n\n{transcription}"
                else:
                    assistant_response = "❌ **Error processing audio file**"
            
            else:
                # Fallback for unknown model types
                assistant_response = "⚠️ **Unsupported model type**"
            
            # Update placeholder with response
            message_placeholder.markdown(assistant_response)
            
            # Add assistant response to chat history
            current_chat_history.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            error_message = f"❌ **An error occurred:** {str(e)}"
            message_placeholder.markdown(error_message)
            current_chat_history.append({"role": "assistant", "content": error_message})

# Display additional information at the bottom
st.markdown("---")
st.markdown("**Note:** This app uses Cloudflare AI models through their API.")