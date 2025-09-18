# myapp.py

import streamlit as st
# CHANGE: Gemini ki library import karein
import google.generativeai as genai
from PyPDF2 import PdfReader
from transformers import pipeline

# ---------------------------
# Page Configuration (sabse upar hona chahiye)
# ---------------------------
st.set_page_config(
    page_title="IGNOU AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# API Key and Client Setup
# ---------------------------
# CHANGE: OpenAI ke bajaye Gemini API key setup karein
try:
    # Apni Gemini API key ko Streamlit Secrets mein save karein.
    # Key ka naam "GEMINI_API_KEY" hona chahiye.
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Gemini API key nahi mili. Kripya ise apne Streamlit Secrets mein 'GEMINI_API_KEY' naam se add karein.")
    st.stop()

# CHANGE: Gemini Model ko load karein (ek baar)
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel('gemini-1.5-flash-latest')

model = load_gemini_model()


# ---------------------------
# Loading Models (Cached for performance)
# ---------------------------
@st.cache_resource(show_spinner="Loading summarizer model...")
def load_summarizer_pipeline():
    """Loads the summarization model from HuggingFace."""
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer_model = load_summarizer_pipeline()

# ---------------------------
# Helper Functions
# ---------------------------
def display_header():
    """Displays the IGNOU logo and app title."""
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3f/IGNOU_logo.svg/1200px-IGNOU_logo.svg.png" width="90" alt="IGNOU logo">
            <h2>Smart IGNOU Hackathon – AI Tutor App</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------
# App Pages (Har page ke liye ek function)
# ---------------------------

def render_home():
    """Displays the Home Page."""
    display_header()
    st.write("### Welcome to AI Tutor App – Your personal learning assistant.")
    st.write(
        """
        This app supports IGNOU students with interactive AI-powered features:
        - *AI Chatbot*: Get instant answers to your study-related questions.
        - *Notes Summarizer*: Condense your PDF or text study material into key points.
        - *Quiz Section*: Test your knowledge with an interactive quiz.
        - *Resources*: A place to keep links to important study materials.
        """
    )
    st.markdown("---")

    # Student Dashboard Form
    st.write("### Student Dashboard")
    if "student_data" not in st.session_state:
        st.session_state.student_data = {}

    with st.form("student_info_form"):
        st.write("Enter your details to get started:")
        name = st.text_input("Student Name", st.session_state.student_data.get("Name", ""))
        enrollment = st.text_input("Enrollment Number", st.session_state.student_data.get("Enrollment", ""))
        college = st.text_input("College Name", st.session_state.student_data.get("College", ""))
        submitted = st.form_submit_button("Save Details")

        if submitted:
            st.session_state.student_data = {
                "Name": name,
                "Enrollment": enrollment,
                "College": college,
            }
            st.success("Details saved successfully!")
    
    if st.session_state.student_data:
        st.write("#### Your Saved Details:")
        st.json(st.session_state.student_data)


# CHANGE: Gemini API ke liye render_chat function ko poora update kiya gaya hai
def render_chat():
    """Displays the AI Chatbot Page using Gemini API."""
    display_header()
    st.write("### AI Chatbot")
    st.info("Ask any question related to your studies or general topics.")

    # Initialize chat history in session state for Gemini
    if "chat_session" not in st.session_state:
        # model.start_chat() ek conversation object banata hai jo history ko yaad rakhta hai
        st.session_state.chat_session = model.start_chat(history=[])

    # Display past messages from the chat session history
    # Shuruaati message display karein
    with st.chat_message("assistant"):
        st.markdown("Hello! How can I help you with your studies today?")

    for message in st.session_state.chat_session.history:
        # Gemini API 'model' role ka upyog karta hai, hum UI mein 'assistant' dikhayenge
        role = "assistant" if message.role == "model" else message.role
        with st.chat_message(role):
            st.markdown(message.parts[0].text)

    # Get user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response from Gemini
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    # send_message ka upyog karke Gemini ko prompt bhejein
                    response = st.session_state.chat_session.send_message(prompt)
                    full_response = response.text
                except Exception as e:
                    full_response = f"Error: Could not connect to Gemini. {e}"
            message_placeholder.markdown(full_response)


def render_notes_summarizer():
    """Displays the Notes Summarizer Page."""
    display_header()
    st.write("### Notes Summarizer")
    st.info("Upload a PDF or Text file to get a concise summary.")

    uploaded_file = st.file_uploader("Upload your notes (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                reader = PdfReader(uploaded_file)
                text_content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            else: # For .txt files
                text_content = uploaded_file.getvalue().decode("utf-8")
            
            if not text_content.strip():
                st.warning("Could not extract text from the file. It might be empty or an image-based PDF.")
                return

            st.text_area("Extracted Text (First 1000 characters)", text_content[:1000], height=200)

            if st.button("Generate Summary"):
                with st.spinner("Summarizing... This may take a moment."):
                    try:
                        summary = summarizer_model(text_content, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                        st.subheader("Summary")
                        st.success(summary)
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {e}")

        except Exception as e:
            st.error(f"Failed to process the file: {e}")


def render_quiz_section():
    """Displays the Quiz Section Page using a Google Form."""
    display_header()
    st.write("### Quiz Section")
    st.info("Test your knowledge by taking the quiz below.")
    
    # IMPORTANT: Replace this URL with your actual Google Form link
    google_form_url = "https://forms.gle/YOUR_GOOGLE_FORM_LINK_HERE" 
    
    st.markdown(f'<iframe src="{google_form_url}" width="100%" height="600" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>', unsafe_allow_html=True)


def render_resources():
    """Displays the Resources Page."""
    display_header()
    st.write("### Important Resources")
    st.info("You can save important links and notes here for quick access.")

    if "resources" not in st.session_state:
        st.session_state.resources = {
            "Book PDFs URLs": "",
            "Previous Year Papers URLs": ""
        }

    with st.form("resources_form"):
        books = st.text_area("Book PDFs (Enter URLs, one per line)", st.session_state.resources["Book PDFs URLs"])
        papers = st.text_area("Previous Year Question Papers (Enter URLs, one per line)", st.session_state.resources["Previous Year Papers URLs"])
        
        if st.form_submit_button("Save Resources"):
            st.session_state.resources["Book PDFs URLs"] = books
            st.session_state.resources["Previous Year Papers URLs"] = papers
            st.success("Resources saved!")

    st.write("#### Your Saved Resources:")
    st.json(st.session_state.resources)


# ---------------------------
# Main App Logic (Sidebar and Page Navigation)
# ---------------------------
def main():
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3f/IGNOU_logo.svg/1200px-IGNOU_logo.svg.png" width="70" alt="IGNOU logo">
                <h4>IGNOU AI Tutor</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        
        page_selection = st.radio(
            "Navigate",
            ["Home", "AI Chatbot", "Notes Summarizer", "Quiz Section", "Resources"]
        )
        st.markdown("---")
        st.info("Built for the Smart IGNOU Hackathon.")

    # Dictionary to map selections to functions
    pages = {
        "Home": render_home,
        "AI Chatbot": render_chat,
        "Notes Summarizer": render_notes_summarizer,
        "Quiz Section": render_quiz_section,
        "Resources": render_resources
    }
    
    # Run the function corresponding to the selection
    pages[page_selection]()

if _name_ == "_main_":
    main()
