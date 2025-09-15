import streamlit as st
import torch
from transformers import pipeline
from io import StringIO
from PyPDF2 import PdfReader
import base64

# Constants
IGNOU_LOGO_URL = "https://upload.wikimedia.org/wikipedia/en/thumb/3/3f/IGNOU_logo.svg/1200px-IGNOU_logo.svg.png"  
GOOGLE_FORM_LINK = "https://forms.gle/YOUR_GOOGLE_FORM_LINK_HERE"  # Replace with your actual Google Form URL

# Cache HuggingFace pipeline loading for performance
@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_resource(show_spinner=False)
def load_summarizer_pipeline():
    return pipeline("summarization", model="facebook/bart-large-cnn")

qa_model = load_qa_pipeline()
summarizer_model = load_summarizer_pipeline()

# Helper functions
def get_logo_markdown():
    logo_md = f"""
    <div style='text-align: center; margin-bottom: 1rem;'>
        <img src="{IGNOU_LOGO_URL}" width="90" alt="IGNOU logo" />
        <h2 style='margin-top: 0.5rem;'>Smart IGNOU Hackathon – AI Tutor App</h2>
    </div>
    """
    return logo_md

def render_home(student_data):
    st.markdown(get_logo_markdown(), unsafe_allow_html=True)
    st.write("### Welcome to AI Tutor App – Your personal learning assistant.")
    st.write("""
        This app supports IGNOU students with interactive AI-powered features:
        - AI Chatbot for instant answers to your questions.
        - Notes Summarizer to condense your study material.
        - Interactive Quiz Section with Google Forms.
        - Resources section with important PDFs.
    """)
    st.markdown("---")
    st.write("### Student Dashboard")

    if not student_data:
        with st.form("student_info_form"):
            student_name = st.text_input("Student Name")
            enrollment_number = st.text_input("Enrollment Number")
            college_name = st.text_input("College Name")
            semester = st.selectbox("Semester", [str(i) for i in range(1, 11)])
            submitted = st.form_submit_button("Save")
            if submitted:
                st.session_state["student_data"] = {
                    "Student Name": student_name,
                    "Enrollment Number": enrollment_number,
                    "College Name": college_name,
                    "Semester": semester,
                }
                st.success("Student details saved!")
    else:
        for label, value in student_data.items():
            st.text(f"{label}: {value}")

def render_chat():
    st.markdown(get_logo_markdown(), unsafe_allow_html=True)
    st.write("### AI Chatbot")
    st.write("Type your question below and get AI-generated answers.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Your question:", key="user_question")
    submit_button = st.button("Send")

    if submit_button and user_input.strip():
        st.session_state.chat_history.append({"sender": "User", "message": user_input.strip()})
        with st.spinner("Thinking..."):
            try:
                answer = qa_model(
                    question=user_input.strip(),
                    context="This is an AI tutor app helping IGNOU students with their questions."
                )
                ai_message = answer["answer"]
            except Exception:
                ai_message = "Sorry, I couldn't generate a response at this moment."
        st.session_state.chat_history.append({"sender": "AI", "message": ai_message})

    for chat in st.session_state.chat_history:
        if chat["sender"] == "User":
            st.markdown(
                f"<div style='background-color:#D6EAF8; text-align:right; padding: 8px; border-radius:10px; margin: 5px; max-width:80%; margin-left:auto;'>{chat['message']}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background-color:#EAECEE; text-align:left; padding: 8px; border-radius:10px; margin: 5px; max-width:80%; margin-right:auto;'>{chat['message']}</div>",
                unsafe_allow_html=True
            )

def render_notes_summarizer():
    st.markdown(get_logo_markdown(), unsafe_allow_html=True)
    st.write("### Notes Summarizer")
    st.write("Upload a PDF or text file to get a concise summary.")

    uploaded_file = st.file_uploader("Upload file (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        text_content = ""
        if uploaded_file.type == "application/pdf":
            try:
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            except Exception:
                st.error("Unable to read the PDF file.")
                return
        else:
            try:
                text_content = uploaded_file.getvalue().decode("utf-8")
            except Exception:
                st.error("Unable to read the text file.")
                return

        st.write("#### Preview of uploaded content (first 1000 characters):")
        st.text(text_content[:1000] + ("..." if len(text_content) > 1000 else ""))

        if st.button("Summarize"):
            with st.spinner("Generating summary..."):
                try:
                    max_input_chars = 1000
                    chunks = [text_content[i:i + max_input_chars] for i in range(0, len(text_content), max_input_chars)]
                    summaries = []
                    for chunk in chunks:
                        summary = summarizer_model(
                            chunk, max_length=130, min_length=30, do_sample=False
                        )[0]['summary_text']
                        summaries.append(summary)
                    full_summary = " ".join(summaries)
                    st.markdown("### Summary:")
                    st.write(full_summary)
                except Exception:
                    st.error("Error generating summary. Please try again with smaller file or simpler content.")

def render_quiz_section():
    st.markdown(get_logo_markdown(), unsafe_allow_html=True)
    st.write("### Quiz Section")

    iframe_code = f"""
    <iframe src="{GOOGLE_FORM_LINK}" width="100%" height="600" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
    """
    try:
        st.markdown(iframe_code, unsafe_allow_html=True)
    except:
        if st.button("Open Quiz"):
            js = f"window.open('{GOOGLE_FORM_LINK}')"
            st.components.v1.html(f"<script>{js}</script>")

def render_resources():
    st.markdown(get_logo_markdown(), unsafe_allow_html=True)
    st.write("### Resources")

    book_pdfs = st.text_area("Book PDFs (enter URLs or descriptions)")
    prev_year_pdfs = st.text_area("Previous Year Question PDFs (enter URLs or descriptions)")
    prev_year = st.text_input("Year for Previous Year Questions")

    st.write("---")
    st.write("You entered:")
    st.write(f"**Book PDFs:** {book_pdfs}")
    st.write(f"**Previous Year Question PDFs:** {prev_year_pdfs}")
    st.write(f"**Year:** {prev_year}")

def main():
    st.set_page_config(page_title="Smart IGNOU Hackathon – AI Tutor App", layout="wide", initial_sidebar_state="expanded")

    pages = {
        "Home": render_home,
        "AI Chatbot": render_chat,
        "Notes Summarizer": render_notes_summarizer,
        "Quiz Section": render_quiz_section,
        "Resources": render_resources,
    }

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Home"
    if "student_data" not in st.session_state:
        st.session_state.student_data = None

    with st.sidebar:
        st.markdown(get_logo_markdown(), unsafe_allow_html=True)
        selection = st.radio("Navigate", list(pages.keys()), index=list(pages.keys()).index(st.session_state.selected_page))
        st.session_state.selected_page = selection

    if st.session_state.selected_page == "Home":
        pages["Home"](st.session_state.student_data)
    else:
        pages[st.session_state.selected_page]()

    if st.session_state.selected_page == "Home":
        with st.expander("Deployment Instructions (Streamlit Cloud)"):
            st.markdown("""
            1. Save this script as `app.py`.
            2. Push to a GitHub repository.
            3. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.
            4. Click 'New app', connect your repo, choose branch and `app.py`.
            5. Deploy your app for free.
            6. Share your app URL.
            """)

if __name__ == "__main__":
    main()
