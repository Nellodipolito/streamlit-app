"""
Streamlit interface for Medical Guidelines QA system.
"""

import streamlit as st
import asyncio
from typing import Optional
from agent import OrchestratorQAAgent

# Page configuration
st.set_page_config(
    page_title="Medical Guidelines & MedlinePlusQA",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'qa_agent' not in st.session_state:
    st.session_state.qa_agent = OrchestratorQAAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

async def process_streaming_response(response_data: dict, message_placeholder):
    """Process streaming response and update UI."""
    answer_text = ""
    async for chunk in response_data["answer_generator"]:
        answer_text += chunk
        message_placeholder.markdown(answer_text + "▌")
    message_placeholder.markdown(answer_text)
    
    # Return complete response for history
    return {
        "question": response_data["question"],
        "answer": answer_text,
        "chunks": response_data["chunks"],
        "sources": response_data["sources"],
        "search_query": response_data["search_query"]
    }

# Main UI
st.title("🏥 Medical Guidelines & MedlinePlusQA")

st.markdown("""
Ask questions about medical guidelines or general medical topics and get evidence-based answers from our AI assistant.\n
The system searches through medical guidelines **and MedlinePlus health topics** to provide answers with citations.
\n
**MedlinePlus** is a trusted source of health information from the U.S. National Library of Medicine. Answers may include information from MedlinePlus as well as official medical guidelines.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        help="Control how many guideline chunks to retrieve for each question"
    )
    
    show_chunks = st.checkbox(
        "Show retrieved chunks",
        value=False,
        help="Display the raw guideline chunks used to generate the answer"
    )
    
    show_query = st.checkbox(
        "Show search query",
        value=False,
        help="Display the generated search query"
    )
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # Show all available guidelines in the index
    with st.expander("📖 View all available guidelines in the index"):
        with st.spinner("Loading available guidelines..."):
            try:
                guidelines = asyncio.run(st.session_state.qa_agent.guidelines_agent.list_all_guidelines())
                if guidelines:
                    for g in guidelines:
                        title = g.get('title', '')
                        link = g.get('link', '')
                        year = g.get('year', '')
                        display_title = f"{title} ({year})" if year else title
                        if link:
                            st.markdown(f"- [{display_title}]({link})")
                        else:
                            st.markdown(f"- {display_title}")
                else:
                    st.info("No guidelines found in the index.")
            except Exception as e:
                st.error(f"Error loading guidelines: {e}")

# Chat interface
chat_container = st.container()
with chat_container:
    for item in st.session_state.chat_history:
        # Display user question
        with st.chat_message("user"):
            st.markdown(item["question"])
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            
            # Show search query if enabled
            if show_query:
                st.markdown("**🔍 Search Query:**")
                st.info(item["search_query"])
            
            # Show sources
            st.markdown("**📚 Sources:**")
            # Deduplicate sources by (title, year, link)
            seen_sources = set()
            unique_sources = []
            for source in item["sources"]:
                key = (source.get("title"), source.get("year"), source.get("link", ""))
                if key not in seen_sources:
                    seen_sources.add(key)
                    unique_sources.append(source)
            for source in unique_sources:
                year = source.get('year', '')
                title = source.get('title', '')
                link = source.get('link') or source.get('url', '')
                # Only show year if present
                show_year = f" ({year})" if year else ""
                # Add MedlinePlus attribution if it's a Medline source (has url but no year)
                is_medline = source.get('url') and not year
                medline_suffix = " (MedlinePlus)" if is_medline else ""
                display_title = f"{title}{show_year}{medline_suffix}"
                if link:
                    line = f"- [{display_title}]({link})"
                else:
                    line = f"- {display_title}"
                st.markdown(line)
                header = source.get('header')
                if header:
                    st.markdown(f"  *Header: {header}*")
            
            # Show chunks if enabled
            if show_chunks:
                st.markdown("**📑 Retrieved Chunks:**")
                for i, chunk in enumerate(item["chunks"], 1):
                    title = chunk.get('title', '')
                    year = chunk.get('year', '')
                    header = chunk.get('header', '')
                    enriched_section_text = chunk.get('enriched_section_text', chunk.get('content', ''))
                    with st.expander(f"Chunk {i}: {title} ({year})"):
                        st.markdown(f"**Header:** {header}")
                        st.markdown(enriched_section_text)

# Question input
question = st.chat_input("Ask a medical question...")

if question:
    # Display user question
    with st.chat_message("user"):
        st.markdown(question)
    
    # Show processing message
    with st.chat_message("assistant"):
        with st.spinner("Searching guidelines..."):
            # Process question
            response = asyncio.run(
                st.session_state.qa_agent.answer_question(
                    question=question,
                    top_k=top_k,
                    chat_history=st.session_state.chat_history  # Pass chat history
                )
            )
            
            # Create placeholder for streaming response
            message_placeholder = st.empty()
            
            # Process streaming response
            result = asyncio.run(
                process_streaming_response(response, message_placeholder)
            )
            
            # Store in chat history
            st.session_state.chat_history.append(result)
            
            # Show search query if enabled
            if show_query:
                st.markdown("**🔍 Search Query:**")
                st.info(result["search_query"])
            
            # Show sources
            st.markdown("**📚 Sources:**")
            # Deduplicate sources by (title, year, link)
            seen_sources = set()
            unique_sources = []
            for source in result["sources"]:
                key = (source.get("title"), source.get("year"), source.get("link", ""))
                if key not in seen_sources:
                    seen_sources.add(key)
                    unique_sources.append(source)
            for source in unique_sources:
                year = source.get('year', '')
                title = source.get('title', '')
                link = source.get('link') or source.get('url', '')
                # Only show year if present
                show_year = f" ({year})" if year else ""
                # Add MedlinePlus attribution if it's a Medline source (has url but no year)
                is_medline = source.get('url') and not year
                medline_suffix = " (MedlinePlus)" if is_medline else ""
                display_title = f"{title}{show_year}{medline_suffix}"
                if link:
                    line = f"- [{display_title}]({link})"
                else:
                    line = f"- {display_title}"
                st.markdown(line)
                header = source.get('header')
                if header:
                    st.markdown(f"  *Header: {header}*")
            
            # Show chunks if enabled
            if show_chunks:
                st.markdown("**📑 Retrieved Chunks:**")
                for i, chunk in enumerate(result["chunks"], 1):
                    title = chunk.get('title', '')
                    year = chunk.get('year', '')
                    header = chunk.get('header', '')
                    enriched_section_text = chunk.get('enriched_section_text', chunk.get('content', ''))
                    with st.expander(f"Chunk {i}: {title} ({year})"):
                        st.markdown(f"**Header:** {header}")
                        st.markdown(enriched_section_text) 