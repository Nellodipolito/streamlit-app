"""
Streamlit interface for Medical Guidelines QA system.
"""

import streamlit as st
import asyncio
from typing import Optional
from agent import OrchestratorQAAgent

# Page configuration
st.set_page_config(
    page_title="Medical Guidelines & Diagnostic AI",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'qa_agent' not in st.session_state:
    st.session_state.qa_agent = OrchestratorQAAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'diagnostic_history' not in st.session_state:
    st.session_state.diagnostic_history = []

# Navigation
st.sidebar.title("🏥 Medical AI Assistant")
page = st.sidebar.selectbox("Choose a function:", ["Q&A Chat", "Diagnostic Analysis", "About"])

if page == "Q&A Chat":
    # Existing Q&A functionality
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
                # Sources are already deduplicated in the agent
                for i, source in enumerate(item["sources"], 1):
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
                        line = f"({i}) [{display_title}]({link})"
                    else:
                        line = f"({i}) {display_title}"
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
                # Sources are already deduplicated in the agent
                for i, source in enumerate(result["sources"], 1):
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
                        line = f"({i}) [{display_title}]({link})"
                    else:
                        line = f"({i}) {display_title}"
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

elif page == "Diagnostic Analysis":
    # New diagnostic functionality
    st.title("🩺 AI Diagnostic Assistant")
    
    st.markdown("""
    Enter patient information to get AI-assisted diagnostic analysis. The system will:
    1. Extract key clinical findings from the patient data
    2. Search medical guidelines for relevant evidence
    3. Provide potential diagnoses with supporting evidence
    4. Suggest next steps and recommendations
    
    **⚠️ Important:** This is an AI-assisted tool for educational and research purposes. 
    Always consult with qualified healthcare professionals for actual medical decisions.
    """)
    
    # Sidebar for diagnostic settings
    with st.sidebar:
        st.header("🔧 Diagnostic Settings")
        diag_top_k = st.slider(
            "Evidence chunks to retrieve",
            min_value=5,
            max_value=30,
            value=15,
            help="Number of evidence chunks to retrieve for diagnostic analysis"
        )
        
        show_search_queries = st.checkbox(
            "Show search queries",
            value=False,
            help="Display the generated search queries used to find evidence"
        )
        
        show_evidence = st.checkbox(
            "Show retrieved evidence",
            value=False,
            help="Display the raw evidence chunks used for analysis"
        )
        
        if st.button("Clear Diagnostic History"):
            st.session_state.diagnostic_history = []
            st.rerun()
    
    # Quick start buttons
    st.subheader("🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Load Example Case"):
            st.session_state.example_loaded = True
    
    with col2:
        if st.button("❓ Show Predefined Questions"):
            st.session_state.show_questions = True
    
    with col3:
        if st.button("📋 Patient Data Template"):
            st.session_state.show_template = True
    
    # Show predefined questions
    if st.session_state.get('show_questions', False):
        st.info("""
        **Predefined Diagnostic Questions:**
        • What are the top 5 potential diagnoses for this case?
        • What are the differential diagnoses to consider?
        • What risk factors should be evaluated?
        • What additional tests or evaluations are recommended?
        • What is the most likely diagnosis based on the presentation?
        • What red flags or concerning symptoms should be monitored?
        • What treatment options should be considered?
        • What is the prognosis for the most likely diagnoses?
        
        Select a specific question below to focus your analysis, or leave blank for comprehensive analysis.
        """)
    
    # Question selection
    st.subheader("🎯 Focus Question (Optional)")
    predefined_questions = [
        "What are the top 5 potential diagnoses for this case?",
        "What are the differential diagnoses to consider?",
        "What risk factors should be evaluated?",
        "What additional tests or evaluations are recommended?",
        "What is the most likely diagnosis based on the presentation?",
        "What red flags or concerning symptoms should be monitored?",
        "What treatment options should be considered?",
        "What is the prognosis for the most likely diagnoses?"
    ]
    
    selected_question = st.selectbox(
        "Select a specific question to focus the analysis (optional):",
        [""] + predefined_questions,
        help="Choose a specific diagnostic question to focus on, or leave blank for comprehensive analysis"
    )
    
    if selected_question:
        st.info(f"🎯 **Focused Analysis**: {selected_question}")
    else:
        st.info("🔍 **Comprehensive Analysis**: Will provide general diagnostic analysis covering multiple aspects")

    # Patient data input form
    st.subheader("📊 Patient Information")
    
    with st.form("patient_data_form"):
        # Load example if requested
        if st.session_state.get('example_loaded', False):
            default_summary = """45-year-old male presents with chest pain that started 2 hours ago. Pain is described as crushing, substernal, radiating to left arm and jaw. Associated with diaphoresis, nausea, and shortness of breath. Patient has history of hypertension and smoking (1 pack/day for 20 years). No previous cardiac events. Vital signs: BP 160/95, HR 110, RR 22, O2 sat 96% on room air."""
            default_conversation = """Doctor: "Can you describe the chest pain?"
Patient: "It feels like someone is squeezing my chest really tight. It started suddenly while I was watching TV."
Doctor: "Does the pain go anywhere else?"
Patient: "Yes, it goes down my left arm and up to my jaw."
Doctor: "Any other symptoms?"
Patient: "I feel nauseous and sweaty, and I'm having trouble catching my breath."""
            default_age = 45
            default_gender = "Male"
            default_complaint = "Chest pain with radiation to left arm and jaw"
            st.session_state.example_loaded = False
        else:
            default_summary = ""
            default_conversation = ""
            default_age = None
            default_gender = ""
            default_complaint = ""
        
        patient_summary = st.text_area(
            "Patient Summary *",
            value=default_summary,
            height=150,
            help="Brief summary of patient's condition, symptoms, history, and examination findings"
        )
        
        doctor_conversation = st.text_area(
            "Doctor-Patient Conversation",
            value=default_conversation,
            height=100,
            help="Key points from doctor-patient conversation or interview"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=default_age)
        with col2:
            gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], index=0 if not default_gender else ["", "Male", "Female", "Other"].index(default_gender))
        
        chief_complaint = st.text_input(
            "Chief Complaint",
            value=default_complaint,
            help="Primary reason for visit or main symptom"
        )
        
        submitted = st.form_submit_button("🔍 Analyze Patient", type="primary")
    
    if submitted and patient_summary.strip():
        with st.spinner("Analyzing patient data and searching for evidence..."):
            try:
                # Prepare patient data
                patient_data = {
                    "patient_summary": patient_summary,
                    "doctor_conversation": doctor_conversation,
                    "age": age if age is not None and age > 0 else None,
                    "gender": gender if gender else None,
                    "chief_complaint": chief_complaint
                }
                
                # Run diagnostic analysis (get streaming result)
                result = asyncio.run(
                    st.session_state.qa_agent.diagnostic_agent.diagnose_patient(
                        patient_data=patient_data,
                        top_k=diag_top_k,
                        selected_question=selected_question if selected_question else None
                    )
                )
                
                # Display initial results with better styling
                st.success("✅ **Analysis Completed Successfully!**")
                st.markdown("---")
                
                # Create tabs for better organization
                tab1, tab2, tab3 = st.tabs(["📋 **Clinical Summary**", "🩺 **Diagnostic Analysis**", "📚 **Evidence Sources**"])
                
                with tab1:
                    # Patient summary
                    st.markdown("#### 👤 Processed Patient Summary")
                    with st.container():
                        st.info(result["patient_summary"])
                    
                    st.markdown("")  # Spacer
                    
                    # Key findings
                    st.markdown("#### 🔍 Key Clinical Findings")
                    with st.container():
                        for i, finding in enumerate(result["key_findings"], 1):
                            st.markdown(f"**{i}.** {finding}")
                
                with tab2:
                    # Streaming diagnosis analysis
                    st.markdown("#### 🩺 Analysis & Recommendations")
                    if selected_question:
                        st.markdown(f"*🎯 Focused on: {selected_question}*")
                        st.markdown("")
                
                    # Create placeholder for streaming response
                    diagnosis_placeholder = st.empty()
                    
                    # Process streaming response
                    async def process_diagnosis_stream():
                        diagnosis_text = ""
                        async for chunk in result["diagnosis_generator"]:
                            diagnosis_text += chunk
                            diagnosis_placeholder.markdown(diagnosis_text + "▌")
                        diagnosis_placeholder.markdown(diagnosis_text)
                        return diagnosis_text
                    
                    # Get the complete diagnosis text
                    complete_diagnosis = asyncio.run(process_diagnosis_stream())
                
                # Parse the completed diagnosis
                parsed_result = st.session_state.qa_agent.diagnostic_agent.parse_completed_diagnosis(
                    complete_diagnosis, result["evidence"]
                )
                
                # Update result with parsed diagnoses for history
                result_for_history = {
                    **result,
                    "recommendations": complete_diagnosis,
                    "potential_diagnoses": parsed_result["parsed_diagnoses"]
                }
                
                # Store in history
                st.session_state.diagnostic_history.append({
                    "patient_data": patient_data,
                    "result": result_for_history,
                    "timestamp": st.session_state.get('timestamp', 'now')
                })
                
                # Store current analysis in session state for follow-up persistence
                st.session_state.current_analysis = {
                    "result": result_for_history,
                    "patient_data": patient_data,
                    "selected_question": selected_question,
                    "diag_top_k": diag_top_k
                }
                
                with tab3:
                    # Sources section in the third tab
                    st.markdown("#### 📚 Evidence Sources")
                    st.caption("Medical literature used for this analysis")
                    
                    # Sources are already deduplicated in the agent
                    for i, source in enumerate(result["sources"], 1):
                        title = source.get('title', '')
                        year = source.get('year', '')
                        section = source.get('section', '')
                        link = source.get('link', '')
                        year_str = f" ({year})" if year else ""
                        section_str = f" - {section}" if section else ""
                        
                        with st.container():
                            if link:
                                st.markdown(f"**({i})** [{title}{year_str}{section_str}]({link})")
                            else:
                                st.markdown(f"**({i})** {title}{year_str}{section_str}")
                    
                    # Search queries used (if enabled)
                    if show_search_queries:
                        st.markdown("---")
                        st.markdown("#### 🔍 Search Queries Used")
                        with st.expander("View search queries", expanded=False):
                            for i, query in enumerate(result["search_queries"], 1):
                                st.code(f"{i}. {query}", language=None)
                    
                    # Show evidence chunks if enabled
                    if show_evidence:
                        st.markdown("---")
                        st.markdown("#### 📄 Retrieved Evidence Details")
                        for i, evidence in enumerate(result.get("evidence", []), 1):
                            title = evidence.get('title', '')
                            year = evidence.get('year', '')
                            link = evidence.get('link', '')
                            year_str = f" ({year})" if year else ""
                            
                            if link:
                                display_title = f"({i}) [{title}{year_str}]({link})"
                            else:
                                display_title = f"({i}) {title}{year_str}"
                                
                            with st.expander(f"Evidence {display_title}"):
                                st.markdown(f"**Section:** {evidence.get('header', '')}")
                                st.markdown(f"**Content:** {evidence.get('enriched_section_text', '')}")
                                st.markdown(f"**Search Query:** {evidence.get('search_query', '')}")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    elif submitted:
        st.warning("Please provide at least a patient summary to perform the analysis.")
    
    # Follow-up Questions Section (persistent across reruns)
    if st.session_state.get('current_analysis'):
        current_analysis = st.session_state.current_analysis
        result_for_history = current_analysis["result"]
        selected_question = current_analysis.get("selected_question")
        diag_top_k = current_analysis.get("diag_top_k", 15)
        
        # Create a clean separator
        st.markdown("---")
        
        # Header with clear context
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("❓ Follow-up Questions")
            st.caption("Ask additional questions about this diagnostic case")
        with col2:
            if st.button("🔄 New Analysis", type="secondary", help="Clear and start fresh analysis"):
                if 'current_analysis' in st.session_state:
                    del st.session_state.current_analysis
                if 'followup_results' in st.session_state:
                    del st.session_state.followup_results
                st.rerun()
        
        # Clean input section in a container
        with st.container():
            st.markdown("#### 💬 Ask a Question")
            
            # Text input with better styling
            follow_up_question = st.text_input(
                "",
                placeholder="💭 Type your follow-up question here (e.g., 'What additional tests should be ordered?')",
                key="followup_input",
                label_visibility="collapsed"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_followup = st.button("🔍 **Ask Question**", type="primary", key="ask_followup_btn", use_container_width=True)
            with col2:
                st.empty()  # Spacer
        
        # Quick suggestions in a clean expandable section
        follow_up_suggestions = [q for q in predefined_questions if q != selected_question][:6]
        
        if follow_up_suggestions:
            with st.expander("💡 **Quick Question Suggestions**", expanded=False):
                st.markdown("*Click any question below to ask it instantly:*")
                st.markdown("")
                
                # Create a grid of suggestion buttons
                for i in range(0, len(follow_up_suggestions), 2):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if i < len(follow_up_suggestions):
                            suggestion = follow_up_suggestions[i]
                            if st.button(
                                f"❓ {suggestion}", 
                                key=f"suggest_persistent_{i}",
                                use_container_width=True,
                                help=f"Click to ask: {suggestion}"
                            ):
                                st.session_state.pending_followup = suggestion
                                st.rerun()
                    
                    with col2:
                        if i + 1 < len(follow_up_suggestions):
                            suggestion = follow_up_suggestions[i + 1]
                            if st.button(
                                f"❓ {suggestion}", 
                                key=f"suggest_persistent_{i+1}",
                                use_container_width=True,
                                help=f"Click to ask: {suggestion}"
                            ):
                                st.session_state.pending_followup = suggestion
                                st.rerun()
        
        # Process follow-up question
        if ask_followup and follow_up_question.strip():
            with st.spinner("🔍 Analyzing your follow-up question..."):
                try:
                    followup_result = asyncio.run(
                        st.session_state.qa_agent.diagnostic_agent.answer_followup_question(
                            question=follow_up_question,
                            case_context=result_for_history,
                            top_k=diag_top_k
                        )
                    )
                    
                    # Store follow-up result in session state
                    if 'followup_results' not in st.session_state:
                        st.session_state.followup_results = []
                    st.session_state.followup_results.append(followup_result)
                    st.rerun()  # Refresh to show new result
                    
                except Exception as e:
                    st.error(f"❌ Error processing follow-up question: {str(e)}")
        
        # Handle pending suggestion from button click
        if st.session_state.get('pending_followup'):
            question = st.session_state.pending_followup
            st.session_state.pending_followup = None
            with st.spinner(f"🔍 Processing: {question[:50]}..."):
                try:
                    followup_result = asyncio.run(
                        st.session_state.qa_agent.diagnostic_agent.answer_followup_question(
                            question=question,
                            case_context=result_for_history,
                            top_k=diag_top_k
                        )
                    )
                    
                    # Store follow-up result in session state
                    if 'followup_results' not in st.session_state:
                        st.session_state.followup_results = []
                    st.session_state.followup_results.append(followup_result)
                    st.rerun()  # Refresh to show new result
                    
                except Exception as e:
                    st.error(f"❌ Error processing suggested question: {str(e)}")
        
        # Display follow-up results in a clean, organized way
        if st.session_state.get('followup_results'):
            st.markdown("---")
            st.markdown("#### 💡 Follow-up Answers")
            
            # Show results in reverse order (newest first)
            for i, followup_result in enumerate(reversed(st.session_state.followup_results), 1):
                question_preview = followup_result['question'][:60] + "..." if len(followup_result['question']) > 60 else followup_result['question']
                
                with st.expander(f"**Q{len(st.session_state.followup_results) - i + 1}:** {question_preview}", expanded=i == 1):
                    # Question header
                    st.markdown(f"**❓ Question:** {followup_result['question']}")
                    
                    # Analysis method indicator
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if followup_result["used_existing_data"]:
                            st.success("📋 Used existing data")
                        else:
                            st.info("🔍 New evidence search")
                    with col2:
                        st.empty()
                    
                    st.markdown("---")
                    
                    # Answer section
                    st.markdown("**💬 Answer:**")
                    st.markdown(followup_result["answer"])
                    
                    # Sources section (cleaner layout)
                    if followup_result["sources"]:
                        with st.expander("📚 **View Sources**", expanded=False):
                            for j, source in enumerate(followup_result["sources"], 1):
                                title = source.get('title', '')
                                year = source.get('year', '')
                                link = source.get('link', '')
                                section = source.get('section', '')
                                year_str = f" ({year})" if year else ""
                                section_str = f" - {section}" if section else ""
                                
                                if link:
                                    st.markdown(f"**({j})** [{title}{year_str}{section_str}]({link})")
                                else:
                                    st.markdown(f"**({j})** {title}{year_str}{section_str}")
                    
                    # Search query (only if new search was performed)
                    if followup_result["search_query"] and not followup_result["used_existing_data"]:
                        with st.expander("🔍 **Search Details**", expanded=False):
                            st.code(followup_result['search_query'], language=None)
            
            # Clear all follow-ups button at the bottom
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ **Clear All Follow-ups**", type="secondary", use_container_width=True):
                    if 'followup_results' in st.session_state:
                        del st.session_state.followup_results
                    st.rerun()
    
    # Display diagnostic history
    if st.session_state.diagnostic_history:
        st.subheader("📋 Previous Analyses")
        for i, item in enumerate(reversed(st.session_state.diagnostic_history), 1):
            with st.expander(f"Analysis {len(st.session_state.diagnostic_history) - i + 1}: {item['patient_data'].get('chief_complaint', 'Case Analysis')}"):
                st.markdown("**Patient Summary:**")
                st.write(item['patient_data']['patient_summary'])
                st.markdown("**Top Diagnoses:**")
                for j, diag in enumerate(item['result']['potential_diagnoses'][:3], 1):
                    st.write(f"{j}. {diag['condition']}")

elif page == "About":
    # About section
    st.title("📋 About Medical AI Assistant")
    
    # Introduction
    st.markdown("""
    ## 🎯 Overview
    
    The **Medical AI Assistant** is a comprehensive AI-powered system designed to support healthcare professionals 
    and researchers with evidence-based medical information and diagnostic assistance. Our platform combines 
    advanced AI technology with trusted medical sources to provide accurate, cited responses.
    """)
    
    # Key Features Section
    st.markdown("---")
    st.markdown("## ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 **Q&A Chat System**
        - **Medical Guidelines Search**: Query clinical guidelines and protocols
        - **MedlinePlus Integration**: Access NIH health information
        - **Evidence-Based Answers**: All responses include numbered citations
        - **Real-time Streaming**: Progressive answer generation
        - **Chat Memory**: Context-aware conversations
        
        ### 🩺 **AI Diagnostic Assistant**
        - **Patient Data Analysis**: Extract clinical findings from text
        - **Focused Questions**: Target specific diagnostic aspects
        - **Evidence-Based Diagnosis**: Backed by medical literature
        - **Multi-Query Search**: Comprehensive evidence gathering
        - **Follow-up Intelligence**: Smart question handling
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 **Smart Citation System**
        - **Numbered References**: (1), (2), (3) format
        - **Direct Links**: Clickable source URLs
        - **Source Verification**: Easy fact-checking
        - **Context Preservation**: Maintains reference accuracy
        
        ### 🧠 **Intelligent Memory**
        - **Context Awareness**: Remembers previous analysis
        - **Smart Routing**: Reuse vs new search decisions
        - **Conversation Flow**: Natural follow-up handling
        - **History Tracking**: Persistent case records
        """)
    
    # How It Works Section
    st.markdown("---")
    st.markdown("## 🏗️ How It Works")
    
    # Create tabs for different workflows
    tab1, tab2 = st.tabs(["🔍 **Q&A Workflow**", "🩺 **Diagnostic Workflow**"])
    
    with tab1:
        st.markdown("""
        ### Question & Answer Process
        
        1. **🎯 Query Processing**
           - User submits medical question
           - AI converts to optimized search query
           - System routes to appropriate data source
        
        2. **🔍 Evidence Search**
           - Vector similarity search through medical guidelines
           - MedlinePlus health topics search
           - Retrieval of most relevant sources
        
        3. **💬 Answer Generation**
           - AI synthesizes information from sources
           - Generates streaming response with citations
           - Provides numbered references to sources
        
        4. **📚 Source Display**
           - Shows all consulted sources with links
           - Enables easy verification and further reading
        """)
    
    with tab2:
        st.markdown("""
        ### Diagnostic Analysis Process
        
        1. **📊 Data Extraction**
           - Analyzes patient summary and conversations
           - Extracts key clinical findings
           - Identifies relevant demographics and history
        
        2. **🎯 Question Focus** *(Optional)*
           - User selects specific diagnostic question
           - System tailors analysis accordingly
           - Maintains comprehensive fallback option
        
        3. **🔍 Evidence Gathering**
           - Generates multiple targeted search queries
           - Searches medical literature for evidence
           - Deduplicates and ranks sources
        
        4. **🩺 Analysis Generation**
           - AI provides diagnostic reasoning
           - Includes numbered citations to evidence
           - Streams results in real-time
        
        5. **❓ Follow-up Intelligence**
           - Suggests related questions
           - Decides whether to reuse existing data
           - Enables deep-dive questioning
        """)
    
    # Technology Stack
    st.markdown("---")
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 AI & ML**
        - OpenAI GPT-4
        - Text Embedding Ada-002
        - Vector Similarity Search
        - Streaming Response Generation
        """)
    
    with col2:
        st.markdown("""
        **🔍 Search & Data**
        - Azure Cognitive Search
        - Medical Guidelines Index
        - MedlinePlus Health Topics
        - Vector Embeddings
        """)
    
    with col3:
        st.markdown("""
        **💻 Application**
        - Streamlit Frontend
        - FastAPI Backend
        - Pydantic Data Models
        - Async/Await Architecture
        """)
    
    # Data Sources
    st.markdown("---")
    st.markdown("## 📚 Data Sources")
    
    st.markdown("""
    ### 🏥 **Medical Guidelines**
    - Clinical practice guidelines from medical societies
    - Evidence-based treatment protocols
    - Diagnostic criteria and recommendations
    - Professional medical standards
    
    ### 🔬 **MedlinePlus (NIH)**
    - Trusted health information from the U.S. National Library of Medicine
    - Patient-friendly explanations of medical conditions
    - General health topics and wellness information
    - Regularly updated and medically reviewed content
    """)
    
    # Important Disclaimers
    st.markdown("---")
    st.markdown("## ⚠️ Important Disclaimers")
    
    st.warning("""
    **🏥 Medical Disclaimer:**
    
    This AI assistant is designed for **educational and research purposes only**. It is not intended to:
    - Replace professional medical advice
    - Provide definitive medical diagnoses
    - Substitute for clinical judgment
    - Offer treatment recommendations without physician oversight
    
    **Always consult qualified healthcare professionals for medical decisions.**
    """)
    
    st.info("""
    **🔍 AI Limitations:**
    
    - AI responses should be verified against official medical sources
    - System knowledge is based on training data with specific cutoff dates
    - Complex medical cases require human clinical expertise
    - Always review cited sources for accuracy and context
    """)
    
    # Usage Guidelines
    st.markdown("---")
    st.markdown("## 📖 Usage Guidelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ **Best Practices**
        - Use for medical education and research
        - Verify information with cited sources
        - Cross-reference with current guidelines
        - Combine with clinical expertise
        - Ask specific, detailed questions
        """)
    
    with col2:
        st.markdown("""
        ### ❌ **Avoid**
        - Using as sole basis for medical decisions
        - Bypassing professional medical consultation
        - Treating AI responses as definitive diagnoses
        - Ignoring cited source materials
        - Making treatment changes without physician approval
        """)
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Medical AI Assistant</strong> - Built with Azure Cognitive Search, OpenAI, FastAPI, and Streamlit</p>
        <p><em>Empowering healthcare professionals with AI-driven evidence-based information</em></p>
    </div>
    """, unsafe_allow_html=True) 