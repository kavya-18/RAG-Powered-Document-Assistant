import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import io
import fitz

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")


def get_pdf_text(pdf_docs):
    text = ''
    for uploaded_file in pdf_docs:
        file_bytes = uploaded_file.read()
        pdf_stream = io.BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        for page in doc:
            text += page.get_text("text")
    return text


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks


def summarize_with_rag(text_chunks, summary_type="comprehensive", specific_topic=None):
    # Create vector store from chunks
    st.info("📦 Creating vector embeddings for summary...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Define different summary prompts with RAG context
    if specific_topic:
        # Topic-specific prompts
        prompt_templates = {
            "comprehensive": f"""
                Based on the following relevant context from the document, provide a comprehensive summary specifically about: "{specific_topic}".
                Focus on all information related to this topic, including main points, key findings, important details, and conclusions.
                If the topic is not covered in the context, clearly state that the topic was not found in the document.

                Context: {{context}}

                Comprehensive Summary about "{specific_topic}":
            """,

            "brief": f"""
                Based on the following relevant context from the document, provide a brief summary specifically about: "{specific_topic}".
                Focus only on the most important points related to this topic in 2-3 paragraphs.
                If the topic is not covered in the context, clearly state that the topic was not found.

                Context: {{context}}

                Brief Summary about "{specific_topic}":
            """,

            "bullet_points": f"""
                Based on the following relevant context from the document, create a bullet point summary specifically about: "{specific_topic}".
                Extract only the main ideas related to this topic and present them as clear, concise bullet points.
                If the topic is not covered in the context, clearly state that the topic was not found.

                Context: {{context}}

                Key Points about "{specific_topic}":
            """,

            "executive": f"""
                Based on the following relevant context from the document, provide an executive summary specifically about: "{specific_topic}".
                Focus on key insights, conclusions, and actionable information related to this topic that would be relevant for decision-making.
                If the topic is not covered in the context, clearly state that the topic was not found.

                Context: {{context}}

                Executive Summary about "{specific_topic}":
            """
        }

        # Use the specific topic as the main search query
        summary_queries = [
            specific_topic,
            f"{specific_topic} details",
            f"{specific_topic} findings",
            f"{specific_topic} information"
        ]
    else:
        # General summary prompts
        prompt_templates = {
            "comprehensive": """
                Based on the following relevant context from the document, provide a comprehensive summary. 
                Include all main points, key findings, important details, and conclusions found in the context.
                Make it detailed but well-organized:
                
                Context: {context}
                
                Comprehensive Summary:
            """,

            "brief": """
                Based on the following relevant context from the document, provide a brief summary. 
                Focus only on the most important points and key takeaways in 2-3 paragraphs:
                
                Context: {context}
                
                Brief Summary:
            """,

            "bullet_points": """
                Based on the following relevant context from the document, create a bullet point summary. 
                Extract the main ideas and present them as clear, concise bullet points:
                
                Context: {context}
                
                Key Points:
            """,

            "executive": """
                Based on the following relevant context from the document, provide an executive summary. 
                Focus on key insights, main conclusions, and actionable information relevant for decision-making:
                
                Context: {context}
                
                Executive Summary:
            """
        }

        # General summary queries
        summary_queries = [
            "main topics and key points",
            "important findings and conclusions",
            "significant details and insights",
            "recommendations and actionable items"
        ]

    model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    prompt = PromptTemplate(template=prompt_templates[summary_type], input_variables=["context"])

    # Retrieve relevant chunks using similarity search
    if specific_topic:
        st.info(f"🔍 Searching for content related to: '{specific_topic}'...")
    else:
        st.info("🔍 Retrieving relevant content for summary...")

    all_relevant_docs = []
    for query in summary_queries:
        docs = vector_store.similarity_search(query, k=4)
        all_relevant_docs.extend(docs)

    # Remove duplicates while preserving order
    seen_content = set()
    unique_docs = []
    for doc in all_relevant_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)

    # Combine relevant context
    context = "\n\n".join([doc.page_content for doc in unique_docs[:12]])  # Limit to top 12 unique chunks

    # Generate summary using RAG
    if specific_topic:
        st.info(f"✍️ Generating {summary_type} summary about '{specific_topic}' with RAG...")
    else:
        st.info(f"✍️ Generating {summary_type} summary with RAG...")

    formatted_prompt = prompt.format(context=context)
    response = model.invoke(formatted_prompt)

    return response.content, len(unique_docs)


def answer_question_with_rag(text_chunks, question: str):
    """New: Q&A over PDFs using RAG."""
    st.info("📦 Creating vector embeddings for Q&A...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)

    qa_prompt = PromptTemplate(
        template="""
You are a careful assistant that answers questions using ONLY the context below.
If the answer is not present in the context, reply:
"I couldn't find that information in the provided documents."

Context:
{context}

Question: {question}

Answer in a clear, concise way:
""",
        input_variables=["context", "question"],
    )

    # Retrieve chunks relevant to the user's question
    st.info("🔍 Retrieving relevant content for your question...")
    docs = vector_store.similarity_search(question, k=6)

    if not docs:
        return "I couldn't find that information in the provided documents.", 0

    context = "\n\n".join([d.page_content for d in docs])
    formatted_prompt = qa_prompt.format(context=context, question=question)
    response = model.invoke(formatted_prompt)

    return response.content, len(docs)


def main():
    st.set_page_config(page_title="PDF RAG Summarizer", page_icon="📄")
    st.header("📄 PDF RAG Summarizer using GPT")
    st.subheader("Upload PDFs and get intelligent summaries or Q&A using Retrieval-Augmented Generation")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("⚙️ RAG Options")

        # File uploader
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple PDF files at once"
        )

        # Mode selection
        mode = st.radio(
            "Choose what you want to do:",
            ["Summary", "Q&A (Ask Questions from PDFs)"],
            help="Select whether you want a summary or direct question answering from the PDFs",
        )

        summary_type = "comprehensive"
        if mode == "Summary":
            # Summary mode options
            summary_type = st.selectbox(
                "Choose summary type:",
                ["comprehensive", "brief", "bullet_points", "executive"],
                help="Select the type of summary you want"
            )

        # RAG parameters
        st.markdown("### 🔧 RAG Parameters")
        chunk_size = st.slider(
            "Chunk Size", 500, 2000, 1000, 100,
            help="Size of text chunks for embedding"
        )
        chunk_overlap = st.slider(
            "Chunk Overlap", 50, 400, 200, 50,
            help="Overlap between chunks"
        )

        # Action buttons
        summarize_button = None
        qa_button = None
        if mode == "Summary":
            summarize_button = st.button("Generate RAG Summary", type="primary", key="summary_btn")
        else:
            qa_button = st.button("Get Answer from PDFs", type="primary", key="qa_btn")

    # ---------- MAIN AREA ----------

    # Topic-specific summary input (only for summary mode)
    specific_topic = None
    if mode == "Summary":
        st.markdown("### 🎯 Topic-Specific Summary (Optional)")
        specific_topic = st.text_area(
            "Enter a specific topic, question, or theme you'd like to focus on:",
            help="The RAG system will search for content specifically related to your topic and generate a focused summary",
            height=120,
            key="topic_input"
        )

        if specific_topic.strip():
            st.success(f"🎯 Focus Topic: **{specific_topic.strip()}**")
            st.info("The system will search for content specifically related to this topic")
        else:
            st.info("General Summary Mode: Will summarize the entire document")

    # Question input for Q&A mode
    user_question = ""
    if mode == "Q&A (Ask Questions from PDFs)":
        st.markdown("### ❓ Ask a Question About Your PDF(s)")
        user_question = st.text_input(
            "Type your question (e.g., 'What is my white blood cell count?' or 'What is the test date?')",
            key="qa_question"
        )

    # ---------- LOGIC ----------

    if not pdf_docs:
        st.info("📥 Please upload PDF files using the sidebar to get started")
        st.markdown("---")
        st.markdown("Built with Streamlit, OpenAI GPT-4, FAISS Vector Store & RAG Pipeline")
        return

    # If PDFs are uploaded:
    if mode == "Summary" and summarize_button:
        with st.spinner("⏳ Processing PDFs with RAG pipeline for summary..."):
            try:
                # Extract text
                st.info("📄 Extracting text from PDFs...")
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.error("No text could be extracted from the PDFs. Please check if the PDFs contain readable text.")
                    return

                word_count = len(raw_text.split())
                st.info(f"✅ Extracted {word_count:,} words from {len(pdf_docs)} PDF(s)")

                # Chunking
                st.info("✂️ Creating text chunks for RAG...")
                text_chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.info(f"✅ Created {len(text_chunks)} text chunks for RAG processing")

                # RAG-based summary
                topic_to_use = specific_topic.strip() if (mode == "Summary" and specific_topic and specific_topic.strip()) else None
                summary, relevant_chunks_count = summarize_with_rag(text_chunks, summary_type, topic_to_use)

                # Display results
                st.success("✅ RAG Summary completed!")

                if topic_to_use:
                    st.markdown(f"### 📘 RAG Topic Summary: '{topic_to_use}'")
                else:
                    st.markdown("### 📘 RAG General Summary Results")

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", len(text_chunks))
                with col2:
                    st.metric("Relevant Chunks Used", relevant_chunks_count)
                with col3:
                    st.metric(
                        "Retrieval Efficiency",
                        f"{(relevant_chunks_count / len(text_chunks) * 100):.1f}%"
                    )

                # Expanders
                if topic_to_use:
                    with st.expander(f"📄 Generated Topic Summary: '{topic_to_use}'", expanded=True):
                        st.markdown(summary)
                else:
                    with st.expander("📄 Generated RAG Summary", expanded=True):
                        st.markdown(summary)

                # Download option
                summary_with_metadata = f"""RAG SUMMARY REPORT
Generated using Retrieval-Augmented Generation

Document(s): {len(pdf_docs)} PDF file(s)
Total Words: {word_count:,}
Total Chunks: {len(text_chunks)}
Relevant Chunks Used: {relevant_chunks_count}
Summary Type: {summary_type.title()}
Specific Topic: {topic_to_use if topic_to_use else 'General Summary'}
Chunk Size: {chunk_size}
Chunk Overlap: {chunk_overlap}

=== SUMMARY ===
{summary}
"""

                filename_suffix = f"_{topic_to_use.replace(' ', '_')}" if topic_to_use else ""
                st.download_button(
                    label="💾 Download RAG Summary Report",
                    data=summary_with_metadata,
                    file_name=f"rag_summary_{summary_type}{filename_suffix}.txt",
                    mime="text/plain"
                )

                # Chunk analysis
                with st.expander("📊 RAG Chunk Analysis"):
                    st.write("**RAG Processing Details:**")
                    if topic_to_use:
                        st.write(f"- **Topic Focus**: '{topic_to_use}'")
                        st.write("- Searched specifically for content related to this topic")
                    st.write(f"- Original document split into {len(text_chunks)} chunks of ~{chunk_size} characters each")
                    st.write(f"- System retrieved {relevant_chunks_count} most relevant chunks")
                    st.write(f"- Summary generated from the most pertinent content ({(relevant_chunks_count / len(text_chunks) * 100):.1f}% of total)")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.error("Please check your OpenAI API key and try again.")

    elif mode == "Q&A (Ask Questions from PDFs)" and qa_button:
        if not user_question.strip():
            st.warning("Please enter a question to ask about your PDFs.")
        else:
            with st.spinner("⏳ Processing PDFs with RAG pipeline for Q&A..."):
                try:
                    # Extract text
                    st.info("📄 Extracting text from PDFs...")
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDFs. Please check if the PDFs contain readable text.")
                        return

                    word_count = len(raw_text.split())
                    st.info(f"✅ Extracted {word_count:,} words from {len(pdf_docs)} PDF(s)")

                    # Chunking
                    st.info("✂️ Creating text chunks for RAG Q&A...")
                    text_chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.info(f"✅ Created {len(text_chunks)} text chunks for Q&A")

                    # Get answer with RAG
                    answer, used_chunks = answer_question_with_rag(text_chunks, user_question)

                    st.success("✅ Answer generated using RAG!")

                    st.markdown("### ❓ Your Question")
                    st.markdown(f"> {user_question}")

                    st.markdown("### 🧠 Answer")
                    st.write(answer)

                    # Simple metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Chunks", len(text_chunks))
                    with col2:
                        st.metric("Relevant Chunks Used", used_chunks)

                    with st.expander("📊 Q&A RAG Details"):
                        st.write("- Used similarity search over embedded chunks to find the most relevant parts of your PDFs.")
                        st.write(f"- Retrieved **{used_chunks}** chunks as context for this answer.")
                        st.write("- The model is instructed to answer **only** from the provided context and say if it cannot find the information.")

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.error("Please check your OpenAI API key and try again.")

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, OpenAI GPT-4, FAISS Vector Store & RAG Pipeline")


if __name__ == "__main__":
    main()
