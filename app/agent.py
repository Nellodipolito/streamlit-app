"""
QA Agent for medical guidelines using Azure Cognitive Search and LangChain.
"""

import os
import httpx
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

class GuidelinesQAAgent:
    """Agent for answering medical questions using guidelines."""
    
    def __init__(self):
        # Azure Search configuration
        self.service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
        self.admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        self.index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'guidelines-chunks-index')
        
        if not self.service_name or not self.admin_key:
            raise ValueError("Missing Azure Search configuration!")
        
        self.endpoint = f"https://{self.service_name}.search.windows.net"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.admin_key
        }
        
        # Initialize LangChain clients
        self.openai_client = self._initialize_openai_client()
        self.embeddings_client = self._initialize_embeddings_client()
        
        # System prompt for query generation
        self.query_system_prompt = """You are an expert at converting medical questions into search queries.\nYour task is to extract the key medical concepts and terms from the question to create an effective search query.\nFocus on medical terminology, conditions, treatments, and specific guideline aspects mentioned in the question.\nNever invent or fabricate information."""
        
        # System prompt for answer generation
        self.answer_system_prompt = """You are a medical expert assistant helping healthcare professionals understand medical guidelines.\nYour task is to provide accurate, well-structured answers based on the provided guideline excerpts.\nNever invent or fabricate information. If the answer is not in the provided content, say 'I don't know' or acknowledge the lack of information.\nAlways:\n1. Base your answers strictly on the provided guideline content\n2. Use numbered citations (1), (2), (3), etc. when referencing specific sources in your answer\n3. Cite the specific guideline and year when providing information\n4. Maintain medical accuracy and precision\n5. Acknowledge if information is not available in the provided excerpts\n6. Structure complex answers with clear sections\n7. Include relevant recommendations or evidence levels if present\n\nIMPORTANT: When referencing information from sources, include numbered citations in your response using the format (1), (2), (3), etc. For example: 'According to the guidelines (1), the recommended treatment is...' or 'Clinical studies show (2)...'"""

    def _initialize_openai_client(self):
        """Initialize LangChain Azure OpenAI client based on configuration."""
        # Check for Azure OpenAI configuration
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-02-01')
        
        if azure_endpoint and azure_api_key:
            return AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                deployment_name=azure_deployment,
            )
        else:
            raise ValueError("Missing Azure OpenAI configuration!")

    def _initialize_embeddings_client(self):
        """Initialize LangChain Azure OpenAI embeddings client."""
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-02-01')
        
        if azure_endpoint and azure_api_key:
            return AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                deployment="text-embedding-ada-002",
            )
        else:
            raise ValueError("Missing Azure OpenAI configuration!")

    async def generate_search_query(self, question: str) -> Tuple[str, List[float]]:
        """Generate an effective search query and embedding from the question."""
        try:
            # Generate a better search query using LangChain
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.query_system_prompt),
                HumanMessage(content=f"Convert this question into an effective search query: {question}")
            ]
            
            response = await self.openai_client.ainvoke(messages)
            search_query = response.content.strip()
            
            # Generate embedding for vector search using LangChain
            vector = await self.embeddings_client.aembed_query(search_query)
            
            return search_query, vector
            
        except Exception as e:
            raise Exception(f"Error generating search query: {e}")

    async def search_guidelines(self, query: str, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search the guidelines using vector search.
        
        Args:
            query: The search query text
            vector: The query embedding vector
            top_k: Number of results to return (default: 10)
        """
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
        
        # Prepare the search request
        search_body = {
            # 'search': query,  # <-- Commented out to disable textual search
            'search': '',      # Only use vector search
            "select": "pmid,pmcid,doi,title,year,link,mesh_terms,keywords,target_gender,min_target_age,max_target_age,location,resource_id,header,header_position,chunk_id,enriched_section_text",
            "top": top_k,  # Control overall number of results
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": vector,
                    "k": top_k,  # Control number of vector search results
                    "fields": "embedding"
                }
            ]
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=self.headers, json=search_body)
            if response.status_code == 200:
                return response.json().get('value', [])
            else:
                raise Exception(f"Search failed: {response.text}")
        except Exception as e:
            raise Exception(f"Error searching guidelines: {e}")

    async def list_all_guidelines(self, max_results: int = 2000) -> List[Dict[str, Any]]:
        """Fetch all unique guideline titles and links from the index, paging twice if needed, and cache in memory."""
        if hasattr(self, '_guidelines_cache') and self._guidelines_cache is not None:
            return self._guidelines_cache
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
        all_docs = []
        for skip in [0, 1000]:
            search_body = {
                'search': '*',
                'select': 'title,link,year',
                'top': 1000,
                'skip': skip
            }
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, headers=self.headers, json=search_body)
                if response.status_code == 200:
                    docs = response.json().get('value', [])
                    all_docs.extend(docs)
                else:
                    raise Exception(f"List guidelines failed: {response.text}")
            except Exception as e:
                raise Exception(f"Error listing guidelines: {e}")
        seen = set()
        unique_guidelines = []
        for doc in all_docs:
            title = doc.get('title', '')
            link = doc.get('link', '')
            year = doc.get('year', '')
            if title and title not in seen:
                seen.add(title)
                unique_guidelines.append({'title': title, 'link': link, 'year': year})
        self._guidelines_cache = unique_guidelines
        return unique_guidelines

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context for the answer generation."""
        # Deduplicate sources first to ensure correct numbering
        seen_sources = set()
        unique_results = []
        for result in search_results:
            key = (result.get('title', ''), result.get('year'), result.get('header', ''))
            if key not in seen_sources:
                seen_sources.add(key)
                unique_results.append(result)
        
        context_parts = []
        for i, result in enumerate(unique_results, 1):
            # Extract key information
            title = result.get('title', '')
            year = result.get('year', '')
            header = result.get('header', '')
            enriched_section_text = result.get('enriched_section_text', '')
            
            # Format the context entry with numbered reference
            context_entry = f"""
SOURCE ({i}): {title} ({year})
HEADER: {header}
CONTENT: {enriched_section_text}
---"""
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)

    def format_docs_with_id(self, results: List[Dict]) -> str:
        """Format documents with unique IDs for citation tracking, grouping by source link."""
        # Group chunks by their source link (same source = same link)
        grouped_by_link = {}
        for result in results:
            link = result.get('link', '')
            if link not in grouped_by_link:
                grouped_by_link[link] = []
            grouped_by_link[link].append(result)
        
        parts = []
        for i, (link, chunks) in enumerate(grouped_by_link.items(), 1):
            # Get metadata from first chunk (they should be the same for same source)
            first_chunk = chunks[0]
            title = first_chunk.get('title', '')
            year = first_chunk.get('year', '')
            
            # Combine all content from this source
            combined_content = []
            for chunk in chunks:
                header = chunk.get('header', '')
                content = chunk.get('enriched_section_text', '')
                if header:
                    combined_content.append(f"Section: {header}\n{content}")
                else:
                    combined_content.append(content)
            
            # Join all content with separators
            full_content = "\n\n---\n\n".join(combined_content)
            
            parts.append(
                f"SOURCE ID: {i}\n"
                f"TITLE:     {title} ({year})\n"
                f"CONTENT:   {full_content}"
            )
        
        return "\n\n".join(parts)

    async def generate_cited_answer(self, question: str, search_results: List[Dict]) -> Dict:
        """Generate an answer with structured citations using LangChain's structured output."""
        try:
            # 1) Format context
            formatted = self.format_docs_with_id(search_results)

            # 2) Build messages
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.answer_system_prompt),
                HumanMessage(content=
                    f"Here are your retrieved sources:\n\n{formatted}\n\n"
                    f"Question: {question}\n\n"
                    "Answer the user question based only on the given sources, and cite the sources used. "
                    "Please respond in the following JSON format:\n"
                    "{{\n"
                    '  "answer": "Your answer here with citations like (1), (2), etc.",\n'
                    '  "citations": [1, 2, 3]\n'
                    "}}"
                )
            ]

            # 3) Use regular LLM call with JSON response format
            response = await self.openai_client.ainvoke(messages)
            response_text = response.content.strip()
            
            # Debug: Log the raw response
            print(f"DEBUG: Raw LLM response: {response_text}")
            
            # 4) Parse the response manually
            try:
                import json
                import re
                
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed = json.loads(json_str)
                    
                    # Debug: Log the parsed JSON
                    print(f"DEBUG: Parsed JSON: {parsed}")
                    
                    # Validate the structure
                    if 'answer' in parsed and 'citations' in parsed:
                        answer = parsed['answer']
                        citations = parsed['citations']
                        
                        # Debug: Log the citations array
                        print(f"DEBUG: Citations array: {citations}")
                        
                        # 5) Map IDs back to metadata (using grouped sources)
                        # Group search results by link for proper mapping
                        grouped_by_link = {}
                        for result in search_results:
                            link = result.get('link', '')
                            if link not in grouped_by_link:
                                grouped_by_link[link] = result
                        
                        # Debug: Log the grouped sources
                        print(f"DEBUG: Grouped sources: {list(grouped_by_link.keys())}")
                        
                        mapped = []
                        for i, (link, result) in enumerate(grouped_by_link.items(), 1):
                            if i in citations:
                                mapped.append({
                                    "id": i,
                                    "title": result.get("title", ""),
                                    "year": result.get("year", ""),
                                    "link": link
                                })
                        
                        # Debug: Log the final mapped citations
                        print(f"DEBUG: Final mapped citations: {mapped}")
                        
                        return {
                            "answer": answer,
                            "citations": mapped
                        }
                    else:
                        raise ValueError("Invalid JSON structure")
                else:
                    # Fallback: treat the entire response as answer with no citations
                    return {
                        "answer": response_text,
                        "citations": []
                    }
                    
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: treat the entire response as answer with no citations
                return {
                    "answer": response_text,
                    "citations": []
                }
        except Exception as e:
            raise Exception(f"Error generating cited answer: {e}")

    async def generate_answer(self, question: str, context: str, chat_history=None):
        """Generate an answer using the retrieved context and chat history."""
        try:
            def build_history_prompt(chat_history, max_turns=30):
                if not chat_history:
                    return ""
                history = ""
                for turn in chat_history[-max_turns:]:
                    q = turn.get('question', '')
                    a = turn.get('answer', '')
                    history += f"User: {q}\nAssistant: {a}\n"
                return history

            history_prompt = build_history_prompt(chat_history)
            user_content = f"{history_prompt}\nQuestion: {question}\n\nRetrieved Guidelines:\n{context}\n\nPlease provide a comprehensive answer based on these guidelines."
            
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.answer_system_prompt),
                HumanMessage(content=user_content)
            ]

            # Use LangChain streaming
            async for chunk in self.openai_client.astream(messages):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            raise Exception(f"Error generating answer: {e}")

    async def answer_question(self, question: str, top_k: int = 10, chat_history=None) -> Dict[str, Any]:
        """Process a question and generate an answer.
        
        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve (default: 10)
            
        Returns:
            Dict containing:
            - question: Original question
            - search_query: Generated search query
            - chunks: Retrieved chunks
            - answer: Generated answer
            - sources: Source information
        """
        # Generate search query and embedding
        search_query, vector = await self.generate_search_query(question)
        
        # Search guidelines
        search_results = await self.search_guidelines(search_query, vector, top_k=top_k)
        
        if not search_results:
            return {
                "question": question,
                "search_query": search_query,
                "answer": "No relevant information found in the guidelines.",
                "chunks": [],
                "sources": []
            }
        
        # Format chunks and sources
        chunks = []
        for i, result in enumerate(search_results, 1):
            chunk = {
                "title": result.get('title', ''),
                "year": result.get('year', ''),
                "header": result.get('header', ''),
                "enriched_section_text": result.get('enriched_section_text', '')
            }
            chunks.append(chunk)
            print(f"\nChunk {i}:")
            print(f"Source: {chunk['title']} ({chunk['year']})")
            print(f"Header: {chunk['header']}")
            print("Content:")
            print("-" * 80)
            print(chunk['enriched_section_text'])
            print("-" * 80)
        
        sources = [{
            "title": result.get('title', ''),
            "year": result.get('year', ''),
            "header": result.get('header', ''),
            "link": result.get('link', '')
        } for result in search_results]
        
        # Generate the citation-aware answer
        cited_output = await self.generate_cited_answer(question, search_results)
        
        return {
            "question": question,
            "search_query": search_query,
            "answer": cited_output["answer"],
            "sources": cited_output["citations"],
            "chunks": chunks
        } 

class MedlineQAAgent:
    """Agent for answering general medical questions using MedlinePlus health topics."""
    def __init__(self):
        self.service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
        self.admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        self.index_name = os.getenv('AZURE_MEDLINE_INDEX_NAME', 'medline-health-topics-index')
        if not self.service_name or not self.admin_key:
            raise ValueError("Missing Azure Search configuration!")
        self.endpoint = f"https://{self.service_name}.search.windows.net"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.admin_key
        }
        self.openai_client = self._initialize_openai_client()
        self.embeddings_client = self._initialize_embeddings_client()
        self.query_system_prompt = """You are an expert at converting health questions into search queries. Extract the key medical concepts and terms from the question to create an effective search query for MedlinePlus health topics. Never invent or fabricate information."""
        self.answer_system_prompt = """You are a medical expert assistant helping users understand general health topics. Provide accurate, clear answers based on the provided MedlinePlus content. Never invent or fabricate information. If information is not available, say 'I don't know' or acknowledge it. Always cite the specific sources you use by referencing their source IDs."""

    def _initialize_openai_client(self):
        """Initialize LangChain Azure OpenAI client based on configuration."""
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-02-01')
        
        if azure_endpoint and azure_api_key:
            return AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                deployment_name=azure_deployment,
            )
        else:
            raise ValueError("Missing Azure OpenAI configuration!")

    def _initialize_embeddings_client(self):
        """Initialize LangChain Azure OpenAI embeddings client."""
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-02-01')
        
        if azure_endpoint and azure_api_key:
            return AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                deployment="text-embedding-ada-002",
            )
        else:
            raise ValueError("Missing Azure OpenAI configuration!")

    async def generate_search_query(self, question: str) -> Tuple[str, List[float]]:
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.query_system_prompt),
                HumanMessage(content=f"Convert this question into an effective search query: {question}")
            ]
            
            response = await self.openai_client.ainvoke(messages)
            search_query = response.content.strip()
            
            # Generate embedding for vector search using LangChain
            vector = await self.embeddings_client.aembed_query(search_query)
            
            return search_query, vector
        except Exception as e:
            raise Exception(f"Error generating search query: {e}")

    async def search_medline(self, query: str, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
        search_body = {
            'search': '',
            "select": "id,title,meta_desc,url,groups,also_called,full_summary,mesh_headings,see_references,language,content",
            "top": top_k,
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": vector,
                    "k": top_k,
                    "fields": "embedding"
                }
            ]
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=self.headers, json=search_body)
            if response.status_code == 200:
                return response.json().get('value', [])
            else:
                raise Exception(f"Search failed: {response.text}")
        except Exception as e:
            raise Exception(f"Error searching MedlinePlus: {e}")

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        context_parts = []
        for result in search_results:
            title = result.get('title', '')
            meta_desc = result.get('meta_desc', '')
            content = result.get('content', '')
            context_entry = f"""
TOPIC: {title}
SUMMARY: {meta_desc}
CONTENT: {content}
---"""
            context_parts.append(context_entry)
        return "\n".join(context_parts)

    def format_docs_with_id(self, results: List[Dict]) -> str:
        """Format documents with unique IDs for citation tracking, grouping by source URL."""
        # Group chunks by their source URL (same source = same URL)
        grouped_by_url = {}
        for result in results:
            url = result.get('url', '')
            if url not in grouped_by_url:
                grouped_by_url[url] = []
            grouped_by_url[url].append(result)
        
        parts = []
        for i, (url, chunks) in enumerate(grouped_by_url.items(), 1):
            # Get metadata from first chunk (they should be the same for same source)
            first_chunk = chunks[0]
            title = first_chunk.get('title', '')
            meta_desc = first_chunk.get('meta_desc', '')
            
            # Combine all content from this source
            combined_content = []
            for chunk in chunks:
                content = chunk.get('content', '')
                if content:
                    combined_content.append(content)
            
            # Join all content with separators
            full_content = "\n\n---\n\n".join(combined_content)
            
            parts.append(
                f"SOURCE ID: {i}\n"
                f"TITLE:     {title}\n"
                f"SUMMARY:   {meta_desc}\n"
                f"CONTENT:   {full_content}"
            )
        
        return "\n\n".join(parts)

    async def generate_cited_answer(self, question: str, search_results: List[Dict]) -> Dict:
        """Generate an answer with structured citations using LangChain's structured output."""
        try:
            # 1) Format context
            formatted = self.format_docs_with_id(search_results)

            # 2) Build messages
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.answer_system_prompt),
                HumanMessage(content=
                    f"Here are your retrieved sources:\n\n{formatted}\n\n"
                    f"Question: {question}\n\n"
                    "Answer the user question based only on the given sources, and cite the sources used. "
                    "Please respond in the following JSON format:\n"
                    "{{\n"
                    '  "answer": "Your answer here with citations like (1), (2), etc.",\n'
                    '  "citations": [1, 2, 3]\n'
                    "}}"
                )
            ]

            # 3) Use regular LLM call with JSON response format
            response = await self.openai_client.ainvoke(messages)
            response_text = response.content.strip()
            
            # Debug: Log the raw response
            print(f"DEBUG: Raw LLM response: {response_text}")
            
            # 4) Parse the response manually
            try:
                import json
                import re
                
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed = json.loads(json_str)
                    
                    # Debug: Log the parsed JSON
                    print(f"DEBUG: Parsed JSON: {parsed}")
                    
                    # Validate the structure
                    if 'answer' in parsed and 'citations' in parsed:
                        answer = parsed['answer']
                        citations = parsed['citations']
                        
                        # Debug: Log the citations array
                        print(f"DEBUG: Citations array: {citations}")
                        
                        # 5) Map IDs back to metadata (using grouped sources)
                        # Group search results by URL for proper mapping
                        grouped_by_url = {}
                        for result in search_results:
                            url = result.get('url', '')
                            if url not in grouped_by_url:
                                grouped_by_url[url] = result
                        
                        # Debug: Log the grouped sources
                        print(f"DEBUG: Grouped sources: {list(grouped_by_url.keys())}")
                        
                        mapped = []
                        for i, (url, result) in enumerate(grouped_by_url.items(), 1):
                            if i in citations:
                                mapped.append({
                                    "id": i,
                                    "title": result.get("title", ""),
                                    "url": url,
                                    "meta_desc": result.get("meta_desc", "")
                                })
                        
                        # Debug: Log the final mapped citations
                        print(f"DEBUG: Final mapped citations: {mapped}")
                        
                        return {
                            "answer": answer,
                            "citations": mapped
                        }
                    else:
                        raise ValueError("Invalid JSON structure")
                else:
                    # Fallback: treat the entire response as answer with no citations
                    return {
                        "answer": response_text,
                        "citations": []
                    }
                    
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: treat the entire response as answer with no citations
                return {
                    "answer": response_text,
                    "citations": []
                }
        except Exception as e:
            raise Exception(f"Error generating cited answer: {e}")

    async def generate_answer(self, question: str, context: str, chat_history=None):
        try:
            def build_history_prompt(chat_history, max_turns=30):
                if not chat_history:
                    return ""
                history = ""
                for turn in chat_history[-max_turns:]:
                    q = turn.get('question', '')
                    a = turn.get('answer', '')
                    history += f"User: {q}\nAssistant: {a}\n"
                return history

            history_prompt = build_history_prompt(chat_history)
            user_content = f"{history_prompt}\nQuestion: {question}\n\nRetrieved Health Topics:\n{context}\n\nPlease provide a comprehensive answer based on these health topics."
            
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.answer_system_prompt),
                HumanMessage(content=user_content)
            ]
            
            # Use LangChain streaming
            async for chunk in self.openai_client.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise Exception(f"Error generating answer: {e}")

    async def answer_question(self, question: str, top_k: int = 10, chat_history=None) -> Dict[str, Any]:
        search_query, vector = await self.generate_search_query(question)
        search_results = await self.search_medline(search_query, vector, top_k=top_k)
        if not search_results:
            return {
                "question": question,
                "search_query": search_query,
                "answer": "No relevant information found in MedlinePlus health topics.",
                "chunks": [],
                "sources": []
            }
        chunks = []
        for i, result in enumerate(search_results, 1):
            chunk = {
                "title": result.get('title', ''),
                "meta_desc": result.get('meta_desc', ''),
                "content": result.get('content', ''),
                "url": result.get('url', '')
            }
            chunks.append(chunk)
        sources = [{
            "title": result.get('title', ''),
            "url": result.get('url', '')
        } for result in search_results]
        # Generate the citation-aware answer
        cited_output = await self.generate_cited_answer(question, search_results)
        
        return {
            "question": question,
            "search_query": search_query,
            "answer": cited_output["answer"],
            "sources": cited_output["citations"],
            "chunks": chunks
        }

class DiagnosticAgent:
    """Agent for medical diagnosis assistance using patient data and evidence-based sources."""
    
    def __init__(self):
        # Reuse Azure Search configuration from GuidelinesQAAgent
        self.service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
        self.admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        self.index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'guidelines-chunks-index')
        
        if not self.service_name or not self.admin_key:
            raise ValueError("Missing Azure Search configuration!")
        
        self.endpoint = f"https://{self.service_name}.search.windows.net"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.admin_key
        }
        
        # Initialize LangChain clients
        self.openai_client = self._initialize_openai_client()
        self.embeddings_client = self._initialize_embeddings_client()
        
        # System prompts for different diagnostic tasks
        self.extraction_prompt = """You are a medical expert. Analyze the patient data and extract key clinical information.
        
        Extract the following:
        1. Chief complaint and primary symptoms
        2. Relevant medical history
        3. Physical examination findings
        4. Laboratory/diagnostic results
        5. Current medications
        6. Risk factors
        
        Be precise and focus on clinically relevant information. Do not speculate or add information not present in the data."""
        
        self.diagnosis_prompt = """You are an expert diagnostician. Based on the patient's clinical findings, provide a differential diagnosis.
        
        For each potential diagnosis:
        1. Provide clinical reasoning
        2. Identify supporting and opposing evidence
        
        Focus on evidence-based medicine and consider:
        - Symptom patterns and clinical presentation
        - Patient demographics and risk factors
        - Diagnostic criteria from medical guidelines
        
        Be conservative and acknowledge uncertainty when appropriate."""
        
        self.focused_question_prompt = """You are an expert diagnostician. Based on the patient's clinical findings and available medical evidence, answer the specific diagnostic question provided.
        
        Focus specifically on addressing the question asked. Be thorough but stay on topic.
        
        Always:
        1. Use numbered citations (1), (2), (3) when referencing evidence
        2. Base answers on the provided medical evidence
        3. Be precise and evidence-based
        4. Acknowledge limitations in available data
        
        Be conservative and acknowledge uncertainty when appropriate."""
        
        self.follow_up_decision_prompt = """You are a medical expert assistant. Analyze whether a follow-up question about a diagnostic case can be answered using existing information or requires new evidence search.
        
        Consider:
        1. Is the information needed already available in the previous analysis?
        2. Does the question require different medical evidence or guidelines?
        3. Is the question about the same patient case or a different aspect?
        
        Respond with either 'reuse' if existing information is sufficient, or 'new_search' if additional evidence is needed.
        
        Previous Analysis Context: {context}
        Follow-up Question: {question}
        
        Decision:"""
        
        self.evidence_search_prompt = """You are a medical search expert. Generate specific search queries to find evidence for potential diagnoses.
        
        Create search queries that focus on:
        1. Diagnostic criteria and guidelines
        2. Clinical presentation and symptoms
        3. Risk factors and epidemiology
        4. Differential diagnosis considerations
        
        Make queries specific and medical terminology focused."""

    def _initialize_openai_client(self):
        """Initialize LangChain Azure OpenAI client based on configuration."""
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-02-01')
        
        if azure_endpoint and azure_api_key:
            return AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                deployment_name=azure_deployment,
            )
        else:
            raise ValueError("Missing Azure OpenAI configuration!")

    def _initialize_embeddings_client(self):
        """Initialize LangChain Azure OpenAI embeddings client."""
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-02-01')
        
        if azure_endpoint and azure_api_key:
            return AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                deployment="text-embedding-ada-002",
            )
        else:
            raise ValueError("Missing Azure OpenAI configuration!")

    async def extract_key_findings(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key clinical findings from patient data."""
        try:
            # Combine patient data into a comprehensive summary
            patient_text = f"""
            Patient Summary: {patient_data.get('patient_summary', '')}
            Doctor-Patient Conversation: {patient_data.get('doctor_conversation', '')}
            Age: {patient_data.get('age', 'Not specified')}
            Gender: {patient_data.get('gender', 'Not specified')}
            Chief Complaint: {patient_data.get('chief_complaint', '')}
            """
            
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.extraction_prompt),
                HumanMessage(content=f"Extract key clinical findings from this patient data:\n{patient_text}")
            ]
            
            # Use streaming for faster response
            async def clinical_summary_stream():
                full_response = ""
                async for chunk in self.openai_client.astream(messages):
                    if chunk.content:
                        full_response += chunk.content
                        yield chunk.content
                
                # After streaming is complete, parse findings
                findings = self._parse_findings(full_response)
                
                # Store findings in session state for later retrieval
                import streamlit as st
                if 'clinical_findings' not in st.session_state:
                    st.session_state.clinical_findings = {}
                st.session_state.clinical_findings = {
                    "processed_summary": full_response,
                    "key_findings": findings
                }
            
            return {
                "summary_generator": clinical_summary_stream(),
                "patient_data": patient_data  # Keep original data for reference
            }
            
        except Exception as e:
            raise Exception(f"Error extracting findings: {e}")

    def _parse_findings(self, findings_text: str) -> List[str]:
        """Parse extracted findings into a list."""
        lines = findings_text.split('\n')
        findings = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                # Remove bullet points and numbering
                cleaned = line.lstrip('- •*').strip()
                if cleaned and len(cleaned) > 10:  # Filter out very short items
                    findings.append(cleaned)
        
        return findings[:10]  # Limit to top 10 findings

    async def generate_search_queries(self, findings: Dict[str, Any]) -> List[str]:
        """Generate search queries based on key findings."""
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.evidence_search_prompt),
                HumanMessage(content=f"Generate 5-7 specific search queries based on these clinical findings:\n{findings['processed_summary']}")
            ]
            
            response = await self.openai_client.ainvoke(messages)
            queries_text = response.content.strip()
            queries = [q.strip().lstrip('- •*1234567890.').strip() for q in queries_text.split('\n') if q.strip()]
            
            return queries[:7]  # Limit to 7 queries
            
        except Exception as e:
            raise Exception(f"Error generating search queries: {e}")

    async def search_for_evidence(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for diagnostic evidence using multiple queries."""
        all_results = []
        
        for query in queries:
            try:
                # Generate embedding for the query using LangChain
                vector = await self.embeddings_client.aembed_query(query)
                
                # Search the guidelines index
                url = f"{self.endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
                search_body = {
                    'search': '',
                    "select": "pmid,pmcid,doi,title,year,link,mesh_terms,keywords,target_gender,min_target_age,max_target_age,location,resource_id,header,header_position,chunk_id,enriched_section_text",
                    "top": top_k,
                    "vectorQueries": [
                        {
                            "kind": "vector",
                            "vector": vector,
                            "k": top_k,
                            "fields": "embedding"
                        }
                    ]
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, headers=self.headers, json=search_body)
                
                if response.status_code == 200:
                    results = response.json().get('value', [])
                    for result in results:
                        result['search_query'] = query  # Track which query found this
                    all_results.extend(results)
                    
            except Exception as e:
                print(f"Error searching for query '{query}': {e}")
                continue
        
        # Deduplicate results by chunk_id
        seen_chunks = set()
        unique_results = []
        for result in all_results:
            chunk_id = result.get('chunk_id')
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        return unique_results

    def format_evidence_with_id(self, evidence: List[Dict]) -> str:
        """Format evidence with unique IDs, grouping by link."""
        grouped_by_link = {}
        for item in evidence:
            link = item.get('link', '')
            if link not in grouped_by_link:
                grouped_by_link[link] = []
            grouped_by_link[link].append(item)
        
        parts = []
        for i, (link, chunks) in enumerate(grouped_by_link.items(), 1):
            first_chunk = chunks[0]
            title = first_chunk.get('title', '')
            year = first_chunk.get('year', '')
            
            combined_content = []
            for chunk in chunks:
                header = chunk.get('header', '')
                content = chunk.get('enriched_section_text', '')
                if header:
                    combined_content.append(f"Section: {header}\n{content}")
                else:
                    combined_content.append(content)
            full_content = "\n\n---\n\n".join(combined_content)
            parts.append(
                f"EVIDENCE ID: {i}\n"
                f"TITLE:     {title} ({year})\n"
                f"CONTENT:   {full_content}"
            )
        return "\n\n".join(parts)

    async def generate_focused_analysis(self, findings: Dict[str, Any], evidence: List[Dict[str, Any]], selected_question: str) -> Dict[str, Any]:
        """Generate analysis focused on a specific diagnostic question."""
        try:
            # Group evidence by link for deduplication and joining
            grouped_by_link = {}
            for item in evidence:
                link = item.get('link', '')
                if link not in grouped_by_link:
                    grouped_by_link[link] = []
                grouped_by_link[link].append(item)
            
            # Format evidence for LLM with numbered references
            evidence_text = self.format_evidence_with_id(evidence)
            
            prompt = f"""
            Patient Clinical Findings:
            {findings['processed_summary']}
            
            Available Medical Evidence:
            {evidence_text}
            
            Specific Question to Answer: {selected_question}
            
            Based on the clinical findings and medical evidence, provide a focused answer to the specific question above.
            
            IMPORTANT: When referencing information from the evidence sources, include numbered citations in your response using the format (1), (2), (3), etc. corresponding to the evidence numbers above.
            
            Format your response clearly and focus specifically on answering the question asked.
            """
            
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.focused_question_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Use streaming for faster response
            async def diagnosis_stream():
                full_response = ""
                async for chunk in self.openai_client.astream(messages):
                    if chunk.content:
                        full_response += chunk.content
                        # Display the chunk as is for streaming
                        yield chunk.content
                
                # After streaming is complete, parse citations from the full response
                try:
                    import re
                    
                    # Extract citation numbers from the response using regex
                    # Look for patterns like (1), (2), (3), etc.
                    citation_pattern = r'\((\d+)\)'
                    citations = re.findall(citation_pattern, full_response)
                    
                    # Convert to integers and remove duplicates while preserving order
                    citation_numbers = []
                    for citation in citations:
                        num = int(citation)
                        if num not in citation_numbers:
                            citation_numbers.append(num)
                    
                    # Debug: Log the citations found
                    print(f"DEBUG: Citations found in response: {citation_numbers}")
                    
                    # Map IDs back to metadata (using grouped sources)
                    mapped = []
                    for i, (link, chunks) in enumerate(grouped_by_link.items(), 1):
                        if i in citation_numbers:
                            first_chunk = chunks[0]
                            mapped.append({
                                "id": i,
                                "title": first_chunk.get("title", ""),
                                "year": first_chunk.get("year", ""),
                                "link": link
                            })
                    
                    # Debug: Log the final mapped citations
                    print(f"DEBUG: Final mapped diagnostic citations: {mapped}")
                    
                    # Store citations in session state for later retrieval
                    import streamlit as st
                    if 'diagnostic_citations' not in st.session_state:
                        st.session_state.diagnostic_citations = {}
                    st.session_state.diagnostic_citations['focused'] = mapped
                            
                except Exception as e:
                    print(f"DEBUG: Error parsing citations: {e}")
                    st.session_state.diagnostic_citations = {}
            
            # Return the generator for streaming and grouped evidence for citation mapping
            return {
                "diagnosis_generator": diagnosis_stream(),
                "evidence": [chunks[0] for link, chunks in grouped_by_link.items()]  # All evidence for reference
            }
        except Exception as e:
            raise Exception(f"Error generating focused analysis: {e}")

    async def generate_diagnoses(self, findings: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate potential diagnoses based on findings and evidence."""
        try:
            # Group evidence by link for deduplication and joining
            grouped_by_link = {}
            for item in evidence:
                link = item.get('link', '')
                if link not in grouped_by_link:
                    grouped_by_link[link] = []
                grouped_by_link[link].append(item)
            
            # Format evidence for LLM with numbered references
            evidence_text = self.format_evidence_with_id(evidence)
            
            prompt = f"""
            Patient Clinical Findings:
            {findings['processed_summary']}
            
            Available Medical Evidence:
            {evidence_text}
            
            Based on the clinical findings and medical evidence, provide:
            1. Clinical reasoning for each diagnosis
            2. Supporting evidence from the provided sources
            3. Recommendations for further evaluation or management
            
            IMPORTANT: When referencing information from the evidence sources, include numbered citations in your response using the format (1), (2), (3), etc. corresponding to the evidence numbers above. For example: "According to clinical guidelines (1), chest pain with radiation suggests..." or "The diagnostic criteria include (2)..."
            
            Format your response clearly with numbered diagnoses and structured reasoning. Always cite your sources with numbered references.
            """
            
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=self.diagnosis_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Use streaming for faster response
            async def diagnosis_stream():
                full_response = ""
                async for chunk in self.openai_client.astream(messages):
                    if chunk.content:
                        full_response += chunk.content
                        # Display the chunk as is for streaming
                        yield chunk.content
                
                # After streaming is complete, parse citations from the full response
                try:
                    import re
                    
                    # Extract citation numbers from the response using regex
                    # Look for patterns like (1), (2), (3), etc.
                    citation_pattern = r'\((\d+)\)'
                    citations = re.findall(citation_pattern, full_response)
                    
                    # Convert to integers and remove duplicates while preserving order
                    citation_numbers = []
                    for citation in citations:
                        num = int(citation)
                        if num not in citation_numbers:
                            citation_numbers.append(num)
                    
                    # Debug: Log the citations found
                    print(f"DEBUG: Citations found in response: {citation_numbers}")
                    
                    # Map IDs back to metadata (using grouped sources)
                    mapped = []
                    for i, (link, chunks) in enumerate(grouped_by_link.items(), 1):
                        if i in citation_numbers:
                            first_chunk = chunks[0]
                            mapped.append({
                                "id": i,
                                "title": first_chunk.get("title", ""),
                                "year": first_chunk.get("year", ""),
                                "link": link
                            })
                    
                    # Debug: Log the final mapped citations
                    print(f"DEBUG: Final mapped diagnosis citations: {mapped}")
                    
                    # Store citations in session state for later retrieval
                    import streamlit as st
                    if 'diagnostic_citations' not in st.session_state:
                        st.session_state.diagnostic_citations = {}
                    st.session_state.diagnostic_citations['comprehensive'] = mapped
                            
                except Exception as e:
                    print(f"DEBUG: Error parsing citations: {e}")
                    st.session_state.diagnostic_citations = {}
            
            # Return the generator for streaming and grouped evidence for citation mapping
            return {
                "diagnosis_generator": diagnosis_stream(),
                "evidence": [chunks[0] for link, chunks in grouped_by_link.items()]  # All evidence for reference
            }
        except Exception as e:
            raise Exception(f"Error generating diagnoses: {e}")

    def _parse_diagnoses(self, diagnosis_text: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse diagnosis text into structured format."""
        diagnoses = []
        
        # Simple parsing - look for numbered items
        sections = diagnosis_text.split('\n\n')
        
        for section in sections:
            if any(section.strip().startswith(str(i)) for i in range(1, 6)):
                lines = section.strip().split('\n')
                if lines:
                    # Extract condition name from first line
                    first_line = lines[0]
                    condition = first_line.split(':', 1)[-1].strip() if ':' in first_line else first_line
                    condition = condition.replace('1.', '').replace('2.', '').replace('3.', '').replace('4.', '').replace('5.', '').strip()
                    
                    diagnoses.append({
                        "condition": condition,
                        "reasoning": section,
                        "supporting_evidence": []  # Could be enhanced to link specific evidence
                    })
        
        return diagnoses[:5]  # Limit to top 5

    def parse_completed_diagnosis(self, diagnosis_text: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse completed diagnosis text into structured format."""
        return {
            "diagnosis_analysis": diagnosis_text,
            "parsed_diagnoses": self._parse_diagnoses(diagnosis_text, evidence)
        }

    async def should_reuse_for_followup(self, question: str, case_context: Dict[str, Any]) -> bool:
        """Determine if follow-up question can be answered with existing data."""
        try:
            # Build context summary
            context_summary = f"""
            Patient Summary: {case_context.get('patient_summary', '')}
            Key Findings: {', '.join(case_context.get('key_findings', []))}
            Previous Analysis: {case_context.get('recommendations', '')[:500]}...
            Available Sources: {len(case_context.get('sources', []))} medical sources
            """
            
            prompt = self.follow_up_decision_prompt.format(
                context=context_summary,
                question=question
            )
            
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content="You are a helpful assistant that decides if existing information is sufficient."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.openai_client.ainvoke(messages)
            decision = response.content.strip().lower()
            return "reuse" in decision
            
        except Exception as e:
            print(f"Error in follow-up decision: {e}")
            return False

    async def answer_followup_question(self, question: str, case_context: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """Answer a follow-up question using existing context or new search."""
        try:
            # Decide whether to reuse existing data
            should_reuse = await self.should_reuse_for_followup(question, case_context)
            
            if should_reuse:
                # Use existing evidence and context
                existing_evidence = case_context.get('evidence', [])
                existing_sources = case_context.get('sources', [])
                
                # Format existing evidence for LLM
                evidence_text = ""
                for i, source in enumerate(existing_sources, 1):
                    evidence_text += f"""
                    Evidence ({i}):
                    Source: {source.get('title', '')} ({source.get('year', '')})
                    Section: {source.get('section', '')}
                    Content: [From previous analysis]
                    ---
                    """
                
                prompt = f"""
                Patient Clinical Context:
                {case_context.get('patient_summary', '')}
                
                Previous Analysis Context:
                {case_context.get('recommendations', '')}
                
                Available Evidence Sources:
                {evidence_text}
                
                Follow-up Question: {question}
                
                Based on the existing analysis and context, provide a focused answer to the follow-up question.
                
                IMPORTANT: When referencing information, include numbered citations (1), (2), (3), etc. corresponding to the evidence sources above.
                """
                
                from langchain.schema import HumanMessage, SystemMessage
                
                messages = [
                    SystemMessage(content=self.focused_question_prompt),
                    HumanMessage(content=prompt)
                ]
                
                # Collect streamed response
                answer = ""
                async for chunk in self.openai_client.astream(messages):
                    if chunk.content:
                        answer += chunk.content
                
                return {
                    "question": question,
                    "answer": answer,
                    "used_existing_data": True,
                    "sources": existing_sources,
                    "search_query": None
                }
                
            else:
                # Need new search - generate search queries for the follow-up question
                from langchain.schema import HumanMessage, SystemMessage
                
                search_messages = [
                    SystemMessage(content=self.evidence_search_prompt),
                    HumanMessage(content=f"Generate 3-5 specific search queries for this follow-up question: {question}")
                ]
                
                search_query_response = await self.openai_client.ainvoke(search_messages)
                queries_text = search_query_response.content.strip()
                queries = [q.strip().lstrip('- •*1234567890.').strip() for q in queries_text.split('\n') if q.strip()][:5]
                
                # Search for new evidence
                new_evidence = await self.search_for_evidence(queries, top_k=top_k//len(queries) + 1)
                
                if not new_evidence:
                    return {
                        "question": question,
                        "answer": "No relevant additional information found to answer this follow-up question.",
                        "used_existing_data": False,
                        "sources": [],
                        "search_query": "; ".join(queries)
                    }
                
                # Format new evidence and generate answer
                seen_sources = set()
                unique_evidence = []
                for item in new_evidence:
                    key = (item.get('title', ''), item.get('year'), item.get('header', ''))
                    if key not in seen_sources:
                        seen_sources.add(key)
                        unique_evidence.append(item)
                
                evidence_text = ""
                for i, item in enumerate(unique_evidence, 1):
                    evidence_text += f"""
                    Evidence ({i}):
                    Source: {item.get('title', '')} ({item.get('year', '')})
                    Section: {item.get('header', '')}
                    Content: {item.get('enriched_section_text', '')}
                    ---
                    """
                
                prompt = f"""
                Original Patient Context:
                {case_context.get('patient_summary', '')}
                
                New Medical Evidence:
                {evidence_text}
                
                Follow-up Question: {question}
                
                Based on the new medical evidence, provide a focused answer to the follow-up question.
                
                IMPORTANT: When referencing information, include numbered citations (1), (2), (3), etc. corresponding to the evidence numbers above.
                """
                
                from langchain.schema import HumanMessage, SystemMessage
                
                response_messages = [
                    SystemMessage(content=self.focused_question_prompt),
                    HumanMessage(content=prompt)
                ]
                
                # Collect streamed response
                answer = ""
                async for chunk in self.openai_client.astream(response_messages):
                    if chunk.content:
                        answer += chunk.content
                
                # Format new sources
                new_sources = []
                for item in unique_evidence:
                    new_sources.append({
                        "title": item.get('title', ''),
                        "year": item.get('year'),
                        "journal": None,
                        "section": item.get('header', ''),
                        "link": item.get('link', '')
                    })
                
                return {
                    "question": question,
                    "answer": answer,
                    "used_existing_data": False,
                    "sources": new_sources,
                    "search_query": "; ".join(queries)
                }
                
        except Exception as e:
            raise Exception(f"Error answering follow-up question: {e}")

    async def diagnose_patient(self, patient_data: Dict[str, Any], top_k: int = 10, selected_question: str = None) -> Dict[str, Any]:
        """Main method to perform diagnostic analysis."""
        try:
            # Step 1: Extract key findings (now streaming)
            findings_result = await self.extract_key_findings(patient_data)
            
            # Step 2: Generate search queries (still synchronous for now)
            # We need to get the processed summary from session state or run a quick extraction
            # For now, let's use the original patient summary for search queries
            patient_summary = patient_data.get('patient_summary', '')
            search_queries = await self.generate_search_queries({"processed_summary": patient_summary})
            
            # Step 3: Search for evidence
            evidence = await self.search_for_evidence(search_queries, top_k=top_k//len(search_queries) + 1)
            
            # Step 4: Generate diagnoses (streaming format with citation parsing)
            if selected_question:
                diagnosis_result = await self.generate_focused_analysis(
                    {"processed_summary": patient_summary}, evidence, selected_question
                )
            else:
                diagnosis_result = await self.generate_diagnoses(
                    {"processed_summary": patient_summary}, evidence
                )
            
            # Step 5: Return streaming format with both clinical summary and diagnosis
            return {
                "patient_data": patient_data,
                "search_queries": search_queries,
                "clinical_summary_generator": findings_result["summary_generator"],
                "diagnosis_generator": diagnosis_result["diagnosis_generator"],
                "evidence": diagnosis_result["evidence"]  # All evidence for reference
            }
            
        except Exception as e:
            raise Exception(f"Error in diagnostic analysis: {e}")

class OrchestratorQAAgent:
    """Orchestrates which agent to use based on the user question, and decides if a new search is needed."""
    def __init__(self):
        self.guidelines_agent = GuidelinesQAAgent()
        self.medline_agent = MedlineQAAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.openai_client = self.guidelines_agent.openai_client
        self.routing_prompt = (
            "You are an expert assistant. Classify the following user question as either "
            "'guidelines' if it is about clinical guidelines, recommendations, or evidence-based protocols, "
            "or 'medline' if it is a general health or medical information question.\n"
            "Respond with only 'guidelines' or 'medline'.\n"
            "Question: {question}"
        )
        self.reuse_decision_prompt = (
            "You are a helpful assistant. Given the user's new question and the previous conversation, "
            "decide if the new question can be answered using only the information already retrieved in the previous conversation. "
            "If yes, respond with 'reuse'. If not, respond with 'new search'.\n"
            "Previous conversation:\n{history}\nNew question: {question}"
        )

    async def _route(self, question: str) -> str:
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content="You are a helpful assistant that classifies questions for routing."),
                HumanMessage(content=self.routing_prompt.format(question=question))
            ]
            
            response = await self.openai_client.ainvoke(messages)
            classification = response.content.strip().lower()
            if "guideline" in classification:
                return "guidelines"
            return "medline"
        except Exception as e:
            return "medline"

    async def _should_reuse_history(self, question: str, chat_history) -> bool:
        if not chat_history:
            return False
        # Build a short summary of the last N turns
        history_str = "\n".join([
            f"Q: {turn['question']}\nA: {turn['answer']}" for turn in chat_history[-3:]
        ])
        prompt = self.reuse_decision_prompt.format(history=history_str, question=question)
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content="You are a helpful assistant that decides if a new search is needed."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.openai_client.ainvoke(messages)
            decision = response.content.strip().lower()
            print(f"[DEBUG] GPT decision for reuse: {decision}")
            return "reuse" in decision
        except Exception as e:
            print(f"[DEBUG] Error in _should_reuse_history: {e}")
            return False

    async def answer_question(self, question: str, top_k: int = 10, chat_history=None) -> dict:
        agent_type = await self._route(question)
        # Only try to reuse for guidelines agent (can be extended to medline if desired)
        if agent_type == "guidelines" and chat_history:
            should_reuse = await self._should_reuse_history(question, chat_history)
            if should_reuse:
                print("[DEBUG] Reusing previous chunks/context for answer generation.")
                last_turn = chat_history[-1]
                context = self.guidelines_agent.format_context(last_turn["chunks"])
                return {
                    "question": question,
                    "search_query": last_turn["search_query"],
                    "answer_generator": self.guidelines_agent.generate_answer(
                        question, context, chat_history
                    ),
                    "chunks": last_turn["chunks"],
                    "sources": last_turn["sources"]
                }
        # Otherwise, do a new search as normal
        if agent_type == "guidelines":
            return await self.guidelines_agent.answer_question(question, top_k=top_k, chat_history=chat_history)
        else:
            return await self.medline_agent.answer_question(question, top_k=top_k, chat_history=chat_history) 