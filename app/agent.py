"""
QA Agent for medical guidelines using Azure Cognitive Search and OpenAI.
"""

import os
import openai
import httpx
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GuidelinesQAAgent:
    """Agent for answering medical questions using guidelines."""
    
    def __init__(self):
        # Azure Search configuration
        self.service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
        self.admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
        self.index_name = os.getenv('AZURE_SEARCH_INDEX_NAME_OPTIMIZED', 'guidelines-chunks-index-optimized-v2')
        
        if not self.service_name or not self.admin_key:
            raise ValueError("Missing Azure Search configuration!")
        
        self.endpoint = f"https://{self.service_name}.search.windows.net"
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.admin_key
        }
        
        # Initialize OpenAI client
        self.openai_client = self._initialize_openai_client()
        
        # System prompt for query generation
        self.query_system_prompt = """You are an expert at converting medical questions into search queries.
Your task is to extract the key medical concepts and terms from the question to create an effective search query.
Focus on medical terminology, conditions, treatments, and specific guideline aspects mentioned in the question."""
        
        # System prompt for answer generation
        self.answer_system_prompt = """You are a medical expert assistant helping healthcare professionals understand medical guidelines.
Your task is to provide accurate, well-structured answers based on the provided guideline excerpts.
Always:
1. Base your answers strictly on the provided guideline content
2. Cite the specific guideline and year when providing information
3. Maintain medical accuracy and precision
4. Acknowledge if information is not available in the provided excerpts
5. Structure complex answers with clear sections
6. Include relevant recommendations or evidence levels if present"""

    def _initialize_openai_client(self):
        """Initialize OpenAI client based on configuration."""
        # Check for Azure OpenAI configuration
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2023-12-01-preview')
        
        # Check for OpenAI configuration
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if azure_endpoint and azure_api_key:
            return openai.AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version
            )
        elif openai_api_key:
            return openai.OpenAI(api_key=openai_api_key)
        else:
            raise ValueError("Missing OpenAI configuration!")

    async def generate_search_query(self, question: str) -> Tuple[str, List[float]]:
        """Generate an effective search query and embedding from the question."""
        try:
            # Generate a better search query using GPT
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.query_system_prompt},
                    {"role": "user", "content": f"Convert this question into an effective search query: {question}"}
                ],
                temperature=0.0
            )
            search_query = response.choices[0].message.content.strip()
            
            # Generate embedding for vector search
            embedding_response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=search_query
            )
            vector = embedding_response.data[0].embedding
            
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

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context for the answer generation."""
        context_parts = []
        
        for result in search_results:
            # Extract key information
            title = result.get('title', '')
            year = result.get('year', '')
            header = result.get('header', '')
            enriched_section_text = result.get('enriched_section_text', '')
            
            # Format the context entry
            context_entry = f"""
SOURCE: {title} ({year})
HEADER: {header}
CONTENT: {enriched_section_text}
---"""
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)

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
            messages = [
                {"role": "system", "content": self.answer_system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                stream=True
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content

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
        
        # Generate answer
        context = self.format_context(search_results)
        
        return {
            "question": question,
            "search_query": search_query,
            "answer_generator": self.generate_answer(question, context, chat_history),
            "chunks": chunks,
            "sources": sources
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
        self.query_system_prompt = """You are an expert at converting health questions into search queries. Extract the key medical concepts and terms from the question to create an effective search query for MedlinePlus health topics."""
        self.answer_system_prompt = """You are a medical expert assistant helping users understand general health topics. Provide accurate, clear answers based on the provided MedlinePlus content. If information is not available, acknowledge it."""

    def _initialize_openai_client(self):
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2023-12-01-preview')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if azure_endpoint and azure_api_key:
            return openai.AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version
            )
        elif openai_api_key:
            return openai.OpenAI(api_key=openai_api_key)
        else:
            raise ValueError("Missing OpenAI configuration!")

    async def generate_search_query(self, question: str) -> Tuple[str, List[float]]:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.query_system_prompt},
                    {"role": "user", "content": f"Convert this question into an effective search query: {question}"}
                ],
                temperature=0.0
            )
            search_query = response.choices[0].message.content.strip()
            embedding_response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=search_query
            )
            vector = embedding_response.data[0].embedding
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
            messages = [
                {"role": "system", "content": self.answer_system_prompt},
                {"role": "user", "content": user_content}
            ]
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
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
        context = self.format_context(search_results)
        return {
            "question": question,
            "search_query": search_query,
            "answer_generator": self.generate_answer(question, context, chat_history),
            "chunks": chunks,
            "sources": sources
        }

class OrchestratorQAAgent:
    """Orchestrates which agent to use based on the user question."""
    def __init__(self):
        self.guidelines_agent = GuidelinesQAAgent()
        self.medline_agent = MedlineQAAgent()
        # Use the same OpenAI client as the guidelines agent for routing
        self.openai_client = self.guidelines_agent.openai_client
        self.routing_prompt = (
            "You are an expert assistant. Classify the following user question as either "
            "'guidelines' if it is about clinical guidelines, recommendations, or evidence-based protocols, "
            "or 'medline' if it is a general health or medical information question.\n"
            "Respond with only 'guidelines' or 'medline'.\n"
            "Question: {question}"
        )

    async def _route(self, question: str) -> str:
        """Use GPT to classify the question as 'guidelines' or 'medline'."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies questions for routing."},
                    {"role": "user", "content": self.routing_prompt.format(question=question)}
                ],
                temperature=0.0
            )
            classification = response.choices[0].message.content.strip().lower()
            if "guideline" in classification:
                return "guidelines"
            return "medline"
        except Exception as e:
            # Fallback to medline if classification fails
            return "medline"

    async def answer_question(self, question: str, top_k: int = 10, chat_history=None) -> Dict[str, Any]:
        agent_type = await self._route(question)
        if agent_type == "guidelines":
            return await self.guidelines_agent.answer_question(question, top_k=top_k, chat_history=chat_history)
        else:
            return await self.medline_agent.answer_question(question, top_k=top_k, chat_history=chat_history) 