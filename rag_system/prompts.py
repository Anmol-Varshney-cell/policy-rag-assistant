"""Prompt templates for RAG system with iteration improvements."""


class PromptTemplates:
    """Collection of prompt templates for the RAG assistant."""
    
    # VERSION 1: Initial Prompt
    # Basic approach with minimal structure
    INITIAL_PROMPT = """You are a customer support assistant for TechShop Inc. Answer the user's question based on the provided policy documents.

Context from policy documents:
{context}

User Question: {question}

Answer:"""
    
    # VERSION 2: Improved Prompt
    # Enhanced with better instructions, structure, and hallucination prevention
    IMPROVED_PROMPT = """You are a knowledgeable customer support assistant for TechShop Inc.

Your task is to answer customer questions accurately using ONLY the information provided in the company policy documents below.

**IMPORTANT RULES:**
1. Base your answer ONLY on the context provided below
2. If the context doesn't contain enough information to answer the question, explicitly state: "I don't have enough information in our policies to answer this question fully. Please contact our support team at support@techshop.com"
3. Do NOT make up or infer information not present in the context
4. Always cite which policy the information comes from (e.g., "According to our Refund Policy...")
5. If the question is partially answerable, provide what you can and clearly indicate what information is missing

**CONTEXT FROM COMPANY POLICIES:**
{context}

**CUSTOMER QUESTION:**
{question}

**YOUR ANSWER:**
Provide a clear, structured response following this format:
- Start with a direct answer if available
- Include relevant policy details with citations
- Use bullet points for clarity when listing multiple items
- End with next steps or contact information if appropriate"""
    
    # VERSION 3: Structured Output Prompt (Optional - for JSON responses)
    STRUCTURED_PROMPT = """You are a customer support assistant for TechShop Inc.

Answer the question using ONLY the provided policy context. Your response must be accurate and well-sourced.

**CONTEXT:**
{context}

**QUESTION:**
{question}

**INSTRUCTIONS:**
Provide your answer in the following structured format:

**Answer:** [Direct answer to the question]

**Details:**
- [Key point 1 with policy citation]
- [Key point 2 with policy citation]

**Source Policy:** [Name of the policy document(s) used]

**Confidence:** [High/Medium/Low based on context sufficiency]

**Additional Notes:** [Any caveats, missing information, or recommended next steps]

If you cannot answer from the context, state: "Information not available in provided policies" and suggest contacting support."""
    
    @staticmethod
    def get_improvement_explanation() -> str:
        """Explain the improvements made between prompt versions."""
        return """
**PROMPT ITERATION IMPROVEMENTS**

**Changes from V1 (Initial) to V2 (Improved):**

1. **Explicit Grounding Instructions:**
   - Added "ONLY" emphasis to prevent hallucinations
   - Clear rule: if info isn't in context, say so explicitly
   - Prevents model from using pre-trained knowledge

2. **Structured Response Format:**
   - Defined expected answer structure (direct answer → details → next steps)
   - Encourages organized, scannable responses
   - Better user experience with bullet points

3. **Source Citation Requirement:**
   - Forces model to reference which policy contains the information
   - Increases accountability and verifiability
   - Helps users know where to look for more details

4. **Partial Answer Handling:**
   - Explicit instruction for partially answerable questions
   - Acknowledges what's missing rather than guessing
   - Reduces hallucination risk

5. **Clear Role Definition:**
   - "Knowledgeable" sets professional tone
   - Company name (TechShop Inc.) provides context
   - Establishes appropriate assistant persona

6. **Fallback Contact Information:**
   - Provides alternative when RAG fails
   - Improves overall customer experience
   - Realistic handling of edge cases

**Why These Changes Matter:**
- **Accuracy:** Stronger grounding reduces false information
- **Transparency:** Citation requirements make answers auditable
- **User Trust:** Admitting knowledge gaps is better than guessing
- **Maintainability:** Structured format easier to evaluate and improve

**Trade-offs Considered:**
- Longer prompt = slightly higher token cost, but significantly better output quality
- More rigid structure = less creative freedom, but more consistent results
- Explicit rules = verbose prompt, but crucial for preventing hallucinations in production
"""
    
    @staticmethod
    def format_context(retrieved_chunks: list) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        for i, (chunk_text, metadata, distance) in enumerate(retrieved_chunks, 1):
            policy_name = metadata.get('filename', 'Unknown').replace('_', ' ').replace('.txt', '').title()
            context_parts.append(f"[Source {i}: {policy_name}]\n{chunk_text}\n")
        return "\n".join(context_parts)
