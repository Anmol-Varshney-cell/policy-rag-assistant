#!/usr/bin/env python3
"""RAG Assistant CLI - Company Policy Q&A System"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.rag_pipeline import RAGPipeline
from rag_system.evaluate import RAGEvaluator
from rag_system.prompts import PromptTemplates


async def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║    RAG Assistant - Company Policy Q&A System              ║
    ║    TechShop Inc. Support Assistant                        ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize RAG pipeline
    data_dir = Path(__file__).parent / "data" / "policies"
    index_dir = Path(__file__).parent / "vector_index"
    
    print("Initializing RAG system...\n")
    rag = RAGPipeline(str(data_dir), str(index_dir))
    
    # Build or load index
    try:
        await rag.build_index(force_rebuild=False)
    except Exception as e:
        print(f"Error building index: {e}")
        return
    
    # Main menu
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Ask a question (interactive mode)")
        print("2. Run evaluation suite")
        print("3. Compare prompt versions")
        print("4. View prompt improvement explanation")
        print("5. View vector store statistics")
        print("6. Rebuild vector index")
        print("0. Exit")
        print("="*60)
        
        choice = input("\nSelect an option (0-6): ").strip()
        
        if choice == '1':
            await interactive_mode(rag)
        
        elif choice == '2':
            await run_evaluation(rag)
        
        elif choice == '3':
            await compare_prompts(rag)
        
        elif choice == '4':
            print("\n" + PromptTemplates.get_improvement_explanation())
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            stats = rag.vector_store.get_stats()
            print("\nVector Store Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            print("\nRebuilding vector index...")
            await rag.build_index(force_rebuild=True)
            print("Index rebuilt successfully!")
            input("\nPress Enter to continue...")
        
        elif choice == '0':
            print("\nThank you for using RAG Assistant! Goodbye.")
            break
        
        else:
            print("\nInvalid option. Please try again.")


async def interactive_mode(rag: RAGPipeline):
    """Interactive Q&A mode."""
    print("\n" + "="*60)
    print("INTERACTIVE Q&A MODE")
    print("="*60)
    print("Ask questions about TechShop policies.")
    print("Type 'back' to return to main menu.\n")
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['back', 'exit', 'quit']:
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        print("\nProcessing...")
        
        try:
            result = await rag.answer_question(question, use_improved=True)
            
            print("\n" + "-"*60)
            print("ANSWER:")
            print("-"*60)
            print(result['answer'])
            print("\n" + "-"*60)
            print("SOURCES:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['policy']} (chunk {source['chunk_id']})")
            print("-"*60)
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or contact support.")


async def run_evaluation(rag: RAGPipeline):
    """Run evaluation suite."""
    print("\n" + "="*60)
    print("RUNNING EVALUATION SUITE")
    print("="*60)
    
    # Run evaluation with improved prompt
    results = await RAGEvaluator.run_evaluation(rag, prompt_version='improved')
    
    # Print summary
    RAGEvaluator.print_summary(results)
    
    # Save results
    output_path = Path(__file__).parent / "evaluation_results.json"
    RAGEvaluator.save_results(results, str(output_path))
    
    input("\nPress Enter to continue...")


async def compare_prompts(rag: RAGPipeline):
    """Compare initial vs improved prompt on sample questions."""
    print("\n" + "="*60)
    print("PROMPT COMPARISON")
    print("="*60)
    
    test_questions = [
        "What is the refund policy for digital products?",
        "Do you offer student discounts?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("\n" + "-"*60)
        
        # Initial prompt
        print("INITIAL PROMPT (V1):")
        result_v1 = await rag.answer_question(question, use_improved=False)
        print(result_v1['answer'])
        
        print("\n" + "-"*60)
        
        # Improved prompt
        print("IMPROVED PROMPT (V2):")
        result_v2 = await rag.answer_question(question, use_improved=True)
        print(result_v2['answer'])
        
        print("\n" + "="*60)
    
    input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
