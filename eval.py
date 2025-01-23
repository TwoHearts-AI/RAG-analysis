from typing import Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field 
from loguru import logger
from langsmith import traceable
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
import httpx
import json
import os
from datetime import datetime
from statistics import mean

from config import CONFIG

load_dotenv()

# Existing model definitions remain the same
class EvaluationGrade(BaseModel):
    consultation_score: int = Field(..., description="Score for addressing psychological consultation")
    context_usage_score: int = Field(..., description="Score for effective context usage")
    conflict_analysis_score: int = Field(..., description="Score for conflict analysis quality")
    explanation: str = Field(..., description="Detailed explanation of the scores")

class EvaluationMetrics(BaseModel):
    explanation: str
    relevance_score: float
    context_quality: float
    conflict_assessment_score: float

class RAGRequest(BaseModel):
    collection_name: str = Field(default="default_collection")
    limit: int = Field(default=3, ge=1, le=10)

class RAGResponse(BaseModel):
    answer: str
    context: str

class AggregatedResults(BaseModel):
    inference_model: str
    judge_model: str
    average_relevance: float
    average_context_quality: float
    average_conflict_assessment: float
    overall_average: float
    collections_analyzed: List[str]
    timestamp: str

llm_query_prompt = "Оцени конфликтность партнеров по шкале 0/10-10/10. Конфликтуют ли партнеры или у них все хорошо? Выдели основные причины конфликтов. Перечисли их в нумерованном списке и к каждой припиши цитирование из-за чего ты сделал такой вывод. Предложи варианты решения конфликтов. Как лучше изменить свое поведение в будущих отношениях, к чему быть более внимательным в разрезе конфликтов?"
evaluator_model = "o1-mini"

@traceable
class RelationshipResponseEvaluator:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=EvaluationGrade)
        self.eval_model = ChatOpenAI(model=evaluator_model, temperature=1)
        
    async def evaluate_response(self, rag_response: RAGResponse) -> EvaluationMetrics:
        evaluation_template = """Вы оцениваете ответ системы психологического консультирования.

        Оцените следующие критерии по шкале от 0 до 10:
        1. Насколько хорошо ответ соответствует запросу на психологическую консультацию?
        2. Насколько эффективно в ответе используется предоставленный контекст переписки?
        3. Насколько качественно проанализированы конфликты в отношениях?

        Полный промпт системы: {question}
        Ответ системы: {answer}

        {format_instructions}
        """
        
        try:
            prompt = ChatPromptTemplate.from_template(template=evaluation_template)
            chain = prompt | self.eval_model | self.parser
            
            full_prompt = f"Контекст из релевантной запросу переписки:\n{rag_response.context}\n\nЗапрос на психологическую консультацию: {llm_query_prompt}"
            
            result = chain.invoke({
                "question": full_prompt,
                "answer": rag_response.answer,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return EvaluationMetrics(
                explanation=result.explanation,
                relevance_score=result.consultation_score / 10.0,
                context_quality=result.context_usage_score / 10.0,
                conflict_assessment_score=result.conflict_analysis_score / 10.0
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

async def get_collections() -> List[str]:
    """Fetch all collection names from the API"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("http://localhost:8000/collections")
            data = response.json()
            collections = [collection["name"] for collection in data["collections"]]
            logger.info(f"Found collections: {collections}")
            return collections
    except Exception as e:
        logger.error(f"Failed to fetch collections: {str(e)}")
        raise

@traceable
async def run_evaluation_pipeline(collection_name: str) -> Dict:
    """Run evaluation for a single collection"""
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            rag_response = await client.post(
                "http://localhost:8000/rag-inference",
                json={"collection_name": collection_name, "limit": 5}
            )
            
            logger.info(f"RAG response for {collection_name}: {rag_response.json()}")
            
            rag_result = RAGResponse(**rag_response.json())
        
        evaluator = RelationshipResponseEvaluator()
        metrics = await evaluator.evaluate_response(rag_result)
        
        return {
            "metrics": metrics,
            "rag_response": rag_result
        }
    except Exception as e:
        logger.error(f"Evaluation pipeline failed for {collection_name}: {str(e)}")
        raise

async def save_eval_results(results: Dict, collection_name: str) -> str:
    """Save evaluation results for a single collection"""
    os.makedirs("eval_results", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_results/eval_{collection_name}_{timestamp}.json"
    
    json_results = {
        "collection_name": collection_name,
        "timestamp": timestamp,
        "metrics": {
            "relevance_score": results["metrics"].relevance_score,
            "context_quality": results["metrics"].context_quality,
            "conflict_assessment_score": results["metrics"].conflict_assessment_score,
            "explanation": results["metrics"].explanation
        },
        "rag_response": {
            "answer": results["rag_response"].answer,
            "context": results["rag_response"].context
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation results saved to {filename}")
    return filename

async def save_aggregated_results(all_results: List[Dict], collections: List[str]) -> str:
    """Save aggregated results across all collections"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_results/aggregated_results_{timestamp}.json"
    
    # Calculate averages
    relevance_scores = [r["metrics"].relevance_score for r in all_results]
    context_qualities = [r["metrics"].context_quality for r in all_results]
    conflict_scores = [r["metrics"].conflict_assessment_score for r in all_results]
    
    aggregated = AggregatedResults(
        inference_model=CONFIG.MISTRAL_MODEL,  # Update with actual model name
        judge_model=evaluator_model,      # Update with actual judge model
        average_relevance=mean(relevance_scores),
        average_context_quality=mean(context_qualities),
        average_conflict_assessment=mean(conflict_scores),
        overall_average=mean(relevance_scores + context_qualities + conflict_scores),
        collections_analyzed=collections,
        timestamp=timestamp
    )
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(aggregated.dict(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Aggregated results saved to {filename}")
    return filename

async def main():
    try:
        # Get all collections
        collections = await get_collections()
        all_results = []
        
        # Process each collection
        for collection_name in collections:
            logger.info(f"Processing collection: {collection_name}")
            results = await run_evaluation_pipeline(collection_name)
            all_results.append(results)
            
            # Save individual results
            saved_file = await save_eval_results(results, collection_name)
            
            # Print summary for this collection
            print(f"\nResults for collection {collection_name}:")
            print(f"Relevance Score: {results['metrics'].relevance_score}")
            print(f"Context Quality: {results['metrics'].context_quality}")
            print(f"Conflict Assessment: {results['metrics'].conflict_assessment_score}")
            print(f"Results saved to: {saved_file}")
        
        # Save aggregated results
        aggregated_file = await save_aggregated_results(all_results, collections)
        print(f"\nAggregated results saved to: {aggregated_file}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())