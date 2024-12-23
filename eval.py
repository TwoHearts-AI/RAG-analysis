from typing import Dict
from pydantic import BaseModel, Field 
from loguru import logger
from langsmith import traceable
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
import httpx

# Evaluation Models
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

# RAG Request/Response Models
class RAGRequest(BaseModel):
    collection_name: str = Field(default="default_collection")
    limit: int = Field(default=3, ge=1, le=10)

class RAGResponse(BaseModel):
    answer: str
    context: str

# Define the prompts here temporarily until we fix imports
llm_query_prompt = "Оцени конфликтность партнеров по шкале 0/10-10/10. Конфликтуют ли партнеры или у них все хорошо? Выдели основные причины конфликтов. Перечисли их в нумерованном списке и к каждой припиши цитирование из-за чего ты сделал такой вывод. Предложи варианты решения конфликтов. Как лучше изменить свое поведение в будущих отношениях, к чему быть более внимательным в разрезе конфликтов?"

@traceable
class RelationshipResponseEvaluator:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=EvaluationGrade)
        self.eval_model = ChatOpenAI(model="gpt-4", temperature=0)
        
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
            
            # Using individual scores for each criterion
            return EvaluationMetrics(
                explanation=result.explanation,
                relevance_score=result.consultation_score / 10.0,
                context_quality=result.context_usage_score / 10.0,
                conflict_assessment_score=result.conflict_analysis_score / 10.0
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

@traceable
async def run_evaluation_pipeline(collection_name: str) -> Dict:
    """Run evaluation using existing RAG endpoint"""
    try:
        # Call existing RAG endpoint
        async with httpx.AsyncClient(timeout=180.0) as client:
            rag_response = await client.post(
                "http://localhost:8000/rag-inference",
                json={"collection_name": collection_name, "limit": 5}
            )
            
            logger.info(rag_response.json())
            
            rag_result = RAGResponse(**rag_response.json())
        
        # Evaluate response
        evaluator = RelationshipResponseEvaluator()
        metrics = await evaluator.evaluate_response(rag_result)
        
        return {
            "metrics": metrics,
            "rag_response": rag_result
        }
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    import asyncio
    
    async def main():
        results = await run_evaluation_pipeline("chat_chunks_baseline_default")
        print("\nEvaluation Results:")
        print(f"Relevance Score: {results['metrics'].relevance_score}")
        print(f"Context Quality: {results['metrics'].context_quality}")
        print(f"Conflict Assessment: {results['metrics'].conflict_assessment_score}")
        print(f"\nExplanation: {results['metrics'].explanation}")
        print(f"\nRAG Response: {results['rag_response'].answer}")
        
    asyncio.run(main())