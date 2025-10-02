"""
FastAPI-based REST API for the protein science AI system.

This module provides HTTP endpoints for protein analysis, structure prediction,
function analysis, and multi-agent workflows.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import asyncio
import uvicorn
from loguru import logger

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. REST API features will be limited.")

# Import core components
from ..agents.protein_agent import ProteinAgent
from ..collaboration.coordinator import AgentCoordinator


# Pydantic models for API requests/responses
if FASTAPI_AVAILABLE:
    
    class ProteinAnalysisRequest(BaseModel):
        """Request model for protein analysis."""
        sequence: str = Field(..., description="Protein sequence", min_length=1)
        uniprot_id: Optional[str] = Field(None, description="UniProt identifier")
        analysis_type: str = Field("comprehensive", description="Type of analysis")
        include_structure: bool = Field(True, description="Include structure prediction")
        include_function: bool = Field(True, description="Include function prediction")
        
    class WorkflowRequest(BaseModel):
        """Request model for multi-agent workflows."""
        workflow_type: str = Field(..., description="Type of workflow to execute")
        protein_input: Union[str, Dict[str, Any]] = Field(..., description="Protein data")
        config: Optional[Dict[str, Any]] = Field(None, description="Workflow configuration")
        
    class MutationAnalysisRequest(BaseModel):
        """Request model for mutation analysis."""
        sequence: str = Field(..., description="Wild-type protein sequence")
        mutations: List[str] = Field(..., description="List of mutations to analyze")
        uniprot_id: Optional[str] = Field(None, description="UniProt identifier")
        
    class QuestionRequest(BaseModel):
        """Request model for reasoning about results."""
        analysis_results: Dict[str, Any] = Field(..., description="Analysis results")
        question: str = Field(..., description="Question to answer")


class ProteinScienceAPI:
    """
    Main API class for the protein science AI system.
    
    Provides RESTful endpoints for:
    - Protein sequence analysis
    - Structure prediction  
    - Function prediction
    - Multi-agent workflows
    - Reasoning and question answering
    """
    
    def __init__(
        self,
        title: str = "Protein Science AI API",
        version: str = "0.1.0",
        description: str = "Agentic AI system for protein science research",
        enable_cors: bool = True,
        cors_origins: List[str] = ["*"]
    ):
        """
        Initialize the API.
        
        Args:
            title: API title
            version: API version
            description: API description
            enable_cors: Whether to enable CORS
            cors_origins: Allowed CORS origins
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for the REST API")
            
        self.app = FastAPI(
            title=title,
            version=version,
            description=description,
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Enable CORS if requested
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Initialize core components
        self.protein_agent = ProteinAgent()
        self.coordinator = AgentCoordinator(main_agent=self.protein_agent)
        
        # Track active tasks
        self.active_tasks = {}
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Protein Science API initialized")
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "Protein Science AI API",
                "version": "0.1.0",
                "description": "Agentic AI system for protein science research",
                "endpoints": {
                    "analyze": "/analyze - Protein sequence analysis",
                    "workflow": "/workflow - Multi-agent workflows", 
                    "mutations": "/mutations - Mutation analysis",
                    "reason": "/reason - Reasoning about results",
                    "status": "/status/{task_id} - Task status",
                    "docs": "/docs - API documentation"
                }
            }
            
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "agents": {
                    "protein_agent": "active",
                    "coordinator": "active",
                    "specialized_agents": len(self.coordinator.specialized_agents)
                }
            }
            
        @self.app.post("/analyze")
        async def analyze_protein(
            request: ProteinAnalysisRequest,
            background_tasks: BackgroundTasks
        ):
            """
            Analyze a protein sequence.
            
            Performs comprehensive analysis including sequence analysis,
            structure prediction, and function prediction.
            """
            try:
                # Validate sequence
                if not request.sequence.replace(" ", "").isalpha():
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid protein sequence. Must contain only amino acid letters."
                    )
                
                # Prepare protein input
                protein_input = {
                    "sequence": request.sequence.replace(" ", "").upper(),
                    "uniprot_id": request.uniprot_id
                }
                
                # Run analysis
                logger.info(f"Starting protein analysis for sequence of length {len(protein_input['sequence'])}")
                
                result = await self.protein_agent.analyze_protein(
                    protein_input=protein_input,
                    analysis_type=request.analysis_type,
                    include_structure=request.include_structure,
                    include_function=request.include_function
                )
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "analysis_results": result,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Analysis failed: {str(e)}"
                )
                
        @self.app.post("/workflow")
        async def execute_workflow(
            request: WorkflowRequest,
            background_tasks: BackgroundTasks
        ):
            """
            Execute a multi-agent workflow.
            
            Coordinates multiple specialized agents to perform complex
            protein science tasks.
            """
            try:
                # Validate workflow type
                valid_workflows = [
                    "comprehensive_analysis",
                    "drug_discovery", 
                    "protein_engineering",
                    "mutation_analysis"
                ]
                
                if request.workflow_type not in valid_workflows:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid workflow type. Must be one of: {valid_workflows}"
                    )
                
                logger.info(f"Starting {request.workflow_type} workflow")
                
                # Execute workflow
                result = await self.coordinator.execute_workflow(
                    workflow_type=request.workflow_type,
                    protein_input=request.protein_input,
                    workflow_config=request.config
                )
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "workflow_results": result,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow execution failed: {str(e)}"
                )
                
        @self.app.post("/mutations")
        async def analyze_mutations(
            request: MutationAnalysisRequest,
            background_tasks: BackgroundTasks
        ):
            """
            Analyze the effects of protein mutations.
            
            Compares wild-type protein with specified mutants.
            """
            try:
                # Validate mutations format
                for mutation in request.mutations:
                    if len(mutation) < 3 or not mutation[0].isalpha() or not mutation[-1].isalpha():
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid mutation format: {mutation}. Use format like 'A123G'"
                        )
                
                # Prepare workflow config
                config = {
                    "mutations": request.mutations
                }
                
                protein_input = {
                    "sequence": request.sequence.replace(" ", "").upper(),
                    "uniprot_id": request.uniprot_id
                }
                
                logger.info(f"Starting mutation analysis for {len(request.mutations)} mutations")
                
                # Execute mutation analysis workflow
                result = await self.coordinator.execute_workflow(
                    workflow_type="mutation_analysis",
                    protein_input=protein_input,
                    workflow_config=config
                )
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "mutation_results": result,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"Mutation analysis failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Mutation analysis failed: {str(e)}"
                )
                
        @self.app.post("/reason")
        async def reason_about_results(request: QuestionRequest):
            """
            Answer questions about analysis results using AI reasoning.
            
            Provides intelligent interpretation of protein analysis results.
            """
            try:
                # Use the protein agent's reasoning capability
                answer = self.protein_agent.reason_about_results(
                    analysis_results=request.analysis_results,
                    question=request.question
                )
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "question": request.question,
                        "answer": answer,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"Reasoning failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Reasoning failed: {str(e)}"
                )
                
        @self.app.get("/status/{workflow_id}")
        async def get_workflow_status(workflow_id: str):
            """Get the status of a workflow."""
            try:
                status = self.coordinator.get_workflow_status(workflow_id)
                
                if status is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Workflow {workflow_id} not found"
                    )
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "workflow_status": status,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Status check failed: {str(e)}"
                )
                
        @self.app.get("/workflows")
        async def list_workflows():
            """List all active workflows."""
            try:
                active_workflows = self.coordinator.list_active_workflows()
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "success",
                        "active_workflows": active_workflows,
                        "count": len(active_workflows),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
            except Exception as e:
                logger.error(f"Workflow listing failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow listing failed: {str(e)}"
                )
        
        # Add error handlers
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status": "error",
                    "message": exc.detail,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        @self.app.exception_handler(Exception) 
        async def general_exception_handler(request, exc):
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Internal server error",
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        log_level: str = "info"
    ):
        """
        Run the API server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            reload: Enable auto-reload for development
            log_level: Logging level
        """
        logger.info(f"Starting Protein Science API on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level
        )


def create_api() -> ProteinScienceAPI:
    """Create and configure the API instance."""
    return ProteinScienceAPI()


if __name__ == "__main__":
    # Run the API server
    api = create_api()
    api.run(host="0.0.0.0", port=8000, reload=True)