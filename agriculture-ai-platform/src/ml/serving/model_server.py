import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
import queue
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictionRequest:
    """Prediction request"""
    request_id: str
    input_data: torch.Tensor
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PredictionResponse:
    """Prediction response"""
    request_id: str
    predictions: torch.Tensor
    probabilities: torch.Tensor
    latency_ms: float
    model_version: str
    timestamp: datetime


class ModelServer:
    """Production model server with batching and caching"""
    
    def __init__(
        self,
        model: nn.Module,
        model_version: str = "1.0.0",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_batch_size: int = 32,
        max_queue_size: int = 1000,
        num_workers: int = 4,
        cache_size: int = 1000,
        enable_profiling: bool = False
    ):
        self.model = model.to(device)
        self.model.eval()
        self.model_version = model_version
        self.device = device
        self.max_batch_size = max_batch_size
        self.enable_profiling = enable_profiling
        
        # Request queue
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        
        # Response futures
        self.futures: Dict[str, asyncio.Future] = {}
        
        # Cache
        self.cache_size = cache_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Statistics
        self.total_predictions = 0
        self.total_latency = 0.0
        
        # Worker thread
        self.running = False
        self.worker_thread = None
        
    def start(self):
        """Start the model server"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
        logger.info(f"Model server started with version {self.model_version}")
        
    def stop(self):
        """Stop the model server"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Model server stopped")
        
    def _batch_worker(self):
        """Background worker for batch processing"""
        while self.running:
            batch = []
            futures = []
            
            # Collect requests for batching
            try:
                while len(batch) < self.max_batch_size:
                    try:
                        request, future = self.request_queue.get(timeout=0.01)
                        batch.append(request)
                        futures.append(future)
                    except queue.Empty:
                        break
                        
                if batch:
                    # Process batch
                    responses = self._process_batch(batch)
                    
                    # Resolve futures
                    for future, response in zip(futures, responses):
                        future.set_result(response)
                        
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                        
    def _process_batch(
        self,
        requests: List[PredictionRequest]
    ) -> List[PredictionResponse]:
        """Process a batch of requests"""
        # Stack inputs
        batch_input = torch.stack([r.input_data for r in requests]).to(self.device)
        
        # Check cache
        batch_key = batch_input.data_ptr()
        if batch_key in self.cache:
            self.cache_hits += 1
            cached_output = self.cache[batch_key]
            return self._create_responses(requests, cached_output)
        
        self.cache_misses += 1
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(batch_input)
        latency = (time.time() - start_time) * 1000
        
        # Get probabilities
        probabilities = torch.softmax(outputs, dim=1)
        
        # Cache result
        if len(self.cache) < self.cache_size:
            self.cache[batch_key] = outputs
            
        # Update statistics
        self.total_predictions += len(requests)
        self.total_latency += latency
        
        return self._create_responses(requests, outputs, latency)
        
    def _create_responses(
        self,
        requests: List[PredictionRequest],
        outputs: torch.Tensor,
        latency: float = 0.0
    ) -> List[PredictionResponse]:
        """Create response objects"""
        responses = []
        for i, request in enumerate(requests):
            response = PredictionResponse(
                request_id=request.request_id,
                predictions=outputs[i:i+1],
                probabilities=torch.softmax(outputs[i:i+1], dim=1),
                latency_ms=latency / len(requests),
                model_version=self.model_version,
                timestamp=datetime.now()
            )
            responses.append(response)
        return responses
        
    async def predict(
        self,
        input_data: torch.Tensor,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PredictionResponse:
        """Async prediction"""
        if request_id is None:
            request_id = str(time.time())
            
        request = PredictionRequest(
            request_id=request_id,
            input_data=input_data,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        future = asyncio.Future()
        self.request_queue.put((request, future))
        
        return await future
        
    def predict_sync(
        self,
        input_data: torch.Tensor
    ) -> PredictionResponse:
        """Synchronous prediction"""
        # Add batch dimension if needed
        if len(input_data.shape) == 3:
            input_data = input_data.unsqueeze(0)
            
        input_data = input_data.to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_data)
        latency = (time.time() - start_time) * 1000
        
        probabilities = torch.softmax(outputs, dim=1)
        
        self.total_predictions += 1
        self.total_latency += latency
        
        return PredictionResponse(
            request_id=str(time.time()),
            predictions=outputs,
            probabilities=probabilities,
            latency_ms=latency,
            model_version=self.model_version,
            timestamp=datetime.now()
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        avg_latency = (
            self.total_latency / self.total_predictions 
            if self.total_predictions > 0 
            else 0
        )
        
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            self.cache_hits / cache_total 
            if cache_total > 0 
            else 0
        )
        
        return {
            "model_version": self.model_version,
            "total_predictions": self.total_predictions,
            "average_latency_ms": avg_latency,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "queue_size": self.request_queue.qsize()
        }


class DynamicBatchServer(ModelServer):
    """Dynamic batching server"""
    
    def __init__(self, *args, timeout_ms: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout_ms = timeout_ms
        
    def _batch_worker(self):
        """Dynamic batching with timeout"""
        while self.running:
            batch = []
            futures = []
            start_time = time.time()
            
            # Wait for first request
            try:
                request, future = self.request_queue.get(timeout=0.1)
                batch.append(request)
                futures.append(future)
            except queue.Empty:
                continue
                
            # Collect more requests within timeout
            while (
                len(batch) < self.max_batch_size and
                (time.time() - start_time) * 1000 < self.timeout_ms
            ):
                try:
                    request, future = self.request_queue.get_nowait()
                    batch.append(request)
                    futures.append(future)
                except queue.Empty:
                    break
                    
            # Process batch
            responses = self._process_batch(batch)
            for future, response in zip(futures, responses):
                future.set_result(response)


class StreamingServer:
    """Streaming inference server"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    def predict_stream(
        self,
        input_iterator,
        batch_size: int = 32
    ):
        """Stream predictions"""
        for batch in self._batch_iterator(input_iterator, batch_size):
            batch_tensor = torch.stack(batch).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                
            probabilities = torch.softmax(outputs, dim=1)
            
            yield probabilities
            
    def _batch_iterator(self, iterator, batch_size):
        """Create batches from iterator"""
            batch = []
            for item in iterator:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
