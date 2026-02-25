"""
Main orchestration engine for Autonomous Predictive Analytics System.
Coordinates data collection, feature engineering, prediction, and decision making.
Architecture Choice: Event-driven orchestration with graceful shutdown handling
to ensure trading decisions aren't interrupted during system updates.
"""

import asyncio
import signal
import logging
from datetime import datetime
from typing import Dict, Optional

from data_collector import MarketDataCollector
from feature_engineer import FeatureEngineer
from model_predictor import ModelPredictor
from decision_maker import TradingDecisionMaker
from firebase_client import FirebaseClient
from utils.logger import setup_logger

class PredictiveTradingEngine:
    """Core orchestration engine for real-time trading decisions."""
    
    def __init__(self, config_path: str = "./config.yaml"):
        self.logger = setup_logger(__name__)
        self.shutdown_flag = False
        
        # Initialize components with dependency injection
        self.firebase = FirebaseClient()
        self.data_collector = MarketDataCollector(self.firebase)
        self.feature_engineer = FeatureEngineer()
        self.model_predictor = ModelPredictor()
        self.decision_maker = TradingDecisionMaker(self.firebase)
        
        self.iteration_count = 0
        self.last_prediction_time = None
        
        # Register graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received shutdown signal {signum}")
        self.shutdown_flag = True
        
    async def run_iteration(self):
        """Execute one complete prediction-decision cycle."""
        try:
            self.logger.info(f"Starting iteration {self.iteration_count}")
            
            # Phase 1: Data Collection
            raw_data = await self.data_collector.collect_realtime_data()
            if raw_data is None or raw_data.empty:
                self.logger.warning("No data collected, skipping iteration")
                return
                
            # Phase 2: Feature Engineering
            features = self.feature_engineer.transform(raw_data)
            if features is None or features.empty:
                self.logger.error("Feature engineering failed")
                return
                
            # Phase 3: Model Prediction
            predictions = self.model_predictor.predict(features)
            if predictions is None:
                self.logger.error("Model prediction failed")
                return
                
            # Phase 4: Decision Making
            decisions = await self.decision_maker.make_decisions(
                predictions, raw_data
            )
            
            # Phase 5: Log results to Firebase
            await self._log_iteration_results(predictions, decisions)
            
            self.last_prediction_time = datetime.utcnow()
            self.iteration_count += 1
            
            self.logger.info(f"Iteration {self.iteration_count} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Iteration failed: {e}", exc_info=True)
            # Critical: Don't crash on single iteration failure
            # Log to Firebase for monitoring
            await self.firebase.log_error({
                "error": str(e),
                "iteration": self.iteration_count,
                "timestamp": datetime.utcnow().isoformat()
            })
            
    async def _log_iteration_results(self, predictions, decisions):
        """Log iteration results to Firebase for monitoring and auditing."""
        log_entry = {
            "iteration": self.iteration_count,
            "timestamp": datetime.utcnow().isoformat(),
            "predictions": predictions.to_dict() if hasattr(predictions, 'to_dict') else str(predictions),
            "decisions": decisions,
            "system_metrics": {
                "memory_usage": self._get_memory_usage(),
                "cpu_percent": self._get_cpu_usage()
            }
        }
        await self.firebase.write_document("trading_logs", log_entry)
        
    def _get_memory_usage(self):
        """Get current memory usage (simplified)."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return None
            
    def _get_cpu_usage(self):
        """Get current CPU usage (simplified)."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return None
            
    async def run(self):
        """Main run loop with controlled iteration rate."""
        self.logger.info("Starting Predictive Trading Engine")
        
        while not self.shutdown_flag:
            try:
                await self.run_iteration()
                
                # Dynamic sleep based on market conditions
                sleep_time = self._calculate_sleep_time()
                await asyncio.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.shutdown_flag = True
            except Exception as e:
                self.logger.critical(f"Engine crashed: {e}", exc_info=True)
                break
                
        self.logger.info("Shutting down Predictive Trading Engine")
        await self._cleanup()
        
    def _calculate_sleep_time(self) -> float:
        """Calculate optimal sleep time between iterations."""
        # Base 30 seconds, adjust based on market volatility
        base_sleep = 30.0
        
        # TODO: Implement volatility-based adjustment
        # For now, return base sleep time
        return base_sleep
        
    async def _cleanup(self):
        """Cleanup resources before shutdown."""
        self.logger.info("Cleaning up resources...")
        await self.firebase.close()
        self.logger.info("Cleanup complete")