import os
import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class PerformanceMonitor:
    """システムリソースとパフォーマンスを監視するクラス"""
    
    def __init__(self):
        """初期化"""
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, task_name: str):
        """モニタリングを開始
        
        Args:
            task_name (str): タスク名
        """
        self.start_time = time.time()
        self.metrics[task_name] = {
            'start_time': datetime.now().isoformat(),
            'memory_start': self.process.memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent_start': self.process.cpu_percent()
        }
        self.logger.debug(f"{task_name}のモニタリングを開始")
    
    def stop_monitoring(self, task_name: str) -> Dict[str, Any]:
        """モニタリングを停止
        
        Args:
            task_name (str): タスク名
            
        Returns:
            Dict[str, Any]: パフォーマンス指標
        """
        if task_name not in self.metrics:
            self.logger.warning(f"{task_name}のモニタリングが開始されていません")
            return {}
        
        self.end_time = time.time()
        metrics = self.metrics[task_name]
        
        # 終了時の指標を記録
        metrics.update({
            'end_time': datetime.now().isoformat(),
            'memory_end': self.process.memory_info().rss / 1024 / 1024,
            'cpu_percent_end': self.process.cpu_percent(),
            'duration': self.end_time - time.time(),
            'memory_diff': (self.process.memory_info().rss / 1024 / 1024) - metrics['memory_start']
        })
        
        self.logger.debug(f"{task_name}のモニタリングを停止: {metrics}")
        return metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """現在のシステムリソース使用状況を取得
        
        Returns:
            Dict[str, Any]: 現在のリソース使用状況
        """
        return {
            'memory_usage': self.process.memory_info().rss / 1024 / 1024,
            'cpu_percent': self.process.cpu_percent(),
            'threads': self.process.num_threads(),
            'open_files': len(self.process.open_files()),
            'system_memory': dict(psutil.virtual_memory()._asdict())
        }
    
    def get_task_summary(self, task_name: str) -> Optional[Dict[str, Any]]:
        """タスクのパフォーマンスサマリーを取得
        
        Args:
            task_name (str): タスク名
            
        Returns:
            Optional[Dict[str, Any]]: パフォーマンスサマリー
        """
        if task_name not in self.metrics:
            return None
            
        metrics = self.metrics[task_name]
        return {
            'duration': metrics.get('duration', 0),
            'memory_usage': metrics.get('memory_diff', 0),
            'cpu_percent_avg': (metrics.get('cpu_percent_end', 0) + 
                              metrics.get('cpu_percent_start', 0)) / 2
        }
    
    def reset(self):
        """モニタリング情報をリセット"""
        self.metrics.clear()
        self.start_time = None
        self.end_time = None
        self.logger.debug("モニタリング情報をリセットしました")

    def get_memory_usage(self) -> float:
        """現在のメモリ使用量を取得（MB単位）
        
        Returns:
            float: 現在のメモリ使用量（MB）
        """
        return self.process.memory_info().rss / 1024 / 1024 