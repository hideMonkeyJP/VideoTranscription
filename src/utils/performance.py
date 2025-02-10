"""
パフォーマンス監視用のユーティリティ
"""

import time
import psutil
from functools import wraps
from typing import Dict, Any, Callable
import logging
from datetime import datetime

class ResourceCheck:
    """システムリソースの監視クラス"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """
        現在のメモリ使用量を取得
        
        Returns:
            float: メモリ使用量(MB)
        """
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # バイトからMBに変換
    
    @staticmethod
    def get_cpu_usage() -> float:
        """
        現在のCPU使用率を取得
        
        Returns:
            float: CPU使用率(%)
        """
        return psutil.cpu_percent()

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        self.resource_check = ResourceCheck()
    
    def start_operation(self, operation_name: str) -> None:
        """
        操作の計測を開始
        
        Args:
            operation_name: 操作の名前
        """
        self.metrics[operation_name] = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "status": "running",
            "start_memory": self.resource_check.get_memory_usage(),
            "start_cpu": self.resource_check.get_cpu_usage()
        }
    
    def end_operation(self, operation_name: str, status: str = "success") -> float:
        """
        操作の計測を終了
        
        Args:
            operation_name: 操作の名前
            status: 操作の状態 ("success" or "error")
        
        Returns:
            float: 操作の所要時間(秒)
        """
        if operation_name not in self.metrics:
            raise KeyError(f"Operation '{operation_name}' has not been started")
        
        end_time = time.time()
        self.metrics[operation_name]["end_time"] = end_time
        self.metrics[operation_name]["status"] = status
        duration = end_time - self.metrics[operation_name]["start_time"]
        self.metrics[operation_name]["duration"] = duration
        
        # リソース使用状況の記録
        self.metrics[operation_name]["end_memory"] = self.resource_check.get_memory_usage()
        self.metrics[operation_name]["end_cpu"] = self.resource_check.get_cpu_usage()
        self.metrics[operation_name]["memory_diff"] = (
            self.metrics[operation_name]["end_memory"] - 
            self.metrics[operation_name]["start_memory"]
        )
        
        # ログ出力
        self.logger.info(
            f"Operation '{operation_name}' completed with status '{status}' in {duration:.2f} seconds"
        )
        
        return duration
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        全ての計測結果を取得
        
        Returns:
            Dict: 計測結果
        """
        return self.metrics
    
    def get_operation_duration(self, operation_name: str) -> float:
        """
        特定の操作の所要時間を取得
        
        Args:
            operation_name: 操作の名前
        
        Returns:
            float: 操作の所要時間(秒)
        """
        if operation_name not in self.metrics:
            raise KeyError(f"Operation '{operation_name}' not found")
        
        return self.metrics[operation_name]["duration"]
    
    def get_memory_usage(self) -> float:
        """
        現在のメモリ使用量を取得
        
        Returns:
            float: メモリ使用量(MB)
        """
        return self.resource_check.get_memory_usage()
    
    def get_cpu_usage(self) -> float:
        """
        現在のCPU使用率を取得
        
        Returns:
            float: CPU使用率(%)
        """
        return self.resource_check.get_cpu_usage()
    
    def generate_report(self) -> str:
        """
        パフォーマンスレポートを生成
        
        Returns:
            str: レポート文字列
        """
        report = ["Performance Report", "=" * 50]
        report.append(f"Generated at: {datetime.now().isoformat()}\n")
        
        total_time = 0
        for operation, data in self.metrics.items():
            duration = data["duration"]
            if duration is not None:
                total_time += duration
                status = data["status"]
                memory_diff = data.get("memory_diff", 0)
                
                report.append(f"Operation: {operation}")
                report.append(f"Status: {status}")
                report.append(f"Duration: {duration:.2f} seconds")
                report.append(f"Memory Change: {memory_diff:.2f} MB\n")
        
        report.append(f"Total Time: {total_time:.2f} seconds")
        report.append(f"Current Memory Usage: {self.get_memory_usage():.2f} MB")
        report.append(f"Current CPU Usage: {self.get_cpu_usage():.2f}%")
        
        return "\n".join(report)

def measure_performance(func: Callable = None, *, operation_name: str = None):
    """
    パフォーマンス計測用のデコレータ
    
    Args:
        func: デコレート対象の関数
        operation_name: 操作の名前(指定がない場合は関数名を使用)
    """
    if func is None:
        return lambda f: measure_performance(f, operation_name=operation_name)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        op_name = operation_name or func.__name__
        monitor.start_operation(op_name)
        try:
            result = func(*args, **kwargs)
            monitor.end_operation(op_name, "success")
            return result
        except Exception as e:
            monitor.end_operation(op_name, "error")
            raise e
    return wrapper