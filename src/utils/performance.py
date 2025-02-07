import time
import psutil
import threading
from typing import Dict, Any, Optional
from functools import wraps
from contextlib import contextmanager

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread = None
        self.stats = {
            'peak_memory_usage': 0,
            'average_cpu_usage': 0,
            'processing_time': 0,
            'cpu_samples': []
        }

    def get_memory_usage(self) -> float:
        """現在のメモリ使用量を返す（MB単位）"""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """現在のCPU使用率を返す（%）"""
        return self.process.cpu_percent()

    def start_monitoring(self):
        """パフォーマンスモニタリングを開始"""
        self.start_time = time.time()
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """パフォーマンスモニタリングを停止し、統計情報を返す"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        self.end_time = time.time()
        
        # 統計情報の計算
        self.stats['processing_time'] = self.end_time - self.start_time
        if self.stats['cpu_samples']:
            self.stats['average_cpu_usage'] = sum(self.stats['cpu_samples']) / len(self.stats['cpu_samples'])
        
        return self.stats

    def _monitor_resources(self):
        """システムリソースの使用状況をモニタリング"""
        while self._monitoring:
            try:
                # メモリ使用量の監視
                memory_info = self.process.memory_info()
                current_memory = memory_info.rss / 1024 / 1024  # MB単位
                self.stats['peak_memory_usage'] = max(
                    self.stats['peak_memory_usage'],
                    current_memory
                )
                
                # CPU使用率の監視
                cpu_percent = self.process.cpu_percent()
                self.stats['cpu_samples'].append(cpu_percent)
                
                time.sleep(1)  # 1秒間隔でサンプリング
                
            except Exception as e:
                print(f"リソースモニタリング中にエラーが発生: {str(e)}")
                break

@contextmanager
def performance_tracker(description: str = ""):
    """パフォーマンス計測のコンテキストマネージャ"""
    monitor = PerformanceMonitor()
    try:
        monitor.start_monitoring()
        yield monitor
    finally:
        stats = monitor.stop_monitoring()
        print(f"パフォーマンス統計 ({description}):")
        print(f"- 処理時間: {stats['processing_time']:.2f}秒")
        print(f"- 最大メモリ使用量: {stats['peak_memory_usage']:.2f}MB")
        print(f"- 平均CPU使用率: {stats['average_cpu_usage']:.2f}%")

def measure_performance(func):
    """パフォーマンス計測のデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with performance_tracker(func.__name__):
            return func(*args, **kwargs)
    return wrapper

class ResourceCheck:
    @staticmethod
    def check_available_memory(required_mb: float) -> bool:
        """必要なメモリが利用可能かチェック"""
        available = psutil.virtual_memory().available / 1024 / 1024  # MB単位
        return available >= required_mb

    @staticmethod
    def check_available_disk_space(required_mb: float, path: str = '.') -> bool:
        """必要なディスク容量が利用可能かチェック"""
        available = psutil.disk_usage(path).free / 1024 / 1024  # MB単位
        return available >= required_mb

    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """システムリソースの情報を取得"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024,  # MB単位
            'memory_available': psutil.virtual_memory().available / 1024 / 1024,  # MB単位
            'disk_total': psutil.disk_usage('.').total / 1024 / 1024,  # MB単位
            'disk_available': psutil.disk_usage('.').free / 1024 / 1024  # MB単位
        } 