"""
core/dataset_manager.py
=======================
Dataset management, saving, and searching functionality
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import gzip

from core.data_structures import BaseStrategy, ProbabilityMetrics, VolatilityProfile


@dataclass
class DatasetInfo:
    """Information about a saved dataset"""
    id: str
    name: str
    filepath: str
    created_at: datetime
    
    # What's in this dataset
    strategies_used: List[str] = field(default_factory=list)
    patterns_included: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    location_strategies: List[str] = field(default_factory=list)
    
    # Data properties
    row_count: int = 0
    date_range: Tuple[str, str] = ("", "")
    volatility: Optional[VolatilityProfile] = None
    
    # Performance metrics
    probability: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    acceptance_status: str = "pending"  # 'accepted', 'rejected', 'pending'
    
    # Tags for searching
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Parent datasets (if this is a combination)
    parent_datasets: List[str] = field(default_factory=list)
    combination_method: str = ""  # 'AND', 'OR', etc.
    
    def to_dict(self):
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('volatility'):
            data['volatility'] = VolatilityProfile(**data['volatility'])
        return cls(**data)
    
    def matches_search(self, query: str) -> bool:
        """Check if dataset matches search query"""
        query_lower = query.lower()
        
        # Search in various fields
        searchable = [
            self.name.lower(),
            self.notes.lower(),
            ' '.join(self.tags).lower(),
            ' '.join(self.strategies_used).lower(),
            ' '.join(self.patterns_included).lower(),
            ' '.join(self.location_strategies).lower(),
            self.acceptance_status.lower()
        ]
        
        return any(query_lower in field for field in searchable)


class DatasetManager:
    """Manages dataset storage and retrieval"""
    
    def __init__(self, base_directory: str = "datasets"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.base_directory / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.base_directory / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Cache for quick access
        self.dataset_index = {}
        self._load_index()
        
    def save_dataset(self, 
                    name: str,
                    data: pd.DataFrame,
                    strategies: List[BaseStrategy],
                    probability: Optional[ProbabilityMetrics] = None,
                    parent_datasets: List[str] = None,
                    combination_method: str = "",
                    notes: str = "",
                    tags: List[str] = None) -> DatasetInfo:
        """Save a dataset with full metadata"""
        
        # Generate unique ID
        dataset_id = self._generate_id(name)
        
        # Extract metadata from strategies
        patterns = []
        timeframes = []
        location_strategies = []
        strategy_names = []
        
        for strategy in strategies:
            strategy_names.append(strategy.name)
            
            if hasattr(strategy, 'actions'):
                for action in strategy.actions:
                    if action.pattern:
                        patterns.append(action.pattern.name)
                    if action.time_range:
                        timeframes.append(str(action.time_range))
                    if action.location_strategy:
                        location_strategies.append(action.location_strategy)
        
        # Remove duplicates
        patterns = list(set(patterns))
        timeframes = list(set(timeframes))
        location_strategies = list(set(location_strategies))
        
        # Create dataset info
        info = DatasetInfo(
            id=dataset_id,
            name=name,
            filepath=str(self.data_dir / f"{dataset_id}.pkl.gz"),
            created_at=datetime.now(),
            strategies_used=strategy_names,
            patterns_included=patterns,
            timeframes=timeframes,
            location_strategies=location_strategies,
            row_count=len(data),
            date_range=(str(data.index.min()), str(data.index.max())),
            probability=probability.probability if probability else None,
            confidence_interval=probability.confidence_interval if probability else None,
            parent_datasets=parent_datasets or [],
            combination_method=combination_method,
            notes=notes,
            tags=tags or []
        )
        
        # Save data (compressed)
        self._save_data(info.filepath, data)
        
        # Save metadata
        self._save_metadata(dataset_id, info)
        
        # Update index
        self.dataset_index[dataset_id] = info
        self._save_index()
        
        return info
    
    def load_dataset(self, dataset_id: str) -> Tuple[pd.DataFrame, DatasetInfo]:
        """Load a dataset and its metadata"""
        if dataset_id not in self.dataset_index:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        info = self.dataset_index[dataset_id]
        data = self._load_data(info.filepath)
        
        return data, info
    
    def update_acceptance(self, dataset_id: str, 
                         status: str,
                         probability: Optional[float] = None,
                         notes: Optional[str] = None):
        """Update dataset acceptance status"""
        if dataset_id not in self.dataset_index:
            raise ValueError(f"Dataset {dataset_id} not found")
            
        info = self.dataset_index[dataset_id]
        info.acceptance_status = status
        
        if probability is not None:
            info.probability = probability
            
        if notes:
            info.notes += f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {notes}"
            
        # Save updated metadata
        self._save_metadata(dataset_id, info)
        self._save_index()
    
    def search_datasets(self, 
                       query: str = "",
                       filters: Dict[str, Any] = None) -> List[DatasetInfo]:
        """Search datasets with query and filters"""
        results = []
        
        for dataset_id, info in self.dataset_index.items():
            # Text search
            if query and not info.matches_search(query):
                continue
                
            # Apply filters
            if filters:
                # Filter by acceptance status
                if 'acceptance_status' in filters:
                    if info.acceptance_status != filters['acceptance_status']:
                        continue
                        
                # Filter by patterns
                if 'patterns' in filters:
                    required_patterns = filters['patterns']
                    if not all(p in info.patterns_included for p in required_patterns):
                        continue
                        
                # Filter by timeframes
                if 'timeframes' in filters:
                    required_timeframes = filters['timeframes']
                    if not all(tf in info.timeframes for tf in required_timeframes):
                        continue
                        
                # Filter by location strategies
                if 'location_strategies' in filters:
                    required_locations = filters['location_strategies']
                    if not all(loc in info.location_strategies for loc in required_locations):
                        continue
                        
                # Filter by probability
                if 'min_probability' in filters and info.probability:
                    if info.probability < filters['min_probability']:
                        continue
                        
                # Filter by date
                if 'created_after' in filters:
                    if info.created_at < filters['created_after']:
                        continue
                        
            results.append(info)
            
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        
        return results
    
    def get_related_datasets(self, dataset_id: str) -> List[DatasetInfo]:
        """Find datasets related to a given dataset"""
        if dataset_id not in self.dataset_index:
            return []
            
        target_info = self.dataset_index[dataset_id]
        related = []
        
        for other_id, other_info in self.dataset_index.items():
            if other_id == dataset_id:
                continue
                
            # Check if it's a parent or child
            if dataset_id in other_info.parent_datasets:
                related.append(other_info)
            elif other_id in target_info.parent_datasets:
                related.append(other_info)
            # Check for similar patterns/strategies
            elif (set(target_info.patterns_included) & set(other_info.patterns_included) and
                  set(target_info.timeframes) & set(other_info.timeframes)):
                related.append(other_info)
                
        return related
    
    def export_dataset(self, dataset_id: str, export_path: str, 
                      format: str = 'csv'):
        """Export dataset to external format"""
        data, info = self.load_dataset(dataset_id)
        
        if format == 'csv':
            data.to_csv(export_path)
        elif format == 'excel':
            data.to_excel(export_path)
        elif format == 'json':
            # Include metadata in JSON export
            export_data = {
                'metadata': info.to_dict(),
                'data': data.to_dict(orient='records')
            }
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
    def _generate_id(self, name: str) -> str:
        """Generate unique dataset ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:6]
        return f"{timestamp}_{name_hash}"
    
    def _save_data(self, filepath: str, data: pd.DataFrame):
        """Save compressed data"""
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    def _load_data(self, filepath: str) -> pd.DataFrame:
        """Load compressed data"""
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)
            
    def _save_metadata(self, dataset_id: str, info: DatasetInfo):
        """Save dataset metadata"""
        filepath = self.metadata_dir / f"{dataset_id}.json"
        with open(filepath, 'w') as f:
            json.dump(info.to_dict(), f, indent=2, default=str)
            
    def _load_index(self):
        """Load dataset index"""
        index_file = self.base_directory / "index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
                
            for dataset_id, info_dict in index_data.items():
                self.dataset_index[dataset_id] = DatasetInfo.from_dict(info_dict)
        else:
            # Rebuild index from metadata files
            for metadata_file in self.metadata_dir.glob("*.json"):
                dataset_id = metadata_file.stem
                with open(metadata_file, 'r') as f:
                    info_dict = json.load(f)
                self.dataset_index[dataset_id] = DatasetInfo.from_dict(info_dict)
                
    def _save_index(self):
        """Save dataset index"""
        index_file = self.base_directory / "index.json"
        index_data = {
            dataset_id: info.to_dict() 
            for dataset_id, info in self.dataset_index.items()
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2, default=str)


class DatasetCombiner:
    """Handles combining multiple datasets"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        
    def combine_datasets(self,
                        dataset_ids: List[str],
                        combination_method: str = 'AND',
                        name: str = None) -> Tuple[pd.DataFrame, ProbabilityMetrics]:
        """Combine multiple datasets with specified logic"""
        
        if len(dataset_ids) < 2:
            raise ValueError("Need at least 2 datasets to combine")
            
        # Load all datasets
        datasets = []
        infos = []
        
        for dataset_id in dataset_ids:
            data, info = self.dataset_manager.load_dataset(dataset_id)
            datasets.append(data)
            infos.append(info)
            
        # Generate combined name if not provided
        if not name:
            name = f"Combined_{combination_method}_" + "_".join([info.name[:10] for info in infos])
            
        # Combine based on method
        if combination_method == 'AND':
            combined_data = self._combine_and(datasets)
        elif combination_method == 'OR':
            combined_data = self._combine_or(datasets)
        elif combination_method == 'XOR':
            combined_data = self._combine_xor(datasets)
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
            
        # Calculate probability for combination
        probability = self._calculate_combination_probability(infos, combination_method)
        
        return combined_data, probability
    
    def _combine_and(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine datasets with AND logic (intersection)"""
        # Start with first dataset
        result = datasets[0]
        
        # Intersect with remaining datasets
        for data in datasets[1:]:
            # Find common indices
            common_idx = result.index.intersection(data.index)
            result = result.loc[common_idx]
            
        return result
    
    def _combine_or(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine datasets with OR logic (union)"""
        # Concatenate all datasets and remove duplicates
        result = pd.concat(datasets, axis=0)
        result = result[~result.index.duplicated(keep='first')]
        return result.sort_index()
    
    def _combine_xor(self, datasets: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine datasets with XOR logic (exclusive or)"""
        # Get indices that appear in odd number of datasets
        all_indices = pd.Index([])
        
        for data in datasets:
            all_indices = all_indices.symmetric_difference(data.index)
            
        # Combine data for these indices
        result_parts = []
        for data in datasets:
            mask = data.index.isin(all_indices)
            if mask.any():
                result_parts.append(data[mask])
                
        if result_parts:
            result = pd.concat(result_parts, axis=0)
            result = result[~result.index.duplicated(keep='first')]
            return result.sort_index()
        else:
            return pd.DataFrame()
    
    def _calculate_combination_probability(self, 
                                         infos: List[DatasetInfo],
                                         method: str) -> ProbabilityMetrics:
        """Calculate probability for dataset combination"""
        metrics = ProbabilityMetrics()
        
        # Extract individual probabilities
        probs = [info.probability for info in infos if info.probability is not None]
        
        if not probs:
            return metrics
            
        # Calculate based on method
        if method == 'AND':
            # Independent assumption
            metrics.probability = np.prod(probs)
        elif method == 'OR':
            # P(A or B) = P(A) + P(B) - P(A and B)
            # Generalized for multiple events
            metrics.probability = 1 - np.prod([1 - p for p in probs])
        elif method == 'XOR':
            # Complex calculation - simplified
            metrics.probability = np.mean(probs) * 0.8  # Penalty for exclusivity
            
        # Confidence interval (simplified)
        if len(probs) >= 2:
            std_err = np.std(probs) / np.sqrt(len(probs))
            metrics.confidence_interval = (
                max(0, metrics.probability - 1.96 * std_err),
                min(1, metrics.probability + 1.96 * std_err)
            )
            
        return metrics
