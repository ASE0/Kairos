"""
core/data_structures.py
=======================
Core data structures for the trading strategy hub
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import uuid


@dataclass
class TimeRange:
    """Represents a time range with value and unit"""
    value: int
    unit: str  # 's', 'm', 'h', 'd'
    
    def to_seconds(self) -> int:
        """Convert to seconds"""
        multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        return self.value * multipliers.get(self.unit, 1)
    
    def __str__(self):
        return f"{self.value}{self.unit}"
    
    def to_dict(self):
        return {'value': self.value, 'unit': self.unit}


@dataclass
class OHLCRatio:
    """Defines ratios between OHLC values"""
    body_ratio: Optional[float] = None  # (close-open)/(high-low)
    upper_wick_ratio: Optional[float] = None  # (high-max(open,close))/(high-low)
    lower_wick_ratio: Optional[float] = None  # (min(open,close)-low)/(high-low)
    custom_formula: Optional[str] = None  # Custom formula string
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class VolatilityProfile:
    """Volatility categorization for a dataset"""
    value: int  # 0-100
    category: str  # 'low', 'medium', 'high', 'extreme'
    calculated_metrics: Dict[str, float] = field(default_factory=dict)
    user_defined: bool = True
    
    @staticmethod
    def categorize(value: int) -> str:
        """Categorize volatility based on value"""
        if value < 25:
            return 'low'
        elif value < 50:
            return 'medium'
        elif value < 75:
            return 'high'
        else:
            return 'extreme'
    
    def __post_init__(self):
        if not self.category:
            self.category = self.categorize(self.value)


@dataclass
class DatasetMetadata:
    """Metadata for a processed dataset"""
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None
    rows_original: int = 0
    rows_processed: int = 0
    columns_kept: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    filters_applied: List[Dict[str, Any]] = field(default_factory=list)
    volatility: Optional[VolatilityProfile] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self):
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class StrategyConstraint:
    """Represents a constraint applied to a strategy"""
    type: str  # 'time_range', 'volatility', 'volume', etc.
    operator: str  # '>', '<', '==', 'between', etc.
    value: Any
    description: str
    
    def to_dict(self):
        return asdict(self)
    
    def evaluate(self, data: Any) -> bool:
        """Evaluate if data meets this constraint"""
        # Implementation depends on constraint type
        pass


@dataclass
class ProbabilityMetrics:
    """Statistical metrics for probability calculation"""
    occurrence_count: int = 0
    total_opportunities: int = 0
    success_count: int = 0
    failure_count: int = 0
    probability: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    statistical_significance: float = 0.0
    sample_size_adequate: bool = False
    
    def calculate_probability(self):
        """Calculate probability with confidence intervals"""
        if self.total_opportunities > 0:
            self.probability = self.success_count / self.total_opportunities
            
            # Calculate 95% confidence interval using Wilson score
            if self.total_opportunities >= 30:
                from scipy import stats
                z = 1.96  # 95% confidence
                p = self.probability
                n = self.total_opportunities
                
                denominator = 1 + z**2/n
                center = (p + z**2/(2*n)) / denominator
                margin = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / denominator
                
                self.confidence_interval = (
                    max(0, center - margin),
                    min(1, center + margin)
                )
                self.sample_size_adequate = True
            else:
                self.sample_size_adequate = False
                
    def to_dict(self):
        return asdict(self)


@dataclass
class BaseStrategy:
    """Base class for all strategy types"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ""  # 'pattern', 'risk', 'combined'
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    constraints: List[StrategyConstraint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    probability_metrics: Optional[ProbabilityMetrics] = None
    
    def to_json(self, pretty=True):
        """Convert to JSON string"""
        data = self.to_dict()
        return json.dumps(data, indent=2 if pretty else None, default=str)
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['modified_at'] = self.modified_at.isoformat()
        return data
    
    def add_constraint(self, constraint: StrategyConstraint):
        """Add a constraint to this strategy"""
        self.constraints.append(constraint)
        self.modified_at = datetime.now()
    
    def update_probability(self, metrics: ProbabilityMetrics):
        """Update probability metrics"""
        self.probability_metrics = metrics
        self.modified_at = datetime.now()
