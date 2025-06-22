"""
statistics1/probability_calculator.py
====================================
Statistical methods for strategy evaluation and combination
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

from core.data_structures import ProbabilityMetrics

logger = logging.getLogger(__name__)


class ProbabilityCalculator:
    """Calculates probabilities for strategy combinations"""

    def __init__(self):
        self.calculation_methods = {
            'bayesian': self.calculate_bayesian_probability,
            'frequency': self.calculate_frequency_probability,
            'monte_carlo': self.calculate_monte_carlo_probability,
            'conditional': self.calculate_conditional_probability
        }

    def calculate_combination_probability(self,
                                          strategy1_metrics: ProbabilityMetrics,
                                          strategy2_metrics: ProbabilityMetrics,
                                          combination_data: pd.DataFrame,
                                          method: str = 'bayesian') -> ProbabilityMetrics:
        """Calculate probability for strategy combination"""
        if method not in self.calculation_methods:
            raise ValueError(f"Unknown method: {method}")

        return self.calculation_methods[method](
            strategy1_metrics,
            strategy2_metrics,
            combination_data
        )

    def calculate_bayesian_probability(self,
                                       strategy1_metrics: ProbabilityMetrics,
                                       strategy2_metrics: ProbabilityMetrics,
                                       combination_data: pd.DataFrame) -> ProbabilityMetrics:
        """Use Bayesian inference for combination probability"""
        # Prior probabilities
        p_s1 = strategy1_metrics.probability
        p_s2 = strategy2_metrics.probability

        # Create metrics object
        metrics = ProbabilityMetrics()

        # Calculate joint occurrences
        if 'strategy1_signal' in combination_data and 'strategy2_signal' in combination_data:
            joint_signals = combination_data['strategy1_signal'] & combination_data['strategy2_signal']
            joint_success = combination_data['success'] & joint_signals

            # Likelihood
            p_success_given_both = joint_success.sum() / max(joint_signals.sum(), 1)

            # Prior for success
            p_success = combination_data['success'].mean()

            # Posterior probability using Bayes' theorem
            if p_success > 0:
                p_both_given_success = (p_success_given_both * p_s1 * p_s2) / p_success
            else:
                p_both_given_success = 0

        else:
            # Fallback to independent assumption
            p_both_given_success = p_s1 * p_s2

        # Update metrics
        metrics.probability = p_both_given_success
        metrics.occurrence_count = int(joint_signals.sum()) if 'joint_signals' in locals() else 0
        metrics.total_opportunities = len(combination_data)

        # Calculate confidence interval
        if metrics.occurrence_count >= 30:
            # Use beta distribution for Bayesian confidence interval
            alpha = metrics.occurrence_count * metrics.probability + 1
            beta = metrics.occurrence_count * (1 - metrics.probability) + 1

            metrics.confidence_interval = (
                stats.beta.ppf(0.025, alpha, beta),
                stats.beta.ppf(0.975, alpha, beta)
            )
            metrics.sample_size_adequate = True

        return metrics

    def calculate_frequency_probability(self,
                                        strategy1_metrics: ProbabilityMetrics,
                                        strategy2_metrics: ProbabilityMetrics,
                                        combination_data: pd.DataFrame) -> ProbabilityMetrics:
        """Simple frequency-based probability"""
        metrics = ProbabilityMetrics()

        if 'strategy1_signal' in combination_data and 'strategy2_signal' in combination_data:
            joint_signals = combination_data['strategy1_signal'] & combination_data['strategy2_signal']

            metrics.occurrence_count = joint_signals.sum()
            metrics.total_opportunities = len(combination_data)

            if 'success' in combination_data:
                joint_success = combination_data['success'] & joint_signals
                metrics.success_count = joint_success.sum()
                metrics.failure_count = metrics.occurrence_count - metrics.success_count

        metrics.calculate_probability()
        return metrics

    def calculate_monte_carlo_probability(self,
                                          strategy1_metrics: ProbabilityMetrics,
                                          strategy2_metrics: ProbabilityMetrics,
                                          combination_data: pd.DataFrame,
                                          n_simulations: int = 10000) -> ProbabilityMetrics:
        """Monte Carlo simulation for probability estimation"""
        metrics = ProbabilityMetrics()

        # Run simulations
        success_rates = []

        for _ in range(n_simulations):
            # Simulate strategy occurrences
            s1_occurs = np.random.random(len(combination_data)) < strategy1_metrics.probability
            s2_occurs = np.random.random(len(combination_data)) < strategy2_metrics.probability

            both_occur = s1_occurs & s2_occurs

            if both_occur.sum() > 0:
                # Simulate success given both strategies
                base_success_rate = (strategy1_metrics.probability + strategy2_metrics.probability) / 2
                success = np.random.random(both_occur.sum()) < base_success_rate
                success_rates.append(success.mean())

        if success_rates:
            metrics.probability = np.mean(success_rates)
            metrics.confidence_interval = (
                np.percentile(success_rates, 2.5),
                np.percentile(success_rates, 97.5)
            )
            metrics.sample_size_adequate = True

        return metrics

    def calculate_conditional_probability(self,
                                          strategy1_metrics: ProbabilityMetrics,
                                          strategy2_metrics: ProbabilityMetrics,
                                          combination_data: pd.DataFrame) -> ProbabilityMetrics:
        """Calculate conditional probability P(Success|S1 and S2)"""
        metrics = ProbabilityMetrics()

        if all(col in combination_data for col in ['strategy1_signal', 'strategy2_signal', 'success']):
            # Calculate P(Success|S1 and S2)
            both_signals = combination_data['strategy1_signal'] & combination_data['strategy2_signal']

            if both_signals.sum() > 0:
                success_given_both = combination_data.loc[both_signals, 'success'].mean()
                metrics.probability = success_given_both
                metrics.occurrence_count = both_signals.sum()
                metrics.success_count = combination_data.loc[both_signals, 'success'].sum()
                metrics.failure_count = metrics.occurrence_count - metrics.success_count
                metrics.total_opportunities = len(combination_data)

                # Calculate confidence interval
                if metrics.occurrence_count >= 30:
                    se = np.sqrt(metrics.probability * (1 - metrics.probability) / metrics.occurrence_count)
                    metrics.confidence_interval = (
                        max(0, metrics.probability - 1.96 * se),
                        min(1, metrics.probability + 1.96 * se)
                    )
                    metrics.sample_size_adequate = True

        return metrics


class StatisticalValidator:
    """Validates strategy combinations using statistical tests"""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def validate_combination(self,
                             strategy1_data: pd.DataFrame,
                             strategy2_data: pd.DataFrame,
                             combined_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical validation of strategy combination"""
        validation_results = {
            'is_valid': False,
            'tests_performed': [],
            'recommendations': []
        }

        # Test 1: Independence test (Chi-square)
        independence_result = self.test_independence(strategy1_data, strategy2_data)
        validation_results['tests_performed'].append(independence_result)

        # Test 2: Performance improvement test (T-test)
        improvement_result = self.test_performance_improvement(
            strategy1_data, strategy2_data, combined_data
        )
        validation_results['tests_performed'].append(improvement_result)

        # Test 3: Correlation analysis
        correlation_result = self.analyze_correlation(strategy1_data, strategy2_data)
        validation_results['tests_performed'].append(correlation_result)

        # Test 4: Risk-adjusted performance (Sharpe ratio test)
        sharpe_result = self.test_sharpe_improvement(
            strategy1_data, strategy2_data, combined_data
        )
        validation_results['tests_performed'].append(sharpe_result)

        # Overall validation
        passed_tests = sum(1 for test in validation_results['tests_performed']
                           if test['passed'])
        validation_results['is_valid'] = passed_tests >= 3  # Need at least 3/4 tests

        # Generate recommendations
        validation_results['recommendations'] = self.generate_recommendations(
            validation_results['tests_performed']
        )

        return validation_results

    def test_independence(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
        """Test if two strategies are independent using Chi-square test"""
        result = {
            'test': 'Chi-square Independence Test',
            'passed': False,
            'p_value': None,
            'statistic': None,
            'interpretation': ''
        }

        if 'signal' in data1 and 'signal' in data2:
            # Create contingency table
            merged = pd.merge(data1[['signal']], data2[['signal']],
                              left_index=True, right_index=True,
                              suffixes=('_1', '_2'))

            contingency = pd.crosstab(merged['signal_1'], merged['signal_2'])

            # Perform Chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            result['statistic'] = chi2
            result['p_value'] = p_value
            result['passed'] = p_value < self.significance_level

            if result['passed']:
                result['interpretation'] = "Strategies show significant dependence (good for combination)"
            else:
                result['interpretation'] = "Strategies appear independent"

        return result

    def test_performance_improvement(self,
                                     data1: pd.DataFrame,
                                     data2: pd.DataFrame,
                                     combined: pd.DataFrame) -> Dict[str, Any]:
        """Test if combination improves performance using T-test"""
        result = {
            'test': 'Performance Improvement T-test',
            'passed': False,
            'p_value': None,
            'statistic': None,
            'interpretation': ''
        }

        if 'returns' in data1 and 'returns' in data2 and 'returns' in combined:
            # Compare combined returns vs individual returns
            individual_returns = pd.concat([data1['returns'], data2['returns']])
            combined_returns = combined['returns']

            # Perform T-test
            t_stat, p_value = stats.ttest_ind(combined_returns, individual_returns)

            result['statistic'] = t_stat
            result['p_value'] = p_value
            result['passed'] = p_value < self.significance_level and t_stat > 0

            if result['passed']:
                result['interpretation'] = "Combination significantly improves returns"
            else:
                result['interpretation'] = "No significant improvement in returns"

        return result

    def analyze_correlation(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation between strategies"""
        result = {
            'test': 'Correlation Analysis',
            'passed': False,
            'correlation': None,
            'interpretation': ''
        }

        if 'returns' in data1 and 'returns' in data2:
            # Calculate correlation
            correlation = data1['returns'].corr(data2['returns'])
            result['correlation'] = correlation

            # Ideal correlation is between -0.3 and 0.3 (low correlation)
            result['passed'] = abs(correlation) < 0.3

            if correlation > 0.7:
                result['interpretation'] = "Strategies are highly correlated (poor diversification)"
            elif correlation < -0.7:
                result['interpretation'] = "Strategies are negatively correlated (good for hedging)"
            elif abs(correlation) < 0.3:
                result['interpretation'] = "Strategies have low correlation (good for combination)"
            else:
                result['interpretation'] = "Strategies have moderate correlation"

        return result

    def test_sharpe_improvement(self,
                                data1: pd.DataFrame,
                                data2: pd.DataFrame,
                                combined: pd.DataFrame) -> Dict[str, Any]:
        """Test if combination improves risk-adjusted returns"""
        result = {
            'test': 'Sharpe Ratio Improvement',
            'passed': False,
            'sharpe_1': None,
            'sharpe_2': None,
            'sharpe_combined': None,
            'interpretation': ''
        }

        if 'returns' in data1 and 'returns' in data2 and 'returns' in combined:
            # Calculate Sharpe ratios
            result['sharpe_1'] = self._calculate_sharpe(data1['returns'])
            result['sharpe_2'] = self._calculate_sharpe(data2['returns'])
            result['sharpe_combined'] = self._calculate_sharpe(combined['returns'])

            # Check if combined Sharpe is better than both individual
            result['passed'] = (result['sharpe_combined'] > result['sharpe_1'] and
                                result['sharpe_combined'] > result['sharpe_2'])

            if result['passed']:
                improvement = ((result['sharpe_combined'] -
                                max(result['sharpe_1'], result['sharpe_2'])) /
                               max(result['sharpe_1'], result['sharpe_2']) * 100)
                result['interpretation'] = f"Risk-adjusted returns improved by {improvement:.1f}%"
            else:
                result['interpretation'] = "No improvement in risk-adjusted returns"

        return result

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

        if excess_returns.std() > 0:
            return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            return 0

    def generate_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        for test in test_results:
            if test['test'] == 'Chi-square Independence Test':
                if not test['passed']:
                    recommendations.append(
                        "Consider strategies with more complementary signals"
                    )

            elif test['test'] == 'Performance Improvement T-test':
                if not test['passed']:
                    recommendations.append(
                        "Combination may not add value - review strategy logic"
                    )

            elif test['test'] == 'Correlation Analysis':
                if test['correlation'] and abs(test['correlation']) > 0.7:
                    recommendations.append(
                        "High correlation reduces diversification benefits"
                    )

            elif test['test'] == 'Sharpe Ratio Improvement':
                if not test['passed']:
                    recommendations.append(
                        "Consider adjusting position sizing or risk parameters"
                    )

        if not recommendations:
            recommendations.append("Strategy combination appears statistically sound")

        return recommendations


class AcceptanceCalculator:
    """Calculates acceptance criteria for strategy combinations"""

    def __init__(self):
        self.criteria_weights = {
            'probability': 0.3,
            'sharpe_ratio': 0.25,
            'max_drawdown': 0.2,
            'consistency': 0.15,
            'sample_size': 0.1
        }

    def calculate_acceptance_score(self,
                                   strategy_metrics: Dict[str, Any],
                                   backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall acceptance score for a strategy"""
        scores = {}

        # Probability score (0-1)
        if 'probability' in strategy_metrics:
            prob = strategy_metrics['probability']
            scores['probability'] = min(prob / 0.6, 1.0)  # 60% probability = max score

        # Sharpe ratio score (0-1)
        if 'sharpe_ratio' in backtest_results:
            sharpe = backtest_results['sharpe_ratio']
            scores['sharpe_ratio'] = min(sharpe / 2.0, 1.0)  # Sharpe of 2 = max score

        # Max drawdown score (0-1)
        if 'max_drawdown' in backtest_results:
            dd = abs(backtest_results['max_drawdown'])
            scores['max_drawdown'] = max(0, 1 - dd / 0.2)  # 20% drawdown = 0 score

        # Consistency score (0-1)
        if 'win_rate' in backtest_results:
            wr = backtest_results['win_rate']
            scores['consistency'] = wr  # Win rate directly as score

        # Sample size score (0-1)
        if 'sample_size' in strategy_metrics:
            samples = strategy_metrics['sample_size']
            scores['sample_size'] = min(samples / 100, 1.0)  # 100+ samples = max score

        # Calculate weighted score
        weighted_score = sum(scores.get(criterion, 0) * weight
                             for criterion, weight in self.criteria_weights.items())

        # Determine acceptance
        acceptance_result = {
            'score': weighted_score,
            'scores_breakdown': scores,
            'accepted': weighted_score >= 0.6,  # 60% threshold
            'confidence': self._calculate_confidence(scores),
            'recommendation': self._generate_recommendation(weighted_score, scores)
        }

        return acceptance_result

    def _calculate_confidence(self, scores: Dict[str, float]) -> str:
        """Calculate confidence level based on score consistency"""
        if not scores:
            return 'low'

        score_values = list(scores.values())
        std_dev = np.std(score_values)

        if std_dev < 0.15:
            return 'high'
        elif std_dev < 0.25:
            return 'medium'
        else:
            return 'low'

    def _generate_recommendation(self,
                                 overall_score: float,
                                 scores: Dict[str, float]) -> str:
        """Generate recommendation based on scores"""
        if overall_score >= 0.8:
            return "Highly recommended - excellent metrics across all criteria"
        elif overall_score >= 0.6:
            weakest = min(scores.items(), key=lambda x: x[1])
            return f"Acceptable - consider improving {weakest[0]}"
        else:
            return "Not recommended - significant improvements needed"