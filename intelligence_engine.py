"""
Intelligence Engine for Game Launch Decision Support System

Provides market positioning and actionable improvement recommendations.

Key Metrics:
- Ranking model Spearman correlation: ~0.72 (reliable for market positioning)
- Uses machine learning on 27,000+ Steam games
- Focuses on relative positioning and actionable insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ranking_integration import IntelligentRanker


class SuccessTier(Enum):
    EXCEPTIONAL = "exceptional"
    STRONG = "strong"
    ABOVE_AVERAGE = "above_average"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    STRUGGLING = "struggling"


class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GameInsight:
    category: str
    title: str
    message: str
    impact: str
    action_type: str
    data: Optional[Dict] = None


@dataclass
class ImprovementScenario:
    change_description: str
    current_percentile: float
    new_percentile: float
    percentile_gain: float
    confidence: str
    similar_games_count: int
    historical_lift: str = "N/A"


class IntelligenceEngine:
    """
    Transforms ML predictions into actionable business intelligence.

    Uses ranking model with Spearman correlation ~0.72 for market positioning.
    Improvements are based on the ranking model with confidence from historical data.
    """

    def __init__(self, df: pd.DataFrame, models: Dict):
        self.df = df
        self.models = models
        self.feature_cols = models.get('feature_cols', [])

        # Initialize ranking system
        print("🎯 Initializing Intelligent Ranking System...")
        self.ranker = IntelligentRanker(df)
        self.ranker.train()

        # Store ranking performance metric
        self.random_spearman = self.ranker.random_spearman
        self.temporal_spearman = self.ranker.temporal_spearman

        # Use random split for ranking quality assessment
        self.spearman_score = self.random_spearman
        self.ranking_reliable = self.random_spearman >= 0.60

        print(f"   Spearman Correlation: {self.random_spearman:.4f}")
        print(f"   Reliable (>=0.60): {'✅ Yes' if self.ranking_reliable else '⚠️ No'}")

        self._compute_market_statistics()
    
    def _compute_market_statistics(self):
        """Pre-compute market statistics."""
        self.market_stats = {
            'overall': {
                'median_owners': self.df['owners'].median(),
                'mean_owners': self.df['owners'].mean(),
                'total_games': len(self.df)
            }
        }
        
        # Genre stats
        self.genre_stats = {}
        for col in self.df.columns:
            if col.startswith('genre_'):
                genre_df = self.df[self.df[col] == 1]
                if len(genre_df) > 50:
                    genre_name = col.replace('genre_', '')
                    self.genre_stats[genre_name] = {
                        'median_owners': genre_df['owners'].median(),
                        'count': len(genre_df)
                    }
    
    def analyze_game(self, features: Dict) -> Dict:
        """Main analysis function."""
        ranker_features = self._build_ranker_features(features)
        
        # Get percentile from ranking model
        percentile = self.ranker.get_percentile(ranker_features)
        
        # Get improvements from ranking model
        ranker_improvements = self.ranker.get_improvements(ranker_features, top_n=6)
        
        positioning = self._compute_positioning(percentile, features)
        improvements = self._convert_improvements(ranker_improvements, percentile)
        risks = self._assess_risks(features, positioning)
        success_factors = self._analyze_success_factors(features)
        insights = self._generate_insights(features, positioning, improvements, risks, success_factors)
        
        return {
            'positioning': positioning,
            'improvements': improvements,
            'risks': risks,
            'success_factors': success_factors,
            'insights': insights,
            'confidence_statement': self._generate_confidence_statement(positioning),
            'ranking_quality': {
                'spearman': self.random_spearman,
                'reliable': self.ranking_reliable,
                'note': 'Based on cross-validation ranking performance'
            }
        }
    
    def _build_ranker_features(self, features: Dict) -> Dict:
        """Convert user features to ranker format."""
        rf = {}
        
        # Price
        price = features.get('price', 0)
        rf['price'] = price
        rf['price_log'] = np.log1p(price)
        rf['is_free'] = 1 if price == 0 else 0
        rf['price_tier'] = 0 if price == 0 else 1 if price <= 5 else 2 if price <= 10 else 3 if price <= 20 else 4 if price <= 40 else 5
        
        # Platforms
        platforms = features.get('platforms', ['windows'])
        rf['windows'] = 1 if 'windows' in platforms else 0
        rf['mac'] = 1 if 'mac' in platforms else 0
        rf['linux'] = 1 if 'linux' in platforms else 0
        rf['platform_count'] = rf['windows'] + rf['mac'] + rf['linux']
        
        # Achievements
        achievements = features.get('achievements', 0)
        rf['achievements'] = achievements
        rf['has_achievements'] = 1 if achievements > 0 else 0
        rf['achievements_log'] = np.log1p(achievements)
        
        # Time
        rf['game_age_days'] = features.get('game_age_days', 100)
        rf['game_age_log'] = np.log1p(rf['game_age_days'])
        rf['release_year'] = features.get('release_year', 2024)
        rf['release_month'] = features.get('release_month', 6)
        
        # Genres
        for genre in features.get('genres', []):
            rf[f'genre_{genre.lower().replace(" ", "_")}'] = 1
        
        # Tags
        for tag in features.get('tags', []):
            rf[f'tag_{tag.lower().replace(" ", "_").replace("-", "_")}'] = 1
        
        # Categories
        for cat in features.get('categories', []):
            rf[f'cat_{cat.lower().replace(" ", "_").replace("-", "_")}'] = 1
        
        # Developer
        rf['dev_game_count'] = features.get('dev_game_count', 1)
        rf['dev_game_count_log'] = np.log1p(rf['dev_game_count'])
        
        return rf
    
    def _compute_positioning(self, percentile: float, features: Dict) -> Dict:
        """Compute market positioning."""
        if percentile >= 95:
            tier = SuccessTier.EXCEPTIONAL
        elif percentile >= 80:
            tier = SuccessTier.STRONG
        elif percentile >= 60:
            tier = SuccessTier.ABOVE_AVERAGE
        elif percentile >= 40:
            tier = SuccessTier.AVERAGE
        elif percentile >= 20:
            tier = SuccessTier.BELOW_AVERAGE
        else:
            tier = SuccessTier.STRUGGLING
        
        return {
            'overall_percentile': percentile,
            'tier': tier,
            'tier_name': tier.value,
            'ranking_confidence': 'High' if self.ranking_reliable else 'Moderate'
        }
    
    def _convert_improvements(self, ranker_improvements: List[Dict], current_percentile: float) -> List[ImprovementScenario]:
        """Convert ranker improvements."""
        return [
            ImprovementScenario(
                change_description=imp['description'],
                current_percentile=current_percentile,
                new_percentile=imp['new_percentile'],
                percentile_gain=imp['percentile_gain'],
                confidence=imp['confidence'],
                similar_games_count=0,
                historical_lift=imp.get('historical_lift', 'N/A')
            )
            for imp in ranker_improvements
        ]
    
    def _assess_risks(self, features: Dict, positioning: Dict) -> List[Dict]:
        """Assess risks."""
        risks = []
        
        if len(features.get('platforms', [])) == 1:
            risks.append({
                'level': 'moderate',
                'title': 'Limited Platform Support',
                'description': 'Single platform limits audience.',
                'recommendation': 'Consider Mac/Linux ports.'
            })
        
        if features.get('price', 0) > 40:
            risks.append({
                'level': 'high',
                'title': 'Premium Pricing Risk',
                'description': 'Competing with AAA titles.',
                'recommendation': 'Ensure clear value proposition.'
            })
        
        if positioning['overall_percentile'] < 30:
            risks.append({
                'level': 'high',
                'title': 'Below-Average Position',
                'description': 'In bottom third of market.',
                'recommendation': 'Review improvement scenarios.'
            })
        
        return risks
    
    def _analyze_success_factors(self, features: Dict) -> Dict:
        """Analyze success factors."""
        present = []
        missing = []
        
        if len(features.get('platforms', [])) >= 2:
            present.append({'factor': 'Multi-platform', 'impact': 'high'})
        else:
            missing.append({'factor': 'Multi-platform', 'impact': 'high'})
        
        if features.get('achievements', 0) > 0:
            present.append({'factor': 'Achievements', 'impact': 'medium'})
        else:
            missing.append({'factor': 'Achievements', 'impact': 'medium'})
        
        categories = [c.lower() for c in features.get('categories', [])]
        if 'steam trading cards' in categories:
            present.append({'factor': 'Trading Cards', 'impact': 'medium'})
        else:
            missing.append({'factor': 'Trading Cards', 'impact': 'medium'})
        
        score = sum(3 if f['impact'] == 'high' else 2 for f in present)
        
        return {'present': present, 'missing': missing, 'readiness_score': min(100, score * 10)}
    
    def _generate_insights(self, features, positioning, improvements, risks, success_factors) -> List[GameInsight]:
        """Generate insights."""
        insights = []
        pct = positioning['overall_percentile']
        
        # Position insight
        if pct >= 70:
            insights.append(GameInsight(
                "Position", "Strong Position",
                f"Top {100-pct:.0f}% of market. High confidence ranking (Spearman: {self.random_spearman:.2f})",
                "high", "success"
            ))
        elif pct >= 40:
            insights.append(GameInsight(
                "Position", "Average Position",
                f"{pct:.0f}th percentile. See improvements below.",
                "medium", "info"
            ))
        else:
            insights.append(GameInsight(
                "Position", "Below Average",
                f"Bottom {pct:.0f}%. Review recommendations.",
                "high", "warning"
            ))
        
        # Top improvement
        if improvements:
            imp = improvements[0]
            insights.append(GameInsight(
                "Opportunity", imp.change_description,
                f"+{imp.percentile_gain:.1f}pp → {imp.new_percentile:.0f}th. Confidence: {imp.confidence}",
                "high", "opportunity"
            ))
        
        return insights
    
    def _generate_confidence_statement(self, positioning: Dict) -> str:
        """Generate confidence statement."""
        reliability = "High" if self.ranking_reliable else "Moderate"
        return (
            f"Based on {self.market_stats['overall']['total_games']:,} Steam games. "
            f"Ranking model Spearman correlation: {self.random_spearman:.2f}. "
            f"Confidence level: {reliability}. "
            f"Use insights for strategic guidance and relative positioning."
        )
    
    def get_ranker_stats(self) -> Dict:
        """Get ranking statistics."""
        return self.ranker.get_stats()


def create_intelligence_engine(df: pd.DataFrame, models: Dict) -> IntelligenceEngine:
    """Factory function."""
    return IntelligenceEngine(df, models)
