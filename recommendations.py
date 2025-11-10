"""
Recommendation generation based on feature impacts and model analysis
"""

import pandas as pd
import streamlit as st


def generate_data_driven_recommendations(prediction_results, input_features, data_analysis):
    """Generate recommendations based on improvement analysis"""
    recommendations = []

    # Get feature impacts and importance data
    feature_impacts = data_analysis.get('feature_impacts', {})
    feature_importance_owners = data_analysis.get('feature_importance_owners', pd.DataFrame())
    feature_importance_reviews = data_analysis.get('feature_importance_reviews', pd.DataFrame())

    owners_pred = prediction_results['owners']
    review_pred = prediction_results['review_ratio']
    price = input_features.get('price', 0)

    # Identify top improvement opportunities based on feature impacts
    if feature_impacts:
        # Find features NOT currently used that would help
        top_owner_improvements = sorted(
            [(k, v) for k, v in feature_impacts.items() if v.get('owners_improvement_pct', 0) > 0],
            key=lambda x: x[1].get('owners_improvement_pct', 0),
            reverse=True
        )[:5]

        top_review_improvements = sorted(
            [(k, v) for k, v in feature_impacts.items() if v.get('reviews_improvement_pct', 0) > 0],
            key=lambda x: x[1].get('reviews_improvement_pct', 0),
            reverse=True
        )[:3]

        # Recommend top owners improvement
        if top_owner_improvements:
            feat, impact = top_owner_improvements[0]
            feat_name = feat.replace('tag_', '').replace('_', ' ').title()
            pct = impact['owners_improvement_pct']

            if impact['type'] == 'binary':
                with_val = impact['owners_with']
                without_val = impact['owners_without']
                recommendations.append({
                    'type': f'Top Opportunity: {feat_name}',
                    'message': f'Games with {feat_name} have {pct:+.0f}% more owners ({without_val:,} → {with_val:,}). This is the biggest success factor in the dataset.',
                    'priority': 'High',
                    'data_confidence': 'Strong'
                })
            else:
                top_val = impact['owners_top25']
                bottom_val = impact['owners_bottom25']
                recommendations.append({
                    'type': f'Top Opportunity: {feat_name}',
                    'message': f'Higher {feat_name} correlates with {pct:+.0f}% more owners (top 25%: {top_val:,} vs bottom 25%: {bottom_val:,}).',
                    'priority': 'High',
                    'data_confidence': 'Strong'
                })

        # Recommend top review improvement
        if top_review_improvements:
            feat, impact = top_review_improvements[0]
            feat_name = feat.replace('tag_', '').replace('_', ' ').title()
            pct = impact['reviews_improvement_pct']

            if impact['type'] == 'binary':
                with_val = impact['reviews_with']
                without_val = impact['reviews_without']
                recommendations.append({
                    'type': f'Review Quality: {feat_name}',
                    'message': f'Games with {feat_name} have {pct:+.0f}% better review ratios ({without_val:.1%} → {with_val:.1%}).',
                    'priority': 'Medium',
                    'data_confidence': 'Strong'
                })

    # Model confidence indicator
    feature_count = len(feature_impacts) if feature_impacts else len(feature_importance_owners)
    recommendations.append({
        'type': 'Model Information',
        'message': f'Analysis based on {feature_count} features from real Steam data ({len(st.session_state.models.get("X_test", []))} test games). Improvements are calculated from actual game performance.',
        'priority': 'Info',
        'data_confidence': 'Model Metric'
    })

    return recommendations
