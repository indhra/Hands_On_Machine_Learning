#!/usr/bin/env python3
"""
ML-Based Product Analysis Demo
Demonstrating what can be done with this ML repository vs. external web access

This script shows the ML capabilities available in this repository for
product analysis, while clearly explaining the limitations regarding
external web access.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create realistic sample refrigerator data for demonstration."""
    np.random.seed(42)
    
    brands = ['Samsung', 'LG', 'Whirlpool', 'GE', 'Frigidaire', 'KitchenAid', 'Bosch']
    types = ['Top Freezer', 'Bottom Freezer', 'Side by Side', 'French Door', 'Compact']
    energy_ratings = ['A+++', 'A++', 'A+', 'A', 'B']
    
    n_samples = 100  # Smaller dataset for quick demo
    
    # Generate features
    data = {
        'brand': np.random.choice(brands, n_samples),
        'type': np.random.choice(types, n_samples),
        'capacity_liters': np.random.normal(400, 150, n_samples).clip(150, 800),
        'energy_rating': np.random.choice(energy_ratings, n_samples),
        'width_cm': np.random.normal(60, 10, n_samples).clip(50, 90),
        'height_cm': np.random.normal(180, 20, n_samples).clip(150, 220),
        'has_ice_maker': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'has_water_dispenser': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'warranty_years': np.random.choice([1, 2, 3, 5], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
        'avg_rating': np.random.normal(4.2, 0.8, n_samples).clip(1, 5),
        'num_reviews': np.random.exponential(50, n_samples).astype(int).clip(1, 500)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate realistic prices based on features
    base_price = 500
    brand_multiplier = {'Samsung': 1.2, 'LG': 1.15, 'Whirlpool': 1.0, 'GE': 1.1, 
                       'Frigidaire': 0.9, 'KitchenAid': 1.3, 'Bosch': 1.25}
    type_multiplier = {'Top Freezer': 0.8, 'Bottom Freezer': 1.0, 'Side by Side': 1.1,
                      'French Door': 1.3, 'Compact': 0.6}
    energy_multiplier = {'A+++': 1.2, 'A++': 1.1, 'A+': 1.05, 'A': 1.0, 'B': 0.95}
    
    df['price'] = (
        base_price +
        df['capacity_liters'] * 1.5 +
        df['brand'].map(brand_multiplier) * 200 +
        df['type'].map(type_multiplier) * 300 +
        df['energy_rating'].map(energy_multiplier) * 100 +
        df['has_ice_maker'] * 150 +
        df['has_water_dispenser'] * 100 +
        df['warranty_years'] * 50 +
        np.random.normal(0, 100, n_samples)  # Add some noise
    ).round(0).astype(int)
    
    return df

def build_price_prediction_model(df):
    """Build and evaluate a price prediction model."""
    # Encode categorical variables
    le_brand = LabelEncoder()
    le_type = LabelEncoder()
    le_energy = LabelEncoder()
    
    df_model = df.copy()
    df_model['brand_encoded'] = le_brand.fit_transform(df['brand'])
    df_model['type_encoded'] = le_type.fit_transform(df['type'])
    df_model['energy_encoded'] = le_energy.fit_transform(df['energy_rating'])
    
    # Prepare features
    features = ['capacity_liters', 'width_cm', 'height_cm', 'has_ice_maker',
               'has_water_dispenser', 'warranty_years', 'brand_encoded', 
               'type_encoded', 'energy_encoded']
    
    X = df_model[features]
    y = df_model['price']
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'model': model,
        'features': features,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'encoders': {'brand': le_brand, 'type': le_type, 'energy': le_energy}
    }

def find_best_value_products(df, top_n=5):
    """Find products with the best value (high rating, reasonable price)."""
    # Calculate value score
    price_normalized = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
    rating_normalized = (df['avg_rating'] - df['avg_rating'].min()) / (df['avg_rating'].max() - df['avg_rating'].min())
    
    df['value_score'] = rating_normalized - price_normalized
    
    return df.nlargest(top_n, 'value_score')[['brand', 'type', 'capacity_liters', 
                                             'price', 'avg_rating', 'value_score']]

def recommend_refrigerator(df, budget_max=None, min_capacity=None, preferred_brand=None, 
                          min_rating=None, top_n=3):
    """Recommend refrigerators based on criteria."""
    filtered_df = df.copy()
    
    if budget_max:
        filtered_df = filtered_df[filtered_df['price'] <= budget_max]
    
    if min_capacity:
        filtered_df = filtered_df[filtered_df['capacity_liters'] >= min_capacity]
    
    if preferred_brand:
        filtered_df = filtered_df[filtered_df['brand'] == preferred_brand]
    
    if min_rating:
        filtered_df = filtered_df[filtered_df['avg_rating'] >= min_rating]
    
    if len(filtered_df) == 0:
        return "No refrigerators match your criteria."
    
    return filtered_df.nlargest(top_n, 'value_score')[['brand', 'type', 'capacity_liters', 
                                                       'price', 'avg_rating']]

def main():
    print("="*80)
    print("ML-BASED PRODUCT ANALYSIS DEMO")
    print("Demonstrating ML Capabilities vs External Web Access Limitations")
    print("="*80)
    
    print("\n📋 REPOSITORY CONTEXT:")
    print("This is a Machine Learning educational repository based on")
    print("'Hands-On Machine Learning' by Aurélien Géron")
    print("\n✅ WHAT I CAN DO with this repository:")
    print("- Build ML models for price prediction")
    print("- Create recommendation systems")
    print("- Analyze product features and trends")
    print("- Develop data analysis pipelines")
    
    print("\n❌ WHAT I CANNOT DO:")
    print("- Access external shopping websites")
    print("- Scrape real-time product data")
    print("- Get live pricing or reviews")
    print("- Browse shopping portals for refrigerators")
    
    print("\n🔬 DEMONSTRATION WITH SAMPLE DATA:")
    print("Since I can't access external sites, I'll demonstrate ML capabilities")
    print("using realistic sample refrigerator data...")
    
    # Create sample data
    print("\n1. Creating sample refrigerator dataset...")
    df = create_sample_data()
    print(f"   ✓ Generated {len(df)} sample refrigerators")
    
    # Show data sample
    print("\n   Sample data:")
    print(df[['brand', 'type', 'capacity_liters', 'price', 'avg_rating']].head())
    
    # Build prediction model
    print("\n2. Building price prediction model...")
    model_results = build_price_prediction_model(df)
    print(f"   ✓ Model trained with RMSE: ${model_results['rmse']:.2f}")
    print(f"   ✓ R² Score: {model_results['r2']:.3f}")
    
    # Find best value products
    print("\n3. Finding best value refrigerators...")
    best_value = find_best_value_products(df)
    print("   ✓ Top 5 best value refrigerators:")
    print(best_value.to_string(index=False))
    
    # Demonstrate recommendation system
    print("\n4. Demonstrating recommendation system...")
    
    print("\n   Example 1: Budget under $1000, min 300L capacity:")
    rec1 = recommend_refrigerator(df, budget_max=1000, min_capacity=300)
    if isinstance(rec1, str):
        print(f"   {rec1}")
    else:
        print(rec1.to_string(index=False))
    
    print("\n   Example 2: Samsung brand, rating > 4.0:")
    rec2 = recommend_refrigerator(df, preferred_brand='Samsung', min_rating=4.0)
    if isinstance(rec2, str):
        print(f"   {rec2}")
    else:
        print(rec2.to_string(index=False))
    
    print("\n" + "="*80)
    print("SUMMARY - ANSWERING YOUR ORIGINAL QUESTION:")
    print("="*80)
    
    print("\n❓ Your question: 'Can you go to shopping portal and find best priced")
    print("   and reviewed refrigerator?'")
    
    print("\n💡 Answer: NO - I cannot access external shopping portals, but...")
    
    print("\n✅ What I CAN do instead:")
    print("   • Build ML models to predict prices based on features")
    print("   • Create recommendation systems for product comparison")  
    print("   • Analyze value propositions (price vs. rating)")
    print("   • Develop feature importance analysis")
    print("   • Create data visualization and insights")
    
    print("\n🎯 Real-world application:")
    print("   If you had product data (via APIs, datasets, or manual collection),")
    print("   you could use these EXACT ML techniques to build a comprehensive")
    print("   product comparison and recommendation system!")
    
    print("\n📚 This repository contains the ML building blocks for such systems.")
    print("   Check out the Jupyter notebooks for more advanced ML techniques!")

if __name__ == "__main__":
    main()