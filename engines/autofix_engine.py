# engines/autofix_engine.py
import pandas as pd

def autofix_missing(df, null_threshold=0.20):
    """
    Intelligently fill or filter missing values based on percentage threshold.
    
    Strategy:
    - If null% <= threshold (default 20%): Fill with mean (numeric) or "N/A" (text)
    - If null% > threshold: Drop rows with nulls in that column
    
    Args:
        df: DataFrame to process
        null_threshold: Decimal threshold (0.20 = 20%)
    """
    filled_cols = []
    dropped_cols = []
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            null_pct = null_count / len(df)
            
            if null_pct <= null_threshold:
                # Fill with mean (numeric) or "N/A" (text)
                if df[col].dtype != "object":
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    filled_cols.append({
                        "column": col,
                        "nulls": null_count,
                        "null_pct": f"{null_pct*100:.1f}%",
                        "action": f"Filled with mean ({mean_val:.2f})" if isinstance(mean_val, float) else f"Filled with mean"
                    })
                else:
                    df[col].fillna("N/A", inplace=True)
                    filled_cols.append({
                        "column": col,
                        "nulls": null_count,
                        "null_pct": f"{null_pct*100:.1f}%",
                        "action": "Filled with 'N/A'"
                    })
            else:
                # Drop rows where this column is null (nulls > threshold)
                rows_before = len(df)
                df = df.dropna(subset=[col])
                rows_after = len(df)
                rows_dropped = rows_before - rows_after
                
                dropped_cols.append({
                    "column": col,
                    "nulls": null_count,
                    "null_pct": f"{null_pct*100:.1f}%",
                    "action": f"Dropped {rows_dropped} rows",
                    "rows_before": rows_before,
                    "rows_after": rows_after
                })
    
    # Print summary
    if filled_cols:
        print(f"\nFilled missing values (threshold <= 20%):")
        for item in filled_cols:
            print(f"  • {item['column']}: {item['nulls']} nulls ({item['null_pct']}) -> {item['action']}")
    
    if dropped_cols:
        print(f"\nFiltered out rows with high nulls (threshold > 20%):")
        for item in dropped_cols:
            print(f"  • {item['column']}: {item['nulls']} nulls ({item['null_pct']}) -> {item['action']}")
    
    if not filled_cols and not dropped_cols:
        print("Data is clean - no missing values detected")
    
    return df, {
        "filled": filled_cols,
        "dropped": dropped_cols,
        "rows_removed": sum(item.get("rows_before", 0) - item.get("rows_after", 0) for item in dropped_cols)
    }

def autofix_duplicates(df):
    """Remove duplicate rows"""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    if before - after > 0:
        print(f"Removed {before - after} duplicate rows")
    return df

def autofix_constant(df):
    """Remove constant columns"""
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=const_cols)
    if const_cols:
        print(f"Dropped constant columns: {const_cols}")
    return df

def apply_autofix(df):
    """Apply all autofix steps and return summary"""
    summary = {}
    
    # Missing values handling
    df, missing_summary = autofix_missing(df)
    summary["missing"] = missing_summary
    
    # Duplicates handling
    df = autofix_duplicates(df)
    
    # Constant columns handling
    df = autofix_constant(df)
    
    print("\nAutofix applied to dataset")
    print(f"Final dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    
    return df, summary
