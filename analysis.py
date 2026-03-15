import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# Define YOUR dataset's numeric columns (from our cleaning)
NUMERIC_COLUMNS = [
    'Total Lang1', 'Total Lang2', 'Total Math', 'Total PHY', 
    'Total CHE', 'Total BIO/other', 'Total CS/IT',
    '12th_Pct', '10th_Pct', 'STEM_Total', 'STEM_Avg', 'Overall_Avg'
]

def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported input file type. Use CSV or Excel.")

def ensure_analysis_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are properly typed"""
    available_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    if len(available_cols) > 0:
        out = df.copy()
        for col in available_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
    raise ValueError("No recognized numeric columns found!")

def save_descriptive_stats(df: pd.DataFrame, out_dir: Path) -> None:
    available_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    stats = df[available_cols].describe().T
    stats["missing_count"] = df[available_cols].isna().sum()
    stats.to_csv(out_dir / "descriptive_stats.csv")
    print(f"📊 Stats saved: {len(available_cols)} columns analyzed")

def plot_missing_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    available_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[available_cols].isna(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title("Missing Value Pattern - Academic Marks Dataset")
    plt.xlabel("Subjects & Percentages")
    plt.ylabel("Students")
    plt.tight_layout()
    plt.savefig(out_dir / "missing_pattern_heatmap.png", dpi=180, bbox_inches='tight')
    plt.close()

def plot_distribution_grid(df: pd.DataFrame, out_dir: Path) -> None:
    available_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    cols = 3
    rows = math.ceil(len(available_cols) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes]

    for idx, col in enumerate(available_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], bins=20)
        axes[idx].set_title(f"{col}")
        axes[idx].set_xlabel("Marks (%)")

    for idx in range(len(available_cols), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Distribution of Academic Performance", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / "numeric_distributions.png", dpi=180, bbox_inches='tight')
    plt.close(fig)

def plot_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    available_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    corr = df[available_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, 
                center=0, cbar_kws={'label': 'Correlation'})
    plt.title("Subject Correlation Heatmap\n(Higher = subjects taken together)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=180, bbox_inches='tight')
    plt.close()

def plot_10th_vs_12th(df: pd.DataFrame, out_dir: Path) -> None:
    if '10th_Pct' in df.columns and '12th_Pct' in df.columns:
        plt.figure(figsize=(10, 8))
        sns.regplot(data=df, x='10th_Pct', y='12th_Pct', scatter_kws={"alpha": 0.6},
                   line_kws={"color": "red", "lw": 2})
        plt.title("10th vs 12th Percentage\n(Trend: Consistent performers rise together)", fontsize=14)
        plt.xlabel("10th Percentage")
        plt.ylabel("12th Percentage")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "10th_vs_12th_regplot.png", dpi=180, bbox_inches='tight')
        plt.close()

def plot_stem_performance(df: pd.DataFrame, out_dir: Path) -> None:
    """Custom plot for STEM subjects"""
    stem_cols = ['Total Math', 'Total PHY', 'Total CHE', 'Total CS/IT']
    stem_cols = [col for col in stem_cols if col in df.columns]
    
    if stem_cols:
        plt.figure(figsize=(10, 6))
        df_stem = df[stem_cols].melt(var_name='Subject', value_name='Marks')
        sns.boxplot(data=df_stem, x='Subject', y='Marks')
        plt.title("STEM Subjects Performance Distribution")
        plt.ylabel("Marks (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "stem_boxplot.png", dpi=180, bbox_inches='tight')
        plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="🎓 Academic Dataset Analysis for Career Recommendation")
    parser.add_argument("--input", default="cleaned_career_data.csv", 
                       help="Path to cleaned CSV/Excel (default: cleaned_career_data.csv)")
    parser.add_argument("--output-dir", default="analysis_outputs", 
                       help="Output folder for plots & stats")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"❌ Input not found: {in_path}\n"
                               f"Run cleaning script first to generate cleaned_career_data.csv")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔄 Loading dataset...")
    raw = load_input(in_path)
    df = ensure_analysis_ready(raw)

    print("📈 Generating analysis...")
    save_descriptive_stats(df, out_dir)
    plot_missing_heatmap(df, out_dir)
    plot_distribution_grid(df, out_dir)
    plot_correlation(df, out_dir)
    plot_10th_vs_12th(df, out_dir)
    plot_stem_performance(df, out_dir)

    print(f"\n✅ Analysis complete!")
    print(f"📁 Outputs saved in: {out_dir.absolute()}")
    print(f"📊 Files: descriptive_stats.csv, 6 plots (PNG)")

if __name__ == "__main__":
    main()
