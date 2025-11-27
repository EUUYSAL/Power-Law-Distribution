import matplotlib.pyplot as plt
import numpy as np

def simulate_returns(
    n_paths=100_000,
    n_days=252,
    daily_vol=0.01,
    t_df=3,
    random_seed=42
):
    """
    Simulate daily returns under two scenarios:
    1) Gaussian (normal distribution)
    2) Fat-tail (Student-t distribution) as a proxy for power-law behavior

    Parameters:
        n_paths: Number of simulated price paths
        n_days: Number of days per path (≈ 1 trading year)
        daily_vol: Target daily volatility (e.g., 0.01 = 1%)
        t_df: Degrees of freedom for Student-t (lower df = fatter tails)
        random_seed: Seed for reproducibility
    """

    rng = np.random.default_rng(random_seed)

    # --- 1) Gaussian world ---
    gauss_returns = rng.normal(
        loc=0.0,
        scale=daily_vol,
        size=(n_paths, n_days)
    )

    # --- 2) Fat-tail world via Student-t ---
    # Raw Student-t samples
    t_raw = rng.standard_t(df=t_df, size=(n_paths, n_days))

    # Theoretical std deviation of Student-t (df > 2)
    t_std = np.sqrt(t_df / (t_df - 2))

    # Scale to match the same target daily volatility
    t_returns = (t_raw / t_std) * daily_vol

    return gauss_returns, t_returns


def daily_crash_stats(returns, thresholds=(-0.05, -0.08, -0.10)):
    """
    Compute probabilities of single-day crashes.

    thresholds:
        e.g. (-5%, -8%, -10%) represented as (-0.05, -0.08, -0.10)
    """
    flat = returns.reshape(-1)
    probs = {}

    for th in thresholds:
        probs[th] = np.mean(flat < th)

    return probs


def max_drawdown_stats(returns, dd_levels=(-0.30, -0.50)):
    """
    Compute maximum drawdown for each path.
    Drawdown = (price / peak) - 1

    dd_levels:
        Thresholds for extreme drawdowns (e.g. -30%, -50%)
    """

    # Convert returns to price paths (start at 1.0)
    prices = np.cumprod(1 + returns, axis=1)

    # Track rolling peak for each path
    running_peak = np.maximum.accumulate(prices, axis=1)

    # Compute drawdowns
    drawdowns = prices / running_peak - 1.0

    # Worst drawdown in each path
    max_dd_per_path = drawdowns.min(axis=1)

    # Probabilities for extreme drawdown levels
    probs = {}
    for lvl in dd_levels:
        probs[lvl] = np.mean(max_dd_per_path <= lvl)

    return probs, max_dd_per_path


# YENİ FONKSİYON - TARİHSEL KARŞILAŞTIRMA
def compare_to_history(gauss_ret, fat_ret):
    """Compare simulation results with historical crashes"""
    historical_crashes = {
        "1987 Black Monday": -0.20,
        "2008 Lehman Day": -0.09, 
        "2020 COVID crash": -0.12,
        "2025 Tariff shock": -0.105
    }
    
    print("\n--- HISTORICAL CRASH COMPARISON ---")
    for event, drop in historical_crashes.items():
        # Calculate probability of such daily drop
        gauss_prob = np.mean(gauss_ret.flatten() <= drop)
        fat_prob = np.mean(fat_ret.flatten() <= drop)
        
        # Convert to frequency
        gauss_freq = (1/gauss_prob/252) if gauss_prob > 0 else float('inf')
        fat_freq = (1/fat_prob/252) if fat_prob > 0 else float('inf')
        
        print(f"\n{event} (single day {drop*100:.1f}% drop):")
        print(f"  Gaussian world: Once every {gauss_freq:,.0f} years")
        print(f"  Fat-tail world: Once every {fat_freq:,.0f} years")


if __name__ == "__main__":
    # Simulation parameters
    N_PATHS = 100_000
    N_DAYS = 252
    DAILY_VOL = 0.01  # 1% daily volatility
    T_DF = 3          # degrees of freedom (fat-tail intensity)

    # Run simulations
    gauss_ret, fat_ret = simulate_returns(
        n_paths=N_PATHS,
        n_days=N_DAYS,
        daily_vol=DAILY_VOL,
        t_df=T_DF,
        random_seed=42
    )

    # Daily crash probability calculations
    thresholds = (-0.05, -0.08, -0.10)
    gauss_daily = daily_crash_stats(gauss_ret, thresholds)
    fat_daily = daily_crash_stats(fat_ret, thresholds)

    # Drawdown calculations
    dd_levels = (-0.30, -0.50)
    gauss_dd_probs, gauss_dd = max_drawdown_stats(gauss_ret, dd_levels)
    fat_dd_probs, fat_dd = max_drawdown_stats(fat_ret, dd_levels)

    # ---- Print results ----
    print("\n--- SIMULATION PARAMETERS ---")
    print(f"Paths: {N_PATHS}")
    print(f"Days per path: {N_DAYS}")
    print(f"Daily volatility: {DAILY_VOL*100:.2f}%")
    print(f"Student-t df (tail thickness): {T_DF}")

    print("\n--- DAILY CRASH PROBABILITIES ---")

    print("\nGaussian world:")
    for th, p in gauss_daily.items():
        freq = (1/p) if p > 0 else float('inf')
        print(f"  P(return < {th*100:.0f}%): {p:.8f} (~1/{freq:.0f} days)")

    print("\nFat-tail world (Student-t):")
    for th, p in fat_daily.items():
        freq = (1/p) if p > 0 else float('inf')
        print(f"  P(return < {th*100:.0f}%): {p:.8f} (~1/{freq:.0f} days)")

    print("\n--- YEARLY MAXIMUM DRAWDOWN PROBABILITIES ---")

    print("\nGaussian world:")
    for lvl, p in gauss_dd_probs.items():
        print(f"  P(MDD <= {lvl*100:.0f}%): {p:.4f}")

    print("\nFat-tail world (Student-t):")
    for lvl, p in fat_dd_probs.items():
        print(f"  P(MDD <= {lvl*100:.0f}%): {p:.4f}")

    print("\n--- AVERAGE MAXIMUM DRAWDOWN ---")
    print(f"Gaussian world: {gauss_dd.mean()*100:.2f}%")
    print(f"Fat-tail world: {fat_dd.mean()*100:.2f}%")

    # Call historical comparison 
    compare_to_history(gauss_ret, fat_ret)

    # Code corrected
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Simu
    ax.hist(gauss_dd, bins=50, alpha=0.5, label="Gaussian MDD", color='blue', density=True)
    ax.hist(fat_dd, bins=50, alpha=0.5, label="Fat-tail MDD", color='red', density=True)
    
    # Important points
    ax.axvline(-0.30, color='black', linestyle='--', linewidth=1, label='30% drawdown')
    ax.axvline(-0.50, color='black', linestyle=':', linewidth=1, label='50% drawdown')
    
    ax.set_title("Distribution of Maximum Drawdowns (1 Year)", fontsize=14)
    ax.set_xlabel("Maximum Drawdown (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # X percentafge
    ax.set_xlim(-0.7, 0)
    ax.set_xticks([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0])
    ax.set_xticklabels(['-70%', '-60%', '-50%', '-40%', '-30%', '-20%', '-10%', '0%'])
    
    plt.tight_layout()
    plt.show()