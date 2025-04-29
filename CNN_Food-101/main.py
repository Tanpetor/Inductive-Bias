from experiments import run_all_experiments
from visualization import plot_losses, plot_metrics, print_best_results

if __name__ == "__main__":
    all_results = run_all_experiments()
    plot_losses(all_results)
    plot_metrics(all_results)
    print_best_results(all_results)