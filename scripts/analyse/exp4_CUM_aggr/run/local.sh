

SAVE_DIR="results/aggr_results"

python scripts/analyse/exp4_CUM_aggr/plot_aggr.py \
    "from_niflheim/Atom/name_Atom/results/discovery_plots/cumulative_aggregated.npz" \
    "from_niflheim/ai4sc/NIPS-AV-R/relax_steps_final_0/results/discovery_plots/cumulative_aggregated.npz" \
    "from_niflheim/ai4sc/NIPS-AV-R/relax_steps_final_3/results/discovery_plots/cumulative_aggregated.npz" \
    "from_niflheim/ai4sc/NIPS-AV-R/relax_steps_final_6/results/discovery_plots/cumulative_aggregated.npz" \
    "from_niflheim/ai4sc/NIPS-F/name_NIPS-F/results/discovery_plots/cumulative_aggregated.npz" \
    "from_niflheim/ai4sc/NIPS-FV/name_NIPS-FV/results/discovery_plots/cumulative_aggregated.npz" \
    --save_dir $SAVE_DIR
