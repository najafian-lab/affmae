for perc in 0.05 0.25 0.50 0.75 1.0; do
       	echo "Starting training with data_perc: $perc"
	PYTHONPATH=. python finetune_perc_data.py --config configs/aff_small_finetune_cluster_0.5_mask.yaml --data_perc "$perc"
done
