build-sif:
	singularity build --fakeroot output/torch.sif torch.def

EXPORT_NAME = none
USE_AUG = none
run:
	singularity run --nv output/torch.sif python3 exp/tools/bert_train.py  --export_name=${EXPORT_NAME} --use_aug=${USE_AUG}
GPT_EXPORT_NAME = none
SUPERVISED_NUM = 1
run-gpt:
	singularity run --nv output/torch.sif python3 exp/tools/gpt_train.py ${GPT_EXPORT_NAME} ${SUPERVISED_NUM}
pip:
	singularity run output/torch.sif pip install -r requirements.txt

tensorboard:
	singularity exec output/torch.sif tensorboard --bind_all --logdir ./

slurm-run:
	sbatch run.sbatch

slurm-gpt-run:
	sbatch gpt.sbatch ${GPT_EXPORT_NAME} ${SUPERVISED_NUM}
local-run:
	python main.py

dataset_sync:
	ssh braun "rm -rf /home/student/e19/e195755/playground/exp/bert-fine-tuning/exp/data/bert_data"
	ssh braun "rm -rf /home/student/e19/e195755/playground/exp/bert-fine-tuning/exp/data/gpt_data"
	rsync -avhz ./exp/data/bert_data braun:/home/student/e19/e195755/playground/exp/bert-fine-tuning/exp/data/
	rsync -avhz ./exp/data/gpt_data braun:/home/student/e19/e195755/playground/exp/bert-fine-tuning/exp/data/

aug-and-train:
	singularity run --nv output/torch.sif python3 exp/tools/data_aug_and_train.py ${MODEL_DIR}
slurm-aug-and-train:
	sbatch aug-train.sbatch ${MODEL_DIR}

