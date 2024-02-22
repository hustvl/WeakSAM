#------------------------------generation for trainval set ---------------------------------#
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12MS \
                --img-ms-list voc12/trainva.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir weaktr_results/VOC12/attn-patchrefine-npy-ms \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc12/VOCdevkit/VOC2012/JPEGImages \
--attnmap coarse \
	--device cuda:2 \
	--eval_miou_threshold_start 30 \
	--eval_miou_threshold_end 60 \
	--attention-dir weaktr_results/VOC12/visible_result \
--label-file-path voc12/cls_labels1.npy \
--peak-file weaktr_results/VOC12-peak-full/coarse/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90 \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12MS \
                --img-ms-list voc12/trainva.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir weaktr_results/VOC12/attn-patchrefine-npy-ms \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc12/VOCdevkit/VOC2012/JPEGImages \
--attnmap fine \
	--device cuda:2 \
	--eval_miou_threshold_start 30 \
	--eval_miou_threshold_end 60 \
	--attention-dir weaktr_results/VOC12/visible_result \
--label-file-path voc12/cls_labels1.npy \
--peak-file weaktr_results/VOC12-peak-full/fine/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90 \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12MS \
                --img-ms-list voc12/train_aug_id.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir weaktr_results/VOC12/attn-patchrefine-npy-ms \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc12/VOCdevkit/VOC2012/JPEGImages \
--attnmap cross \
	--device cuda:2 \
	--eval_miou_threshold_start 30 \
	--eval_miou_threshold_end 60 \
	--attention-dir weaktr_results/VOC12/visible_result \
--label-file-path voc12/cls_labels.npy \
--peak-file weaktr_results/VOC12-peak-aug/cross/peak-pam-k129-t90 \
--point-type pam \
--kernel 129 \
--t 90 \

#------------------------------generation for test set ---------------------------------#

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12MS \
                --img-ms-list voc12/testset.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir weaktr_results/VOC12/attn-patchrefine-npy-ms \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc12/VOCdevkit/VOC2012/JPEGImages \
--attnmap fine \
	--device cuda:2 \
	--eval_miou_threshold_start 30 \
	--eval_miou_threshold_end 60 \
	--attention-dir weaktr_results/VOC12/visible_result \
--peak-file weaktr_results/VOC12-peak-test/fine/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90 \
--gen-test True \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12MS \
                --img-ms-list voc12/testset.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir weaktr_results/VOC12/attn-patchrefine-npy-ms \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc12/VOCdevkit/VOC2012/JPEGImages \
--attnmap coarse \
	--device cuda:2 \
	--eval_miou_threshold_start 30 \
	--eval_miou_threshold_end 60 \
	--attention-dir weaktr_results/VOC12/visible_result \
--peak-file weaktr_results/VOC12-peak-test/coarse/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90 \
--gen-test True \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC12MS \
                --img-ms-list voc12/testset.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir weaktr_results/VOC12/attn-patchrefine-npy-ms \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir data/voc12/VOCdevkit/VOC2012/JPEGImages \
--attnmap cross \
	--device cuda:2 \
	--eval_miou_threshold_start 30 \
	--eval_miou_threshold_end 60 \
	--attention-dir weaktr_results/VOC12/visible_result \
--peak-file weaktr_results/VOC12-peak-test/cross/peak-pam-k129-t90 \
--point-type pam \
--kernel 129 \
--t 90 \
--gen-test True