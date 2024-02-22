#--------------------------------trainval set generation --------------------------------#
python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/trainval.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir {path_to}/WeakSAM/data/voc07/VOC2007/JPEGImages \
--attnmap coarse \
	--device cuda:0 \
	--attention-dir weaktr_results/VOC07/visible_results \
	--label-file-path voc07/cls_labels.npy \
--peak-file weaktr_results/VOC07-peak/coarse/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90 \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/trainval.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir {path_to}/WeakSAM/data/voc07/VOC2007/JPEGImages \
--attnmap fine \
	--device cuda:0 \
	--attention-dir weaktr_results/VOC07/visible_results \
	--label-file-path voc07/cls_labels.npy \
--peak-file weaktr_results/VOC07-peak/fine/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90 \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/trainval.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results/ \
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir {path_to}/WeakSAM/data/voc07/VOC2007/JPEGImages \
--attnmap cross \
	--device cuda:0 \
	--attention-dir weaktr_results/VOC07/visible_results \
	--label-file-path voc07/cls_labels.npy \
--peak-file weaktr_results/VOC07-peak/coarse/peak-pam-k129-t90 \
--point-type pam \
--kernel 129 \
--t 90 \

python clustering_generation.py --image-set voc07/trainval.txt \
--image-path {your_path}/WeakSAM/data/voc/VOC2007/JPEGImages \
--point-path ./weaktr_results/VOC07-peak/cross/peak-pam-k129-t90 \
--peakfile ./weaktr_results/VOC07-peak/cross/peak-pam-k129-t90-1

#--------------------------------trainval set generation --------------------------------#

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/test.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results\
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir {path_to}/WeakSAM/data/voc07/VOC2007/JPEGImages \
--device cuda:1 \
--attnmap fine \
	--attention-dir weaktr_results/VOC07/visible_results_test \
	--label-file-path voc07/cls_labels_test.npy \
--peak-file weaktr_results/VOC07-peak/fine/peak-pam-k17-t90_test \
--point-type pam \
--kernel 17 \
--t 90 \
--gen-test True \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/test.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results\
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir {path_to}/WeakSAM/data/voc07/VOC2007/JPEGImages \
--device cuda:1 \
--attnmap coarse \
	--attention-dir weaktr_results/VOC07/visible_results_test \
	--label-file-path voc07/cls_labels_test.npy \
--peak-file weaktr_results/VOC07-peak/coarse/peak-pam-k17-t90_test \
--point-type pam \
--kernel 17 \
--t 90 \
--gen-test True \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path data \
                --data-set VOC07MS \
                --img-ms-list voc07/test.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --output_dir weaktr_results\
                --resume weaktr_results/WeakTr_CAM_Generation.pth \
--ori-img-dir {path_to}/WeakSAM/data/voc07/VOC2007/JPEGImages \
--device cuda:1 \
--attnmap cross \
	--attention-dir weaktr_results/VOC07/visible_results_test \
	--label-file-path voc07/cls_labels_test.npy \
--peak-file weaktr_results/VOC07-peak/cross/peak-pam-k129-t90_test \
--point-type pam \
--kernel 129 \
--t 90 \
--gen-test True
