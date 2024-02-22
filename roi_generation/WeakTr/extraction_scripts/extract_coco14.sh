# python main.py --model deit_small_WeakTr_patch16_224 \
#                 --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
#                 --data-set COCOMS \
#                 --img-ms-list coco/val_id.txt \
#                 --scales 1.0 0.8 1.2 \
#                 --gen_attention_maps \
#                 --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
#                 --output_dir weaktr_results_coco \
#                 --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
# --ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
# --attnmap cross \
# --starting 4 \
# --device cuda:1 \
# 	--eval_miou_threshold_start 30 \
# 	--eval_miou_threshold_end 60 \
# 	--attention-dir weaktr_results_coco/visible_results \
# 	--label-file-path coco/label_cls_val.npy \
# --peak-file weaktr_results_coco/COCO-peak/cross/peak-pam-k129-t90-val-l5-6 \
# --point-type pam \
# --kernel 129 \
# --t 90 \

# python main.py --model deit_small_WeakTr_patch16_224 \
#                 --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
#                 --data-set COCOMS \
#                 --img-ms-list coco/val_id.txt \
#                 --scales 1.0 0.8 1.2 \
#                 --gen_attention_maps \
#                 --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
#                 --output_dir weaktr_results_coco \
#                 --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
# --ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
# --attnmap cross \
# --starting 6 \
# --device cuda:1 \
# 	--eval_miou_threshold_start 30 \
# 	--eval_miou_threshold_end 60 \
# 	--attention-dir weaktr_results_coco/visible_results \
# 	--label-file-path coco/label_cls_val.npy \
# --peak-file weaktr_results_coco/COCO-peak/cross/peak-pam-k129-t90-val-l7-8 \
# --point-type pam \
# --kernel 129 \
# --t 90 \

# python main.py --model deit_small_WeakTr_patch16_224 \
#                 --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
#                 --data-set COCOMS \
#                 --img-ms-list coco/val_id.txt \
#                 --scales 1.0 0.8 1.2 \
#                 --gen_attention_maps \
#                 --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
#                 --output_dir weaktr_results_coco \
#                 --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
# --ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
# --attnmap cross \
# --starting 8 \
# --device cuda:1 \
# 	--eval_miou_threshold_start 30 \
# 	--eval_miou_threshold_end 60 \
# 	--attention-dir weaktr_results_coco/visible_results \
# 	--label-file-path coco/label_cls_val.npy \
# --peak-file weaktr_results_coco/COCO-peak/cross/peak-pam-k129-t90-val-l9-10 \
# --point-type pam \
# --kernel 129 \
# --t 90 \

# python main.py --model deit_small_WeakTr_patch16_224 \
#                 --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
#                 --data-set COCOMS \
#                 --img-ms-list coco/val_id.txt \
#                 --scales 1.0 0.8 1.2 \
#                 --gen_attention_maps \
#                 --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
#                 --output_dir weaktr_results_coco \
#                 --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
# --ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
# --attnmap cross \
# --starting 10 \
# --device cuda:1 \
# 	--eval_miou_threshold_start 30 \
# 	--eval_miou_threshold_end 60 \
# 	--attention-dir weaktr_results_coco/visible_results \
# 	--label-file-path coco/label_cls_val.npy \
# --peak-file weaktr_results_coco/COCO-peak/cross/peak-pam-k129-t90-val-l11-12 \
# --point-type pam \
# --kernel 129 \
# --t 90 \

# python main.py --model deit_small_WeakTr_patch16_224 \
#                 --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/train2014 \
#                 --data-set COCOMS \
#                 --img-ms-list coco/train_id.txt \
#                 --scales 1.0 0.8 1.2 \
#                 --gen_attention_maps \
#                 --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
#                 --output_dir weaktr_results_coco \
#                 --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
# --ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/train2014 \
# --attnmap coarse \
# --device cuda:1 \
# 	--eval_miou_threshold_start 30 \
# 	--eval_miou_threshold_end 60 \
# 	--attention-dir weaktr_results_coco/visible_results \
# 	--label-file-path coco/label_cls.npy \
# --peak-file weaktr_results_coco/COCO-peak/coarse/peak-pam-k17-t90 \
# --point-type pam \
# --kernel 17 \
# --t 90 \

python main.py --model deit_small_WeakTr_patch16_224 \
                --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/train2014 \
                --data-set COCOMS \
                --img-ms-list coco/train_id.txt \
                --scales 1.0 0.8 1.2 \
                --gen_attention_maps \
                --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
                --output_dir weaktr_results_coco \
                --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
--ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/train2014 \
--attnmap fine \
--device cuda:1 \
	--eval_miou_threshold_start 30 \
	--eval_miou_threshold_end 60 \
	--attention-dir weaktr_results_coco/visible_results \
	--label-file-path coco/label_cls.npy \
--peak-file weaktr_results_coco/COCO-peak/fine/peak-pam-k17-t90 \
--point-type pam \
--kernel 17 \
--t 90 \

# python main.py --model deit_small_WeakTr_patch16_224 \
#                 --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/train2014 \
#                 --data-set COCOMS \
#                 --img-ms-list coco/train_id.txt \
#                 --scales 1.0 0.8 1.2 \
#                 --gen_attention_maps \
#                 --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
#                 --output_dir weaktr_results_coco \
#                 --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
# --ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/train2014 \
# --attnmap coarse \
# --device cuda:1 \
# 	--eval_miou_threshold_start 30 \
# 	--eval_miou_threshold_end 60 \
# 	--attention-dir weaktr_results_coco/visible_results \
# 	--label-file-path coco/label_cls.npy \
# --peak-file weaktr_results_coco/COCO-peak/coarse/peak-pam-k17-t90 \
# --point-type pam \
# --kernel 17 \
# --t 90 \

# python main.py --model deit_small_WeakTr_patch16_224 \
#                 --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
#                 --data-set COCOMS \
#                 --img-ms-list coco/val_id.txt \
#                 --scales 1.0 0.8 1.2 \
#                 --gen_attention_maps \
#                 --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
#                 --output_dir weaktr_results_coco \
#                 --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
# --ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
# --attnmap fine \
# --device cuda:1 \
# 	--eval_miou_threshold_start 30 \
# 	--eval_miou_threshold_end 60 \
# 	--attention-dir weaktr_results_coco/visible_results-val \
# 	--label-file-path coco/label_cls_val.npy \
# --peak-file weaktr_results_coco/COCO-peak/fine/peak-pam-k17-t90-val \
# --point-type pam \
# --kernel 17 \
# --t 90 \

# python main.py --model deit_small_WeakTr_patch16_224 \
#                 --data-path /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
#                 --data-set COCOMS \
#                 --img-ms-list coco/val_id.txt \
#                 --scales 1.0 0.8 1.2 \
#                 --gen_attention_maps \
#                 --cam-npy-dir weaktr_results_coco/attn-patchrefine-npy-ms \
#                 --output_dir weaktr_results_coco \
#                 --resume weaktr_results/WeakTr_CAM_Generation_COCO.pth \
# --ori-img-dir /home/junweizhou/WeakSAM/WSOD2/data/coco/coco2014/val2014 \
# --attnmap coarse \
# --device cuda:1 \
# 	--eval_miou_threshold_start 30 \
# 	--eval_miou_threshold_end 60 \
# 	--attention-dir weaktr_results_coco/visible_results-val \
# 	--label-file-path coco/label_cls_val.npy \
# --peak-file weaktr_results_coco/COCO-peak/coarse/peak-pam-k17-t90-val \
# --point-type pam \
# --kernel 17 \
# --t 90