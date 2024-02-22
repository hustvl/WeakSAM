#------------------------------- trainval set ----------------------------------#
python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc07/trainval.txt \
--peakfile-name coarse/peak-pam-k17-t90/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/coarse \
--proposal-name k17_t90.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \


python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc07/trainval.txt \
--peakfile-name fine/peak-pam-k17-t90/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/fine \
--proposal-name k17_t90.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \


python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc07/trainval.txt \
--peakfile-name cross/peak-pam-k129-t90/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/cross \
--proposal-name k129_t90_wopost.pkl \
--device 0 \
--starting-layer 4 \
--ending-layer 12 \
--is-multi True \
--num-gpus 4 \
--n-processes 12 \


python post_suppression.py --target-prop /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/cross/k129_t90_wopost.pkl \
--threshold 0.9 \
--save-name k129_t90.pkl \
--save-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/cross \

python proposal_concatenation.py --pickle-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/ \
--storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/ \
--file-name voc07trainval.pkl \

#---------------------------------------test set------------------------------------------#

python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc07/test.txt \
--peakfile-name fine/peak-pam-k17-t90_test/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/fine \
--proposal-name k17_t90_test.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \

python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc07/test.txt \
--peakfile-name coarse/peak-pam-k17-t90_test/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/coarse \
--proposal-name k17_t90_test.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \

python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc07/test.txt \
--peakfile-name cross/peak-pam-k129-t90_test/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC07-peak \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc07/VOCdevkit/VOC2007/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/cross \
--proposal-name k129_t90_wopost_test.pkl \
--device 0 \
--starting-layer 4 \
--ending-layer 12 \
--is-multi True \
--num-gpus 4 \
--n-processes 12 \

python post_suppression.py --target-prop /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/cross/k129_t90_wopost.pkl \
--threshold 0.9 \
--save-name k129_t90_test.pkl \
--save-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/cross \

python proposal_concatenation.py --pickle-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/ \
--storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/ \
--file-name voc07test.pkl \
--istest True


