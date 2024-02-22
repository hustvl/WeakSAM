python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc12/testset.txt \
--peakfile-name coarse/peak-pam-k17-t90/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC12-peak-test \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc12/VOCdevkit/VOC2012/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC12-test/coarse \
--proposal-name k17_t90.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \

python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc12/testset.txt \
--peakfile-name fine/peak-pam-k17-t90/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC12-peak-test \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc12/VOCdevkit/VOC2012/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC12-test/fine \
--proposal-name k17_t90.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \

python pooled_proposal_generation.py --img-set-path /home/junweizhou/WeakTr/WeakTr/voc12/testset.txt \
--peakfile-name cross/peak-pam-k129-t90-1/ \
--txt-folder-path /home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC12-peak-test \
--ori-image-path /home/junweizhou/WeakTr/WeakTr/data/voc12/VOCdevkit/VOC2012/JPEGImages \
--proposal-storage-path /home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC12-test/cross \
--proposal-name k129_t90_wopost.pkl \
--device 0 \
--starting-layer 4 \
--ending-layer 12 \
--is-multi True \
--num-gpus 4 \
--n-processes 12 \
