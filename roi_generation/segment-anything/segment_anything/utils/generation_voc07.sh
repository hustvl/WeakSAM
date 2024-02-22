#------------------------------- trainval set ----------------------------------#
python ./segment-anything/segment_anything/utils/pooled_proposal_generation.py --img-set-path ./WeakTr/voc07/trainval.txt \
--peakfile-name coarse/peak-pam-k17-t90/ \
--txt-folder-path ./WeakTr/weaktr_results/VOC07-peak \
--ori-image-path ../data/voc07/VOC2007/JPEGImages \
--proposal-storage-path ./segment-anything/peak_proposal/VOC07/coarse \
--proposal-name k17_t90.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \


python ./segment-anything/segment_anything/utils/pooled_proposal_generation.py --img-set-path ./WeakTr/voc07/trainval.txt \
--peakfile-name fine/peak-pam-k17-t90/ \
--txt-folder-path ./WeakTr/weaktr_results/VOC07-peak \
--ori-image-path ../data/voc07/VOC2007/JPEGImages \
--proposal-storage-path ./segment-anything/peak_proposal/VOC07/fine \
--proposal-name k17_t90.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \


python ./segment-anything/segment_anything/utils/pooled_proposal_generation.py --img-set-path ./WeakTr/voc07/trainval.txt \
--peakfile-name cross/peak-pam-k129-t90/ \
--txt-folder-path ./WeakTr/weaktr_results/VOC07-peak \
--ori-image-path ../data/voc07/VOC2007/JPEGImages \
--proposal-storage-path ./segment-anything/peak_proposal/VOC07/cross \
--proposal-name k129_t90_wopost.pkl \
--device 0 \
--starting-layer 4 \
--ending-layer 12 \
--is-multi True \
--num-gpus 4 \
--n-processes 12 \


python ./segment-anything/segment_anything/utils/post_suppression.py --target-prop ./segment-anything/peak_proposal/VOC07/cross/k129_t90_wopost.pkl \
--threshold 0.9 \
--save-name k129_t90.pkl \
--save-path ./segment-anything/peak_proposal/VOC07/cross \

python ./segment-anything/segment_anything/utils/proposal_concatenation.py --pickle-path ./segment-anything/peak_proposal/VOC07/ \
--storage-path ./segment-anything/peak_proposal/VOC07/ \
--file-name voc07trainval.pkl \

#---------------------------------------test set------------------------------------------#

python ./segment-anything/segment_anything/utils/pooled_proposal_generation.py --img-set-path ./WeakTr/voc07/test.txt \
--peakfile-name fine/peak-pam-k17-t90_test/ \
--txt-folder-path ./WeakTr/weaktr_results/VOC07-peak \
--ori-image-path ../data/voc07/VOC2007/JPEGImages \
--proposal-storage-path ./segment-anything/peak_proposal/VOC07/fine \
--proposal-name k17_t90_test.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \

python ./segment-anything/segment_anything/utils/pooled_proposal_generation.py --img-set-path ./WeakTr/voc07/test.txt \
--peakfile-name coarse/peak-pam-k17-t90_test/ \
--txt-folder-path ./WeakTr/weaktr_results/VOC07-peak \
--ori-image-path ../data/voc07/VOC2007/JPEGImages \
--proposal-storage-path ./segment-anything/peak_proposal/VOC07/coarse \
--proposal-name k17_t90_test.pkl \
--device 0 \
--starting-layer 1 \
--ending-layer 12 \
--is-multi False \
--num-gpus 4 \
--n-processes 12 \
--is-cls-aware True \

python ./segment-anything/segment_anything/utils/pooled_proposal_generation.py --img-set-path ./WeakTr/voc07/test.txt \
--peakfile-name cross/peak-pam-k129-t90_test/ \
--txt-folder-path ./WeakTr/weaktr_results/VOC07-peak \
--ori-image-path ../data/voc07/VOC2007/JPEGImages \
--proposal-storage-path ./segment-anything/peak_proposal/VOC07/cross \
--proposal-name k129_t90_wopost_test.pkl \
--device 0 \
--starting-layer 4 \
--ending-layer 12 \
--is-multi True \
--num-gpus 4 \
--n-processes 12 \

python ./segment-anything/segment_anything/utils/post_suppression.py --target-prop ./segment-anything/peak_proposal/VOC07/cross/k129_t90_wopost.pkl \
--threshold 0.9 \
--save-name k129_t90_test.pkl \
--save-path ./segment-anything/peak_proposal/VOC07/cross \

python ./segment-anything/segment_anything/utils/proposal_concatenation.py --pickle-path ./segment-anything/peak_proposal/VOC07/ \
--storage-path ./segment-anything/peak_proposal/VOC07/ \
--file-name voc07test.pkl \
--istest True


