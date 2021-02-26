
#run_SimBERT.sh

#conda activate data_aug
export CUDA_VISIBLE_DEVICES=3
python /data/public/wanghao/code/nlpcda/run_SimBERT.py --input_filepath=/data/public/wanghao/data/xqb_die_for_augu_test_wh/xqb_die_merge_3lines_for_augument.txt --output_filepath=/data/public/wanghao/data/xqb_die_for_augu_test_wh/xqb_die_merge_3lines_for_augument_SimBERTaugumented.txt

