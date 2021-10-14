POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    --default)
      DEFAULT=YES
      shift # past argument
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

if [ -z $MODEL ]; then
    MODEL=rinna/japanese-gpt2-medium
fi
echo $MODEL
python $HOME/src/github.com/huggingface/transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path=$MODEL \
    --train_file=data/gpt2_train.txt \
    --validation_file=data/gpt2_train.txt \
    --do_train \
    --do_eval \
    --num_train_epochs=30 \
    --save_steps=10000 \
    --save_total_limit=3 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --output_dir=out/ \
    --use_fast_tokenizer=False
