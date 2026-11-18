# A4 env on 104 (local disk). Usage: source /data/austin/nlp-a4/env.sh p1|p2
unset TRANSFORMERS_CACHE
export HF_HOME=/data/austin/.cache/huggingface
export NLTK_DATA=/data/austin/nltk_data
export UV_CACHE_DIR=/data/austin/.cache/uv
export PATH=$HOME/.local/bin:$PATH
case "$1" in
  p1) source /data/austin/venvs/nlp-a4-p1/bin/activate; cd /data/austin/nlp-a4/part1 ;;
  p2) source /data/austin/venvs/nlp-a4-p2/bin/activate; cd /data/austin/nlp-a4/part2 ;;
  *) echo "usage: source env.sh p1|p2" ;;
esac
