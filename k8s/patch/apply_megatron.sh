set -x


\cp /home/code/verl-gpu/k8s/patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/multi_token_prediction.py /opt/Megatron-LM/megatron/core/transformer/multi_token_prediction.py
# \cp /home/code/verl-gpu/k8s/patch/megatron.patch/0.12.1/Megatron-LM/megatron/core/transformer/dot_product_attention.py /opt/Megatron-LM/megatron/core/transformer/dot_product_attention.py
echo -e "/033[32mApplied megatron MTP done./033[0m"

set +x
