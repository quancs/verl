####################################################
# 本脚本用于，在多机训练时：
# 1. 向其他节点，同步主节点修改后的数据
# 2. 多机启动docker、启动训练脚本
####################################################
# 本脚本需要提前配置好ssh免密登录，即将master的公钥传到目标服务器，指令示例如下：
# ssh-keygen
# ssh-copy-id root@90.90.122.120
####################################################
# 脚本执行方式：
# chmod +x main.sh
# ./main.sh
####################################################

IPs=( # 除开master的其他节点IP
    # "90.90.122.117"
    "90.91.103.33"
)

# 需要同步的目录。目标目录与源目录不相同的内容会被删除
DIRs=(
    "/data/q00887491/models/Moonlight-16B-A3B"
    # "/data/q00887491/models/Moonlight-16B-A3B-Instruct"
    # "/home/q00887491/models/Moonlight-16B-A3B-Instruct-dist-pp4"
    "/data/q00887491/datasets"
    "/data/q00887491/projects/wlf_darkmatter_verl"
    # "/data/q00887491/logs/fsdp-dump"
)

# 映射到docker内的路径
# docker_v_dirs="-v ${DIRs[0]}:/data/models/Moonlight-16B-A3B-Instruct:ro -v ${DIRs[1]}:/data/models/Moonlight-16B-A3B-Instruct-dist -v ${DIRs[2]}:/data/datasets/gsm8k:ro -v ${DIRs[3]}:/workspace/verl -v /home/q00887491/logs:/data/logs"

docker_v_dirs="-v ${DIRs[0]}:/data/models/Moonlight-16B-A3B:ro -v ${DIRs[1]}:/data/datasets:ro -v ${DIRs[2]}:/workspace/verl -v /data/q00887491/logs:/data/logs"
user_used=q00887491

# 启动脚本在docker内的路径
docker_cmd_file="/workspace/verl/recipe/moonlight_GPU_2J/start.sh"


echo "#################  数据同步中  ####################"
index=0
for ip in ${IPs[*]}; do
    echo -e "rsync to Node $index $ip"

    for diri in ${DIRs[*]}; do
        ssh $user_used@$ip "mkdir -p $diri; exit" # 创建目录
        echo -e "rsync -av --delete $diri/ $user_used@$ip:$diri/"
        rsync -av --delete $diri/ $user_used@$ip:$diri/  # 同步数据，会删除目标目录中不一样的内容
        echo -e "\n"
    done

    ((index++))  # 索引自增
    echo -r "\n\n"
done
sleep 1
echo "#################  数据同步结束  ####################"

echo "#################  训练启动中  ####################"
# docker启动
DOCKER_IMAGES_ID=verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
CONTAINER_NAME=verl_dx

DOCKER_START_CMD="docker create --runtime=nvidia \
    --privileged=true \
    --gpus all \
    --net=host \
    --shm-size="10g" \
    --cap-add=SYS_ADMIN \
    ${docker_v_dirs} \
    -v /usr/sbin/ifconfig:/usr/sbin/ifconfig:ro \
    --name ${CONTAINER_NAME} \
    ${DOCKER_IMAGES_ID} \
    sleep infinity"

DOCKER_RUN_CMD="docker start ${CONTAINER_NAME} && docker exec ${CONTAINER_NAME} bash ${docker_cmd_file}"

echo -e "$DOCKER_START_CMD"
echo -e "$DOCKER_RUN_CMD"

echo -e "\nstart process on Master Node"
docker stop $CONTAINER_NAME; docker rm $CONTAINER_NAME # 停止+删除容器
eval $DOCKER_START_CMD # 创建容器
eval $DOCKER_RUN_CMD &
sleep 1

index=0
for ip in ${IPs[*]}; do
    echo -e "\nstart process on Node $index $ip"
    ssh $user_used@$ip "docker stop $CONTAINER_NAME; docker rm $CONTAINER_NAME" # 停止+删除容器
    ssh $user_used@$ip "$DOCKER_START_CMD" # 创建容器
    ssh $user_used@$ip "$DOCKER_RUN_CMD; exit" & # +“&”后，后台起另外一个线程运行
    ((index++))  # 索引自增
    echo "################################################"
done

# 等待后台任务完成
wait

echo "#################  训练结束  ####################"

