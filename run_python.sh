#!/bin/bash

# 启动 Python 脚本，并实时输出到 output.log
python -u main.py --model cnn > output00.log 2> error00.log &

# 实时监控 output.log 并输出到终端
tail -f output.log

# 等待 Python 脚本执行完成
wait

echo "Python 脚本执行完成"

# 输出错误信息
if [ -s error.log ]; then
    echo "错误信息如下："
    cat error.log
fi


# #!/bin/bash

# LOG_OUTPUT="output.log"
# LOG_ERROR="error.log"

# # 启动 Python 脚本，并实时输出到 output.log
# python -u main.py --model cnn > "$LOG_OUTPUT" 2> "$LOG_ERROR" &

# # 实时监控 output.log 并输出到终端
# tail -f "$LOG_OUTPUT"

# # 等待 Python 脚本执行完成
# wait

# echo "Python 脚本执行完成"

# # 输出错误信息
# if [ -s "$LOG_ERROR" ]; then
#     echo "错误信息如下："
#     cat "$LOG_ERROR"
# fi

# #!/bin/bash

# # 启动 Python 脚本，并实时输出到 output.log
# python -u main.py --model cnn > output.log 2> error.log &
# python_pid=$!

# # 实时监控 output.log 并输出到终端
# tail -f output.log &
# tail_pid=$!

# # 等待 Python 脚本执行完成
# wait $python_pid

# # 终止 tail -f 进程
# kill $tail_pid

# echo "Python 脚本执行完成"

# # 输出错误信息
# if [ -s error.log ]; then
#     echo "错误信息如下："
#     cat error.log
# fi

# # 检查是否有残留的 tail -f 进程
# if ps -p $tail_pid > /dev/null; then
#     echo "错误： tail -f 进程 ($tail_pid) 仍然在运行。"
# else
#     echo "没有残留的 tail -f 进程。"
# fi

# # 检查是否有残留的 Python 进程
# if ps -p $python_pid > /dev/null; then
#     echo "错误： Python 进程 ($python_pid) 仍然在运行。"
# else
#     echo "没有残留的 Python 进程。"
# fi