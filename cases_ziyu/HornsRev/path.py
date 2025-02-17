import sys
import os
# 使用 ~ 表示 $HOME 目录
home_path = os.path.expanduser("~")
sys.path.append(home_path+'/solvers/floris/cases_ziyu')