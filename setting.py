import socket
def set_path():
    if(socket.gethostname()) == 'ASCAST-NC304375':
        path_obs="/Users/wang.12220/OneDrive - The Ohio State University/BuckeyeBox Data/Home/Home/2024/10/Gl229B"
        path_data="/Users/wang.12220/OneDrive - The Ohio State University/BuckeyeBox Data/Home/Home/2024/10/Gl229B/database"
        path_repo="/home/kawashima/exojax/github"
    elif(socket.gethostname()) == 'ASCAST-NC304632':
        path_obs="/Users/wang.12220/OneDrive - The Ohio State University/BuckeyeBox Data/Home/Home/2024/10/Gl229B"
        path_data="/Users/wang.12220/OneDrive - The Ohio State University/BuckeyeBox Data/Home/Home/2024/10/Gl229B/database"
        path_repo="/home/kawashima/exojax/github"
    else:
        path_obs="/fs/scratch/PAS2558/hr7672"
        path_data="/fs/scratch/PAS2558/hr7672/database"
        path_repo="/home/kawashimayi/exojax/github"

    return path_obs, path_data, path_repo

# /Users/wang.12220/OneDrive - The Ohio State University/BuckeyeBox Data/Home/Home/2024/10/Gl229B

import git, datetime
import os
import subprocess
def git_install(path_repo, branch):
    repo = git.Repo(path_repo)
    repo.git.checkout(branch)

    print('******************************')
    print("VERSION INFORMATION")
    print("branch:", repo.active_branch.name)
    for item in repo.iter_commits(repo.active_branch.name, max_count=1):
        dt = datetime.datetime.fromtimestamp(item.authored_date).strftime("%Y-%m-%d %H:%M:%S")
        print("commit:", item.hexsha)
        print("Author:", item.author)
        print("Date:", dt)
        print(item.message)
    print('******************************')

    calc_dir = os.getcwd()
    os.chdir(path_repo)

    log = subprocess.check_output(["python", "setup.py", "install"]).decode()
    if "Finished" in log:
        print("Successfully installed")
    else:
        print("Fail to install exojax")

    os.chdir(calc_dir)
