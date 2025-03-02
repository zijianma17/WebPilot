"""
auto login
"""

import os
import subprocess

def cookies_init(set_cookies=False):
    
    if not set_cookies:
        print("Skip saving account cookies")
        return
    
    # create folder
    subprocess.run("mkdir -p ./.auth", shell=True,)

    # save cookies
    subprocess.run("python ./webarena-main/browser_env/auto_login.py", shell=True,)

    print("Done saving account cookies")

    # # run bash prepare.sh to save all account cookies, this only needs to be done once
    # subprocess.run(["bash", "prepare.sh"])
    # print("Done saving account cookies")
