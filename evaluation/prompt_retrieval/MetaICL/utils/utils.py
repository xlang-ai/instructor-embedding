# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess


all_settings = ["hr_to_lr", "hr_to_lr_noinst", "hr_to_lr_inst", "hr_to_lr_inst_all",
                "class_to_class", "non_class_to_class", "qa_to_qa", "non_qa_to_qa",
                "non_nli_to_nli", "non_paraphrase_to_paraphrase"]
all_methods = ["metaicl", "channel-metaicl", "multitask-zero", "channel-multitask-zero"]

checkpoint_dir = "https://dl.fbaipublicfiles.com/MetaICL"

def get_checkpoint_id(key):

    if key in all_methods:
        setting = "hr_to_lr"
        method = key
    elif key in [method + "-inst" for method in all_methods] or \
            key in [method + "-instruction" for method in all_methods]:
        setting = "hr_to_lr_inst_all"
        method = "-".join(key.split("-")[:-1])
    elif key in ["%s/%s" % (method, setting) for method in all_methods for setting in all_settings]:
        method, setting = key.split("/")
    else:
        return None
    return method, setting, os.path.join(checkpoint_dir, method, setting, "model.pt")

def download_file(_id, dest):
    if os.path.exists(dest):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if "/" in dest:
        dest_dir = "/".join(dest.split("/")[:-1])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    else:
        dest_dir = "."

    if _id.startswith("https://"):
        command = """wget -O %s %s""" % (dest, _id)
    else:
        command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt""" % (_id, _id, dest)

    ret_code = subprocess.run([command], shell=True)
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(dest))
    else:
        print("Download {} ... [Success]".format(dest))

    if dest.endswith(".zip"):
        command = """unzip %s -d %s && rm %s""" % (dest, dest_dir, dest)

        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Unzip {} ... [Failed]".format(dest))
        else:
            print("Unzip {} ... [Success]".format(dest))



