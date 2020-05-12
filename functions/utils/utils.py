import os
import shutil
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
def copyfile(src, dst):
    shutil.copyfile(src, dst)

def backup_code(LOGDIR):
    os.mkdir(LOGDIR + '/code')
    os.mkdir(LOGDIR + '/code/functions')
    copytree('./functions', LOGDIR + '/code/functions')
    copyfile('./run_synthesizing.py', LOGDIR + '/code/run_synthesizing.py')
    copyfile('./run.py', LOGDIR + '/code/run.py')
    copyfile('./run_mri.py', LOGDIR + '/code/run_mri.py')
    copyfile('./run_ssim_perf_angio.py', LOGDIR + '/code/run_ssim_perf_angio.py')
