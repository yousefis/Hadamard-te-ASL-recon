from threading import Lock
def init():
    global  mutex,mutex2
    global  patch_count
    global train_queue, read_patche_mutex_tr,read_patche_mutex_vl,tr_isread,vl_isread
    global queue_isready_vl, validation_totalimg_patch, validation_patch_reuse, read_vl_offline, read_off_finished
    global subjects_tr_crush,  subjects_vl_crush,    subjects_tr2_crush,    subjects_vl2_crush
    global subjects_vl_noncrush,  subjects_vl2_noncrush, subjects_tr2_noncrush, subjects_tr_noncrush
    global   subjects_vl_angio, subjects_tr_angio,  subjects_tr2_angio , subjects_vl2_angio
    global subjects_tr2_perf,  subjects_vl_perf, subjects_vl2_perf,  subjects_tr_perf
    global subjects_tr_mri, subjects_vl_mri, subjects_tr2_mri, subjects_vl2_mri
    global subjects_tr2_segmentation, subjects_vl2_segmentation, subjects_tr_segmentation, subjects_vl_segmentation
    queue_isready_vl = False



    validation_totalimg_patch = 0
    read_vl_offline = False
    read_off_finished = False

    patch_count=0
    mutex= Lock()
    mutex2= Lock()
    read_patche_mutex_tr= Lock()
    read_patche_mutex_vl= Lock()
    train_queue=Lock()

    tr_isread=True
    vl_isread = True


    subjects_tr_crush=[]
    subjects_tr_noncrush=[]
    subjects_tr_perf=[]
    subjects_tr_angio=[]
    subjects_vl_crush=[]
    subjects_vl_noncrush=[]
    subjects_vl_perf=[]
    subjects_vl_angio=[]
    subjects_tr2_noncrush=[]
    subjects_tr2_crush=[]
    subjects_tr2_perf=[]
    subjects_tr2_angio=[]
    subjects_vl2_crush=[]
    subjects_vl2_noncrush=[]
    subjects_vl2_perf=[]
    subjects_vl2_angio=[]

    subjects_tr_mri=[]
    subjects_vl_mri=[]
    subjects_tr2_mri=[]
    subjects_vl2_mri=[]

    subjects_tr_segmentation = []
    subjects_vl_segmentation = []
    subjects_tr2_segmentation = []
    subjects_vl2_segmentation = []

    validation_patch_reuse = []