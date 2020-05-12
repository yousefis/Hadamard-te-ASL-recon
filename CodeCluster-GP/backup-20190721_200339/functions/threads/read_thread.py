import threading
import time

import numpy as np

import functions.utils.settings as settings


class read_thread(threading.Thread):
    def __init__ (self,_fill_thread,mutex,validation_sample_no=0,is_training=1):
        threading.Thread.__init__(self)
        self._fill_thread=_fill_thread
        self.mutex=mutex

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.is_training=is_training
        self.validation_sample_no=validation_sample_no
    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                # print('try aquired2')
                # self.mutex.acquire()
                # print('aquired2')
                try:
                    # while len(settings.bunch_GTV_patches) > 300:
                    #     print('sleep bunch_GTV_patches:%d', len(settings.bunch_GTV_patches))
                    #     time.sleep(3)
                    # thread should do the thing if not paused
                    if self.is_training==1:
                        if len(settings.subjects_tr_segmentation)==0 :
                            # print(len(settings.subjects_tr_angio))
                            settings.train_queue.acquire()
                            settings.subjects_tr_crush= settings.subjects_tr2_crush.copy()
                            settings.subjects_tr_noncrush= settings.subjects_tr2_noncrush.copy()
                            settings.subjects_tr_perf= settings.subjects_tr2_perf.copy()
                            settings.subjects_tr_angio= settings.subjects_tr2_angio.copy()
                            settings.subjects_tr_mri= settings.subjects_tr2_mri.copy()
                            settings.subjects_tr_segmentation= settings.subjects_tr2_segmentation.copy()
                            settings.subjects_tr2_crush = []
                            settings.subjects_tr2_noncrush = []
                            settings.subjects_tr2_perf = []
                            settings.subjects_tr2_angio = []
                            settings.subjects_tr2_mri= []
                            settings.subjects_tr2_segmentation = []
                            settings.train_queue.release()
                            self._fill_thread.resume()
                            #time.sleep(2)
                        else:
                            if len(settings.subjects_tr2_segmentation) and len(settings.subjects_tr2_segmentation):
                                settings.train_queue.acquire()
                                settings.subjects_tr_crush = np.vstack((settings.subjects_tr_crush,
                                                                        settings.subjects_tr2_crush))
                                settings.subjects_tr_noncrush = np.vstack((settings.subjects_tr_noncrush,
                                                                           settings.subjects_tr2_noncrush))
                                settings.subjects_tr_perf = np.vstack((settings.subjects_tr_perf,
                                                                       settings.subjects_tr2_perf))
                                settings.subjects_tr_angio = np.vstack((settings.subjects_tr_angio,
                                                                        settings.subjects_tr2_angio))
                                settings.subjects_tr_mri = np.vstack((settings.subjects_tr_mri,
                                                                        settings.subjects_tr2_mri))
                                settings.subjects_tr_segmentation = np.vstack((settings.subjects_tr_segmentation,
                                                                        settings.subjects_tr2_segmentation))
                                settings.subjects_tr2_crush=[]
                                settings.subjects_tr2_noncrush=[]
                                settings.subjects_tr2_perf=[]
                                settings.subjects_tr2_angio=[]
                                settings.subjects_tr2_mri=[]
                                settings.subjects_tr2_segmentation=[]
                                settings.train_queue.release()
                                self._fill_thread.resume()

                    else:
                        if len(settings.subjects_vl_segmentation) > settings.validation_totalimg_patch:
                            del settings.subjects_vl2_crush
                            del settings.subjects_vl2_noncrush
                            del settings.subjects_vl2_perf
                            del settings.subjects_vl2_angio
                            del settings.subjects_vl2_mri
                            del settings.subjects_vl2_segmentation
                            break
                        if (len(settings.subjects_vl_segmentation) == 0) &(len(settings.subjects_vl2_segmentation) > 0):
                            settings.subjects_vl_crush = settings.subjects_vl2_crush
                            settings.subjects_vl_noncrush = settings.subjects_vl2_noncrush
                            settings.subjects_vl_perf = settings.subjects_vl2_perf
                            settings.subjects_vl_angio = settings.subjects_vl2_angio
                            settings.subjects_vl_mri = settings.subjects_vl2_mri
                            settings.subjects_vl_segmentation = settings.subjects_vl2_segmentation
                            settings.subjects_vl2_crush = []
                            settings.subjects_vl2_non_crush = []
                            settings.subjects_vl2_perf = []
                            settings.subjects_vl2_angio = []
                            settings.subjects_vl2_mri = []
                            settings.subjects_vl2_segmentation = []


                            print('settings.subjects_vl lEN: %d' % (len(settings.subjects_vl_segmentation)))
                        elif (len(settings.subjects_vl2_segmentation) > 0)&(len(settings.subjects_vl2_segmentation) > 0):
                            settings.subjects_vl_crush = np.vstack((settings.subjects_vl_crush,
                                                                    settings.subjects_vl2_crush))
                            settings.subjects_vl_noncrush = np.vstack((settings.subjects_vl_noncrush,
                                                                       settings.subjects_vl2_noncrush))
                            settings.subjects_vl_perf = np.vstack((settings.subjects_vl_perf,
                                                                   settings.subjects_vl2_perf))
                            settings.subjects_vl_angio = np.vstack((settings.subjects_vl_angio,
                                                                    settings.subjects_vl2_angio))
                            settings.subjects_vl_mri = np.vstack((settings.subjects_vl_mri,
                                                                    settings.subjects_vl2_mri))
                            settings.subjects_vl_segmentation = np.vstack((settings.subjects_vl_segmentation,
                                                                    settings.subjects_vl2_segmentation))
                            settings.subjects_vl2 = []


                            print('settings.subjects_vl lEN2: %d' % (len(settings.subjects_vl_segmentation)))

                        if len(settings.subjects_vl_segmentation)<self.validation_sample_no:
                            if self._fill_thread.paused==True:
                                self._fill_thread.resume()
                                # time.sleep(2)
                        else:
                            # self.mutex.release()
                            self.finish_thread()
                finally:
                    a=1
                    # settings.mutex.release()
                    time.sleep(3)
                    # print('release2')


            # self.pause()


    def pop_from_queue(self):
        return self.queue.get()

    def pause(self):
        # print('pause read ')

        # If in sleep, we acquire immediately, otherwise we wait for thread
        # to release condition. In race, worker will still see self.paused
        # and begin waiting until it's set back to False

        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        # print('resume read ')
        if self.paused:
            # Notify so thread will wake after lock released
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

    def finish_thread(self):
        self.pause()

