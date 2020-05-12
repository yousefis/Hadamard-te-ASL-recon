import threading
import time

import functions.utils.settings as settings


class _extractor_thread(threading.Thread):
    def __init__ (self, _image_class,
                   patch_window, label_patchs_size,
                   mutex, is_training, vl_sample_no=0):
        threading.Thread.__init__(self)

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.mutex = mutex
        self.patch_window=patch_window
        self.label_patchs_size=label_patchs_size
        if is_training:
            self._image_class=_image_class
        else:
            self._image_class_vl=_image_class

        self.is_training=is_training
        self.validation_sample_no=vl_sample_no


    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()

                try:
                    if self.is_training:

                        self._image_class.read_patche_online_from_image_bunch_tr()
                    else:

                        if len(settings.subjects_vl_angio) < settings.validation_totalimg_patch:
                            self._image_class_vl.read_patche_online_from_image_bunch_vl()
                finally:
                    time.sleep(2)






    def pop_from_queue(self):
        return self.queue.get()

    def pause(self):
        # If in sleep, we acquire immediately, otherwise we wait for thread
        # to release condition. In race, worker will still see self.paused
        # and begin waiting until it's set back to False
        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        # Notify so thread will wake after lock released
        if self.paused :
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

    def finish_thread(self):
        self.pause()