import threading
import time

import functions.utils.settings as settings


class fill_thread(threading.Thread):
    def __init__ (self,data, _image_class,
                  sample_no,total_sample_no,label_patchs_size,
                  mutex,is_training,patch_extractor,fold,
                  mixedup=False):
        """
            Thread for moving images to RAM.

            This thread moves the images to RAM for train and validation process simultaneously fot making this process co-occurrence.

            Parameters
            ----------
            arg1 : int
                Description of arg1
            arg2 : str
                Description of arg2

            Returns
            -------
            nothing


        """
        threading.Thread.__init__(self)

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self._image_class=_image_class
        self.sample_no=sample_no
        self.label_patchs_size=label_patchs_size

        self.mutex=mutex
        self.is_training=is_training
        self.total_sample_no=total_sample_no
        self.patch_extractor=patch_extractor
        self.fold=fold

        self.Kill=False
        self.mixedup=mixedup


    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                try:
                    # thread should do the thing if not paused
                    if self.is_training==0: #validation!
                        delta=10
                        if len(settings.subjects_vl_angio) > settings.validation_totalimg_patch:
                            break
                        # if len(settings.bunch_GTV_patches_vl)>self.total_sample_no-delta:
                        #     break
                        if settings.vl_isread == False:
                            continue
                        self._image_class.read_bunch_of_images_from_dataset_vl(self.is_training)

                        self.patch_extractor.resume()

                    else:#train

                        # while len(settings.bunch_GTV_patches)>300:
                        #     print('sleep bunch_GTV_patches:%d', len(settings.bunch_GTV_patches))
                        #     time.sleep(3)
                        if settings.tr_isread == False:
                            continue
                        if self.mixedup:
                            self._image_class.read_bunch_of_images_from_dataset_mixedup_tr(self.is_training) # for training
                        else:
                            self._image_class.read_bunch_of_images_from_dataset_tr(self.is_training) # for training
                        self.patch_extractor.resume()
                finally:
                    a=1
                    # self.mutex.release()

                    time.sleep(2)
                    # print('realsed 1')

                # print('do the thing')
            # self.pause()




    def pop_from_queue(self):
        return self.queue.get()
    def kill_thread(self):
        self.Kill=True

    def pause(self):
        # print('pause fill ')

        # If in sleep, we acquire immediately, otherwise we wait for thread
        # to release condition. In race, worker will still see self.paused
        # and begin waiting until it's set back to False
        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        # print('resume fill ')

        # Notify so thread will wake after lock released
        if self.paused :
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

