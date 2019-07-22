"""
Запуск 'train.py' как службы windows
"""
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import traceback
from train import main as train_main
from os import mkdir, chdir
from os.path import abspath, dirname


class TrainServerSvc (win32serviceutil.ServiceFramework):
    _svc_name_ = "ZedSegTrain"
    _svc_display_name_ = "Train ZED segmentation"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        try:
            self.main()
        except Exception as e:
            servicemanager.LogErrorMsg('Exception %s' % str(e))
            servicemanager.LogErrorMsg(traceback.format_exc())

    def check_stop(self):
        return win32event.WaitForSingleObject(self.hWaitStop, 0) == win32event.WAIT_OBJECT_0

    def main(self):
        work_dir = dirname(abspath(__file__))
        chdir(work_dir)
        servicemanager.LogInfoMsg('Module directory is %s' % work_dir)
        train_main(with_gui=False, check_stop=self.check_stop)


if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(TrainServerSvc)
