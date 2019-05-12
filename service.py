import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
from train import main as train_main
from os import mkdir, chdir
from os.path import abspath, dirname


class TrainServerSvc (win32serviceutil.ServiceFramework):
    _svc_name_ = "ZedSegTrain"
    _svc_display_name_ = "Train ZED segmentation"

    def __init__(self,args):
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
        self.main()

    @staticmethod
    def main():
        work_dir = dirname(abspath(__file__))
        chdir(work_dir)
        servicemanager.LogInfoMsg('Module directory is %s' % work_dir)
        train_main(with_gui=False)


if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(TrainServerSvc)
