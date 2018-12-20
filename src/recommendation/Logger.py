from datetime import datetime

class Logger():
    '''A universal logger. Use default_logger which is instantiated at the bottom of Logger.py
    '''
    last_checkpoint = datetime.now()

    def get_time_difference_since_last_checkpoint(self, now_checkpoint, reset_checkpoint = False):
        '''Gets the time difference between the passed in datetime 'now_checkpoint' and the stored 'last_checkpoint'

        now_checkpoint: datetime. Should represent the datetime of now.
        reset_checkpoint: Boolean. If enabled, the checkpoint will be updated to the checkpoint (datetime) passed in. 
        '''
        difference = now_checkpoint - self.last_checkpoint

        if reset_checkpoint:
            self.set_checkpoint(now_checkpoint)

        return difference

    def set_checkpoint(self, new_checkpoint = None):
        '''Sets the checkpoint to the most passed in time.

        If the new_checkpoint parameter is None, it will update to datetime.now()
        '''
        self.last_checkpoint = datetime.now() if new_checkpoint is None else new_checkpoint
        
    def log_time(self, reset_checkpoint = True):
        '''Prints out the current time as well as the time since the last checkpoint (print)

        reset_checkpint: Boolean. If true, the checkpoint will be set to the current time, thereby resetting the time since last checkpoint to 0.
        '''
        now_checkpoint = datetime.now()
        time_difference = self.get_time_difference_since_last_checkpoint(now_checkpoint, reset_checkpoint)
        time_string = "{:%m-%d %H:%M}   {:d}s".format(now_checkpoint, time_difference.seconds)
        print("{:_^100s}".format(time_string))

    def log(self, output, with_time=False, reset_checkpoint=True):
        '''Logs an output string with the time optionally
        
        output: The string to output
        with_time: Boolean. If true, the logger will output the time
        reset_checkpoint: Only matters if with_time is true. If with_time is true, the time will be reset
        '''
        if with_time:
            self.log_time(reset_checkpoint)
            
        print(output)
    
default_logger = Logger()
    