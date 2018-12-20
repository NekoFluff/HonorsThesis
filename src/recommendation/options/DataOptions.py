class DataOptions():

    def __init__(self, root_folder_path):
        self.tmp_folder_path = root_folder_path + "/tmp/"
        self.raw_folder_path = self.tmp_folder_path + "/raw/"
        self.models_folder_path = self.tmp_folder_path + "/models/"
        self.cache_folder_path = self.tmp_folder_path + "/cache/"
        self.graphs_folder_path = self.tmp_folder_path + "/graphs/"
